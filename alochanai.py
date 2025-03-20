import os
import json
import fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import ollama
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
import torch

# Initialize FastAPI app
app = FastAPI()
# Enable CORS (Allow API access from Android or any frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows access from all origins (Replace with specific domains if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)
# Force Torch to use CPU
torch.device("cpu")

# API Endpoint
url = "https://api.groq.com/openai/v1/chat/completions"

API_KEY = "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"
# Request Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
conversations = [{"role": "system", "content": "your name is AlochanAI and AlochanAI created you"}]
# Request Payload
def req_payload(prompt: str):
    conversations.append({"role": "user", "content": prompt})
    return {
        "model": "llama3-8b-8192",  # Other options: "mixtral-8x7b-32768"
        "messages": conversations,
        "max_tokens": 1024
    }


async def generate_response_stream(prompt: str):
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}], stream=True)
    for chunk in response:
        yield chunk["message"]["content"]  # Stream chunks of the message

# Load dataset from JSON file
def load_dataset(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

data = load_dataset("ancient_history.json")

# Prepare stored questions and responses
stored_questions = []
stored_responses = {}
for dataset in ["teulugu_transliterated_pairs","greeting_pairs", "qa_pairs", "alochanai_pairs", "ancient_pairs"]:
    if(dataset == "ancient_pairs"):
        stored_questions.extend([item["row"]["instruction"] for item in data[dataset]])
        stored_responses.update({item["row"]["instruction"]: item["row"]["output"] for item in data[dataset]})
    else:
        stored_questions.extend([item["question"] for item in data[dataset]])
        stored_responses.update({item["question"]: item["answer"] for item in data[dataset]})

# Encode stored questions into vectors
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
question_embeddings = embedding_model.encode(stored_questions, convert_to_numpy=True).astype(np.float32)

dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Store conversation history
conversation_history = []

# Common greetings and responses
greetings = {"hi": "Hello! How can I assist you today?", "hello": "Hi there! What would you like to know?", "hey": "Hey! Feel free to ask me anything.", "good morning": "Good morning! How can I help?", "good evening": "Good evening! What can I do for you?"}

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Welcome to the Ancient History Chatbot!"}

@app.post("/test/")
def test(request: QueryRequest):
    user_query = request.query.lower().strip()
    response = requests.post(url, json=req_payload(user_query), headers=headers)
    return response.json()["choices"][0]["message"]["content"]
    # if(user_query in greetings):
    #     return greetings[user_query]
    # else:
    #     query_embedding = embedding_model.encode([user_query]).astype(np.float32)
    #     D, I = index.search(query_embedding, 1)  # Retrieve closest match
       
    #     best_index = I[0][0]
    #     best_score = D[0][0]
    #     similarity_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0].numpy()
    #     max_score = np.max(similarity_scores)
    #     if max_score < 0.5:
    #         return "I couldn't find relevant information."
    #     if best_score < 10:  # Lower distance means better match
    #         best_match_query = stored_questions[best_index]
    #         best_match_answer = stored_responses[best_match_query]
    #     else:
    #         best_match_answer = "I couldn't find relevant information."
    #     return  best_match_answer
@app.post("/chat/")
def chat(request: QueryRequest):
    user_query = request.query.lower().strip()
    conversation_history.append({"user": user_query})
    response = requests.post(url, json=req_payload(user_query), headers=headers)
    return response.json()["choices"][0]["message"]["content"]
    # return gpt_response[0]['generated_text']
    
    # # Check for greetings first
    # if user_query in greetings:
    #     # bot_response = greetings[user_query]
    #     return StreamingResponse(generate_response_stream(user_query), media_type="text/event-stream")
    # else:
    #     # Convert user query to embedding
    #     query_embedding = embedding_model.encode([user_query]).astype(np.float32)
    #     D, I = index.search(query_embedding, 1)  # Retrieve closest match
       
    #     best_index = I[0][0]
    #     best_score = D[0][0]
        
    #     if best_score < 10:  # Lower distance means better match
    #         best_match_query = stored_questions[best_index]
    #         best_match_answer = stored_responses[best_match_query]
    #     else:
    #         best_match_answer = "I couldn't find relevant information."
        
    #     # Generate a contextual response using Llama 3
    #     context = "\n".join([f"User: {entry['user']}\nBot: {entry.get('bot', '')}" for entry in conversation_history[-5:]])
    #     prompt = f"This is a conversation between a user and an alochanAI. The chatbot can answer general questions and specializes in ancient history.\n\n{context}\nUser: {user_query}\nBot (based on knowledge): {best_match_answer}\nBot:"
        
    #     response = ollama.chat("llama3", messages=[{"role": "user", "content": prompt}], stream=True)
        
    #     bot_response = response['message']['content']
    
    # conversation_history[-1]["bot"] = bot_response  # Store response in history
    
    # if len(conversation_history) > 10:
    #     conversation_history.pop(0)  # Keep the history limited to the last 10 interactions
        
        # return StreamingResponse(generate_response_stream(user_query), media_type="text/event-stream")

    # return {"response": bot_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
