# Client Usage Guide for BGE-Large-EN-v1.5 vLLM Server

This guide shows how to interact with the BGE embedding model served via vLLM's OpenAI-compatible API.

## Server Status

The server is running on `http://0.0.0.0:8000` (or `http://localhost:8000` from the same machine).

**Available Endpoint**: `/v1/embeddings` (POST)

## Quick Start

### 1. Using curl

```bash
# Single text embedding
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "BAAI/bge-large-en-v1.5",
        "input": "This is a test sentence."
    }'

# Multiple texts (batch)
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "BAAI/bge-large-en-v1.5",
        "input": [
            "First sentence to embed.",
            "Second sentence to embed.",
            "Third sentence to embed."
        ]
    }'
```

### 2. Using Python (OpenAI SDK)

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM doesn't require real API key
)

# Single text embedding
response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input="This is a test sentence."
)

print(f"Embedding dimension: {len(response.data[0].embedding)}")
print(f"First 5 values: {response.data[0].embedding[:5]}")

# Multiple texts (batch)
response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input=[
        "First sentence to embed.",
        "Second sentence to embed.",
        "Third sentence to embed."
    ]
)

print(f"Number of embeddings: {len(response.data)}")
for i, item in enumerate(response.data):
    print(f"Embedding {i+1} dimension: {len(item.embedding)}")
```

### 3. Using Python (requests library)

```python
import requests
import json

url = "http://localhost:8000/v1/embeddings"
headers = {"Content-Type": "application/json"}

# Single text
data = {
    "model": "BAAI/bge-large-en-v1.5",
    "input": "This is a test sentence."
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

print(f"Status: {response.status_code}")
print(f"Embedding dimension: {len(result['data'][0]['embedding'])}")
print(f"Usage: {result['usage']}")

# Multiple texts
data = {
    "model": "BAAI/bge-large-en-v1.5",
    "input": [
        "First sentence to embed.",
        "Second sentence to embed.",
        "Third sentence to embed."
    ]
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

print(f"Number of embeddings: {len(result['data'])}")
```

### 4. Using JavaScript/Node.js

```javascript
const fetch = require('node-fetch'); // or use native fetch in Node 18+

async function getEmbedding(text) {
    const response = await fetch('http://localhost:8000/v1/embeddings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: 'BAAI/bge-large-en-v1.5',
            input: text,
        }),
    });

    const data = await response.json();
    return data.data[0].embedding;
}

// Usage
getEmbedding('This is a test sentence.')
    .then(embedding => {
        console.log('Embedding dimension:', embedding.length);
        console.log('First 5 values:', embedding.slice(0, 5));
    });
```

## Response Format

### Success Response

```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.123, -0.456, 0.789, ...],
            "index": 0
        }
    ],
    "model": "BAAI/bge-large-en-v1.5",
    "usage": {
        "prompt_tokens": 6,
        "total_tokens": 6
    }
}
```

### Batch Response (Multiple Inputs)

```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.123, -0.456, ...],
            "index": 0
        },
        {
            "object": "embedding",
            "embedding": [0.234, -0.567, ...],
            "index": 1
        },
        {
            "object": "embedding",
            "embedding": [0.345, -0.678, ...],
            "index": 2
        }
    ],
    "model": "BAAI/bge-large-en-v1.5",
    "usage": {
        "prompt_tokens": 18,
        "total_tokens": 18
    }
}
```

## Common Use Cases

### 1. Semantic Search

```python
from openai import OpenAI
import numpy as np
from numpy.linalg import norm

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Embed query
query = "What is machine learning?"
query_embedding = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input=query
).data[0].embedding

# Embed documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a programming language.",
    "Deep learning uses neural networks.",
]

doc_embeddings = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input=documents
).data

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

query_vec = np.array(query_embedding)
for i, doc_emb in enumerate(doc_embeddings):
    doc_vec = np.array(doc_emb.embedding)
    similarity = cosine_similarity(query_vec, doc_vec)
    print(f"Document {i+1}: {similarity:.4f} - {documents[i]}")
```

### 2. Batch Processing

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Process large batch
texts = [f"Sentence {i} for embedding." for i in range(100)]

# vLLM handles batching automatically
response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input=texts
)

print(f"Processed {len(response.data)} embeddings")
```

### 3. Error Handling

```python
from openai import OpenAI
import requests

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

try:
    response = client.embeddings.create(
        model="BAAI/bge-large-en-v1.5",
        input="Test"
    )
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
```

## Troubleshooting

### Common Errors

1. **404 Not Found** - Check the URL:
   - ✅ Correct: `http://localhost:8000/v1/embeddings`
   - ❌ Wrong: `http://localhost:8000//v1/embeddings` (double slash)

2. **Connection Refused** - Make sure the server is running:
   ```bash
   curl http://localhost:8000/ping
   ```

3. **Model Not Found** - Verify the model name:
   ```bash
   curl http://localhost:8000/v1/models
   ```

4. **Timeout** - Increase timeout for large batches:
   ```python
   client = OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="dummy",
       timeout=60.0  # 60 seconds
   )
   ```

## Testing the Server

### Health Check

```bash
curl http://localhost:8000/ping
```

### List Available Models

```bash
curl http://localhost:8000/v1/models
```

### Test Embedding Endpoint

```bash
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "BAAI/bge-large-en-v1.5",
        "input": "Hello, world!"
    }' | jq '.data[0].embedding | length'
```

Expected output: `1024` (embedding dimension for BGE-large)

## Performance Tips

1. **Batch Requests**: Send multiple texts in a single request for better throughput
2. **Connection Pooling**: Reuse HTTP connections when making multiple requests
3. **Async Requests**: Use async/await for concurrent requests

```python
import asyncio
from openai import AsyncOpenAI

async def get_embeddings_async(texts):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    tasks = [
        client.embeddings.create(
            model="BAAI/bge-large-en-v1.5",
            input=text
        )
        for text in texts
    ]

    results = await asyncio.gather(*tasks)
    return [r.data[0].embedding for r in results]

# Usage
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = asyncio.run(get_embeddings_async(texts))
```

## API Reference

### POST /v1/embeddings

**Request Body:**
```json
{
    "model": "BAAI/bge-large-en-v1.5",  // Required
    "input": "string or array of strings",  // Required
    "encoding_format": "float"  // Optional: "float" or "base64"
}
```

**Response:**
- Status: 200 OK
- Body: JSON with embeddings array and usage statistics

**Limits:**
- Max sequence length: 384 tokens
- Max batch size: 256 (configurable via `--max-batch-size`)
