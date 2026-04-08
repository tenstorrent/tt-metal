#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Quick test script for BGE-Large-EN-v1.5 vLLM server client.

Usage:
    python test_client.py
"""

import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    sys.exit(1)

# Server configuration
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def test_health():
    """Test server health."""
    import requests

    try:
        response = requests.get(f"{BASE_URL.replace('/v1', '')}/ping")
        if response.status_code == 200:
            print("✓ Server is healthy")
            return True
        else:
            print(f"✗ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print(f"  Make sure the server is running on {BASE_URL}")
        return False


def test_single_embedding():
    """Test single text embedding."""
    print("\n" + "=" * 60)
    print("Test 1: Single Text Embedding")
    print("=" * 60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        response = client.embeddings.create(model=MODEL_NAME, input="This is a test sentence for BGE embedding.")

        embedding = response.data[0].embedding
        print(f"✓ Success!")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_batch_embedding():
    """Test batch embedding."""
    print("\n" + "=" * 60)
    print("Test 2: Batch Embedding")
    print("=" * 60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    texts = [
        "Artificial intelligence is transforming technology.",
        "Machine learning enables computers to learn.",
        "Deep learning uses neural networks.",
    ]

    try:
        response = client.embeddings.create(model=MODEL_NAME, input=texts)

        print(f"✓ Success!")
        print(f"  Number of embeddings: {len(response.data)}")
        for i, item in enumerate(response.data):
            print(f"  Embedding {i+1}: dimension={len(item.embedding)}")
        print(f"  Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_semantic_similarity():
    """Test semantic similarity calculation."""
    print("\n" + "=" * 60)
    print("Test 3: Semantic Similarity")
    print("=" * 60)

    import numpy as np
    from numpy.linalg import norm

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    # Query and documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language used for data science.",
        "Deep learning uses neural networks with multiple layers.",
    ]

    try:
        # Get embeddings
        query_emb = client.embeddings.create(model=MODEL_NAME, input=query).data[0].embedding

        doc_embs = client.embeddings.create(model=MODEL_NAME, input=documents).data

        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (norm(a) * norm(b))

        query_vec = np.array(query_emb)
        similarities = []

        print(f"Query: {query}\n")
        print("Document similarities:")
        for i, doc_emb in enumerate(doc_embs):
            doc_vec = np.array(doc_emb.embedding)
            sim = cosine_similarity(query_vec, doc_vec)
            similarities.append((sim, i))
            print(f"  [{sim:.4f}] {documents[i]}")

        # Find most similar
        similarities.sort(reverse=True)
        print(f"\n✓ Most similar document: {documents[similarities[0][1]]}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BGE-Large-EN-v1.5 vLLM Server Client Test")
    print("=" * 60)

    # Check server health
    if not test_health():
        print("\n⚠ Server is not available. Please start the server first.")
        return

    # Run tests
    results = []
    results.append(("Single Embedding", test_single_embedding()))
    results.append(("Batch Embedding", test_batch_embedding()))
    results.append(("Semantic Similarity", test_semantic_similarity()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Check the output above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
