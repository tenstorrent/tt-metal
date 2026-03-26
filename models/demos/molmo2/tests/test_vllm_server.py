# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test script for Molmo2 vLLM server integration.

Usage:
    1. Start the vLLM server:
       cd /path/to/tt-inference-server
       python run.py --model Molmo2-8B --workflow vllm-server --device t3k

    2. Run this test script:
       python models/demos/molmo2/tests/test_vllm_server.py

    Or with pytest:
       pytest models/demos/molmo2/tests/test_vllm_server.py -v
"""

import base64
import os
import sys
from pathlib import Path

import pytest
import requests

# Default server URL
SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MOLMO_MODEL_NAME", "allenai/Molmo2-8B")

# Test image path (use the demo dog.jpg)
SCRIPT_DIR = Path(__file__).parent
TEST_IMAGE_PATH = SCRIPT_DIR.parent / "demo" / "dog.jpg"


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for API request."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def check_server_health() -> bool:
    """Check if the vLLM server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def send_chat_completion(
    messages: list,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> dict:
    """Send a chat completion request to the vLLM server."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


@pytest.fixture(scope="module")
def server_running():
    """Check if server is running before tests."""
    if not check_server_health():
        pytest.skip(
            f"vLLM server not running at {SERVER_URL}. "
            "Start with: python run.py --model Molmo2-8B --workflow vllm-server --device t3k"
        )


class TestMolmo2TextOnly:
    """Test text-only completions."""

    def test_simple_question(self, server_running):
        """Test a simple text-only question."""
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        result = send_chat_completion(messages, max_tokens=50)

        assert "choices" in result
        assert len(result["choices"]) > 0
        content = result["choices"][0]["message"]["content"]
        assert content  # Should have some response
        print(f"Response: {content}")

    def test_capital_question(self, server_running):
        """Test another text question."""
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        result = send_chat_completion(messages, max_tokens=50)

        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert "Paris" in content or "paris" in content.lower()
        print(f"Response: {content}")


class TestMolmo2Image:
    """Test image + text completions."""

    @pytest.fixture
    def image_base64(self):
        """Get base64 encoded test image."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")
        return encode_image_base64(str(TEST_IMAGE_PATH))

    def test_image_description(self, server_running, image_base64):
        """Test describing an image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        result = send_chat_completion(messages, max_tokens=100)

        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert content
        # Should mention dog/animal since we're using dog.jpg
        print(f"Response: {content}")

    def test_image_question(self, server_running, image_base64):
        """Test asking a question about an image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "What animal is in this image?"},
                ],
            }
        ]
        result = send_chat_completion(messages, max_tokens=50)

        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert content
        print(f"Response: {content}")


class TestMolmo2Sequential:
    """Test sequential requests to ensure no state corruption."""

    @pytest.fixture
    def image_base64(self):
        """Get base64 encoded test image."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")
        return encode_image_base64(str(TEST_IMAGE_PATH))

    def test_text_after_image(self, server_running, image_base64):
        """Test text request after image request - should not produce garbage."""
        # First: image request
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        result1 = send_chat_completion(image_messages, max_tokens=50)
        content1 = result1["choices"][0]["message"]["content"]
        print(f"Image response: {content1}")
        assert content1

        # Second: text request (should work correctly, not produce garbage)
        text_messages = [{"role": "user", "content": "What is 5 + 5?"}]
        result2 = send_chat_completion(text_messages, max_tokens=50)
        content2 = result2["choices"][0]["message"]["content"]
        print(f"Text response: {content2}")
        assert content2
        # Should not be garbage like "!!!!" or repeated tokens
        assert "!" * 4 not in content2

    def test_multiple_text_requests(self, server_running):
        """Test multiple sequential text requests."""
        questions = [
            "What is 1 + 1?",
            "What color is the sky?",
            "Name a famous scientist.",
        ]

        for question in questions:
            messages = [{"role": "user", "content": question}]
            result = send_chat_completion(messages, max_tokens=50)
            content = result["choices"][0]["message"]["content"]
            print(f"Q: {question}")
            print(f"A: {content}")
            assert content


def run_manual_tests():
    """Run tests manually without pytest."""
    print("=" * 60)
    print("Molmo2 vLLM Server Manual Tests")
    print("=" * 60)

    # Check server
    print("\n1. Checking server health...")
    if not check_server_health():
        print(f"ERROR: Server not running at {SERVER_URL}")
        print("Start with: python run.py --model Molmo2-8B --workflow vllm-server --device t3k")
        return False

    print("   Server is healthy!")

    # Test text
    print("\n2. Testing text-only completion...")
    try:
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        result = send_chat_completion(messages, max_tokens=50)
        content = result["choices"][0]["message"]["content"]
        print(f"   Response: {content}")
        print("   PASS!")
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    # Test image
    print("\n3. Testing image + text completion...")
    if TEST_IMAGE_PATH.exists():
        try:
            image_base64 = encode_image_base64(str(TEST_IMAGE_PATH))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "What is in this image?"},
                    ],
                }
            ]
            result = send_chat_completion(messages, max_tokens=100)
            content = result["choices"][0]["message"]["content"]
            print(f"   Response: {content}")
            print("   PASS!")
        except Exception as e:
            print(f"   FAIL: {e}")
            return False
    else:
        print(f"   SKIP: Test image not found at {TEST_IMAGE_PATH}")

    # Test sequential
    print("\n4. Testing sequential requests (text after image)...")
    try:
        # Text request after image
        messages = [{"role": "user", "content": "What is the capital of Japan?"}]
        result = send_chat_completion(messages, max_tokens=50)
        content = result["choices"][0]["message"]["content"]
        print(f"   Response: {content}")
        if "!!!!" in content or content.count("!") > 10:
            print("   FAIL: Got garbage output (state corruption)")
            return False
        print("   PASS!")
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        pytest.main([__file__, "-v"])
    else:
        success = run_manual_tests()
        sys.exit(0 if success else 1)
