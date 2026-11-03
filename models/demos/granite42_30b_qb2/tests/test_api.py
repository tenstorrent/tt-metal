# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Live serving checks for reasoning channels and partial scheduler batches."""

import concurrent.futures
import os

import httpx
import pytest

MODEL = "ibm-granite/granite-4.2-30b"


@pytest.fixture
def server_url():
    url = os.environ.get("GRANITE_SERVER_URL")
    if not url:
        pytest.skip("Set GRANITE_SERVER_URL for live serving checks")
    return url.rstrip("/")


def completion(
    url,
    *,
    thinking,
    prompt="What is 17 multiplied by 23? Work through the calculation and give the final integer.",
    seed=42,
):
    response = httpx.post(
        url + "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": thinking, "low_effort": False},
        },
        timeout=600,
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] > 0
    return choice["message"]


@pytest.mark.parametrize("thinking", [False, True])
def test_reasoning_and_final_answer_channels(server_url, thinking):
    message = completion(server_url, thinking=thinking)
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    assert content.strip()
    assert "<think>" not in content and "</think>" not in content
    if thinking:
        assert reasoning.strip()
    else:
        assert not reasoning.strip()


@pytest.mark.parametrize("batch", [5, 10])
def test_partial_batches_keep_requests_independent(server_url, batch):
    prompts = [f"Reply with only this word: item{i}" for i in range(batch)]
    before = completion(server_url, thinking=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
        messages = list(pool.map(lambda prompt: completion(server_url, thinking=False, prompt=prompt), prompts))
    assert len(messages) == batch
    assert all(f"item{i}" in message.get("content", "") for i, message in enumerate(messages))
    # A repeated greedy request after a batch transition must preserve its output.
    after = completion(server_url, thinking=False)
    assert before["content"] == after["content"]
    assert httpx.get(server_url + "/health", timeout=10).status_code == 200


def test_seeded_mixed_sampling_survives_slot_reordering(server_url):
    requests = [
        {
            "model": MODEL,
            "prompt": f"Count from {i}: ",
            "temperature": 0 if i % 3 == 0 else 0.8,
            "top_k": 1 if i % 3 == 0 else 32,
            "top_p": 1.0 if i % 2 else 0.9,
            "seed": 42 + i,
            "max_tokens": 8,
        }
        for i in range(10)
    ]

    def sample(request):
        response = httpx.post(server_url + "/v1/completions", json=request, timeout=600)
        assert response.status_code == 200
        body = response.json()
        assert body["usage"]["completion_tokens"] > 0
        return body["choices"][0]["text"]

    baselines = [sample(request) for request in requests]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        reordered = list(pool.map(sample, reversed(requests)))
    assert reordered[::-1] == baselines
