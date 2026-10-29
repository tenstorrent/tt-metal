# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Compare cold chat generation with cached replay and shared-prefix batches."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

MODEL = "ibm-granite/granite-4.2-30b"


def reset_cache(url):
    response = httpx.post(url + "/reset_prefix_cache", timeout=60)
    response.raise_for_status()
    assert response.json()["success"], "Prefix-cache reset must finish before the cold control"


def complete(url, prompt):
    response = httpx.post(
        url + "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False, "low_effort": False},
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    message = choice["message"]
    assert message.get("content", "").strip()
    return message, body["usage"]


@pytest.mark.parametrize("shared_repetitions", [128, 131])
def test_cached_prefix_matches_cold_generation_for_ten_users(shared_repetitions):
    url = os.environ.get("GRANITE_SERVER_URL")
    if not url:
        pytest.skip("Set GRANITE_SERVER_URL for live prefix-cache checks")
    url = url.rstrip("/")
    # Several full KV pages are identical; each final instruction is distinct.
    shared = "Reference notes: " + "The orchard has red apples and green pears. " * shared_repetitions
    prompts = [shared + f"\nReply with only this label: item{i}." for i in range(10)]
    cold = []
    for prompt in prompts:
        reset_cache(url)
        message, usage = complete(url, prompt)
        assert usage["prompt_tokens_details"]["cached_tokens"] == 0
        cold.append(message)

    reset_cache(url)
    prime, _ = complete(url, prompts[0])
    repeated, repeated_usage = complete(url, prompts[0])
    assert repeated == prime == cold[0]
    assert repeated_usage["prompt_tokens_details"]["cached_tokens"] > 0
    if shared_repetitions == 131:
        assert repeated_usage["prompt_tokens_details"]["cached_tokens"] % 128 != 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        warm = list(pool.map(lambda prompt: complete(url, prompt), prompts))
    for i, ((message, usage), expected) in enumerate(zip(warm, cold)):
        assert message == expected, f"Cached output differs from cold control for item{i}"
        assert f"item{i}" in message["content"]
        assert usage["prompt_tokens_details"]["cached_tokens"] > 0

    # Retain accounting and output hashes, without publishing prompts or text.
    report = {
        "concurrency": 10,
        "shared_repetitions": shared_repetitions,
        "hf_revision": "9e668ce1c538387ef24d3644e9b0606647762636",
        "prompt_mode": "chat",
        "temperature": 0,
        "thinking": False,
        "repeated_cached_tokens": repeated_usage["prompt_tokens_details"]["cached_tokens"],
        "requests": [
            {
                "id": i,
                "prompt_tokens": usage["prompt_tokens"],
                "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"],
                "output_sha256": hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest(),
                "matches_cold": message == cold[i],
            }
            for i, (message, usage) in enumerate(warm)
        ],
    }
    destination = Path(
        os.environ.get("GRANITE_PREFIX_REPORT", "generated/test_reports/granite42_30b_qb2/prefix-cache.json")
    )
    destination = destination.with_name(f"{destination.stem}-{shared_repetitions}{destination.suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    reset_cache(url)
