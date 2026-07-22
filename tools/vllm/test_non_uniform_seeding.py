# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import urllib.error
import urllib.request


BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("VLLM_MODEL")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("VLLM_REQUEST_TIMEOUT_SECONDS", "60"))
TOTAL_REQUESTS = int(os.environ.get("VLLM_TOTAL_REQUESTS", "32"))


def _request_json(method, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise AssertionError(f"{method} {path} failed with HTTP {error.code}: {response_body}") from error
    except urllib.error.URLError as error:
        raise AssertionError(f"{method} {path} failed against {BASE_URL}: {error}") from error


def _get_model():
    if MODEL:
        return MODEL

    response = _request_json("GET", "/v1/models")
    models = response.get("data", [])
    if not models:
        raise AssertionError(f"No models returned from {BASE_URL}/v1/models: {response}")

    return models[0]["id"]


def test_non_uniform_seeding_against_running_vllm():
    """
    Run with:
        python -m pytest tools/vllm/test_non_uniform_seeding.py

    Optional overrides:
        VLLM_BASE_URL=http://127.0.0.1:8000
        VLLM_MODEL=<model id>
        VLLM_API_KEY=<token>
        VLLM_TOTAL_REQUESTS=32
    """
    asyncio.run(_run_non_uniform_seeding_check())


async def _run_non_uniform_seeding_check():
    assert TOTAL_REQUESTS % 2 == 0, "VLLM_TOTAL_REQUESTS must be even"

    model = _get_model()
    seeds = []

    # Generate pattern: 1, 0, 2, 0, 3, 0...
    for seed in range(1, (TOTAL_REQUESTS // 2) + 1):
        seeds.append(seed)
        seeds.append(0)

    async def get_chat_response(seed_val):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Generate a list of 10 random colors."}],
            "max_tokens": 50,
            "temperature": 0.9,
            "seed": seed_val,
        }
        response = await asyncio.to_thread(_request_json, "POST", "/v1/chat/completions", payload)

        try:
            content = response["choices"][0]["message"]["content"].strip()
            response_id = response["id"]
        except (KeyError, IndexError, TypeError) as error:
            raise AssertionError(f"Unexpected chat completion response for seed {seed_val}: {response}") from error

        return {"seed": seed_val, "content": content, "id": response_id}

    # Fire all requests concurrently. This floods the server, forcing dynamic
    # batching if enabled on vLLM.
    results = await asyncio.gather(*(get_chat_response(seed) for seed in seeds))

    zero_seed_contents = [result["content"] for result in results if result["seed"] == 0]
    unique_seed_contents = [result["content"] for result in results if result["seed"] != 0]
    assert len(zero_seed_contents) == TOTAL_REQUESTS // 2
    assert len(unique_seed_contents) == TOTAL_REQUESTS // 2

    unique_zero_outputs = set(zero_seed_contents)
    assert len(unique_zero_outputs) == 1, (
        "Determinism Failed for seed=0.\n"
        f"Expected 1 unique output, found {len(unique_zero_outputs)}.\n"
        f"Outputs: {unique_zero_outputs}"
    )

    unique_varied_outputs = set(unique_seed_contents)
    assert len(unique_varied_outputs) == len(unique_seed_contents), (
        "Entropy Failed for non-zero seeds.\n"
        f"Expected {len(unique_seed_contents)} unique outputs, found {len(unique_varied_outputs)}.\n"
        f"Collisions detected in: {unique_seed_contents}"
    )


if __name__ == "__main__":
    test_non_uniform_seeding_against_running_vllm()
    print(f"PASS: non-uniform seeding check passed against {BASE_URL}")
