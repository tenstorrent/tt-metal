# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Live serving contracts for Gemma4 on unmodified vLLM TT plugin main."""

import asyncio

import httpx
import pytest

MODEL = "google/gemma-4-31B-it"


def completions(url, requests, *, token_ids=False):
    async def run():
        async with httpx.AsyncClient(base_url=url, timeout=600) as client:

            async def send(parameters):
                payload = {"model": MODEL, "temperature": 0, "max_tokens": 15, "ignore_eos": True, **parameters}
                if token_ids:
                    payload.update(logprobs=0, return_tokens_as_token_ids=True)
                response = await client.post("/v1/completions", json=payload)
                response.raise_for_status()
                return response.json()

            return await asyncio.gather(*(send(parameters) for parameters in requests))

    return asyncio.run(run())


def sampled_ids(response):
    return [int(token.removeprefix("token_id:")) for token in response["choices"][0]["logprobs"]["tokens"]]


@pytest.mark.parametrize(
    "parameters",
    [
        {"temperature": 0.8, "seed": 42},
        {"temperature": 0.8, "seed": 2**31 - 1, "top_k": 32},
        {"temperature": 0.8, "seed": 0, "top_k": 1},
        {"temperature": 0, "seed": 42},
    ],
    ids=["default-top-k", "max-top-k-and-seed", "min-top-k-and-seed", "greedy"],
)
def test_supported_sampling_domain(gemma_server_url, parameters):
    response = httpx.post(
        f"{gemma_server_url}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Name one planet in our solar system."}],
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
            **parameters,
        },
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    assert result["choices"][0]["message"]["content"]
    assert result["usage"]["completion_tokens"] > 0


def test_seeded_page_growth_and_slot_reordering(gemma_server_url):
    # Logprobs and penalties would force host sampling and hide the device path.
    requests = [
        {
            "prompt": f"Continue this story about explorer {i}: " + "The path crossed a quiet forest. " * (i + 1),
            "max_tokens": 129 + i,
            "temperature": 0.8,
            "seed": 2**30 + 97 * i,
            "top_k": 20,
        }
        for i in range(4)
    ]
    first = completions(gemma_server_url, requests)
    second = completions(gemma_server_url, list(reversed(requests)))[::-1]
    for i, (a, b) in enumerate(zip(first, second)):
        assert a["usage"]["completion_tokens"] == b["usage"]["completion_tokens"] == 129 + i
        assert a["choices"][0]["text"] == b["choices"][0]["text"]


def test_mixed_sampling_parameters_are_request_local(gemma_server_url):
    parameters = [
        {"temperature": 0},
        {"temperature": 2.0, "top_k": 32},
        {"temperature": 0.5, "repetition_penalty": 3.0},
        {"temperature": 0.5, "presence_penalty": 2.0},
        {"temperature": 2.0, "top_k": 1},
        {"temperature": 0.01},
        {"temperature": 2.0, "top_k": 5},
        {"temperature": 0.5, "frequency_penalty": 2.0},
        {"temperature": 0.5, "repetition_penalty": 1.5, "presence_penalty": 1.0, "frequency_penalty": 1.0},
        {"temperature": 1.0, "top_p": 0.5},
    ]
    requests = [{"prompt": f"Count from {i}: ", "seed": 42 + i, "max_tokens": 5, **p} for i, p in enumerate(parameters)]
    first = completions(gemma_server_url, requests)
    second = completions(gemma_server_url, list(reversed(requests)))[::-1]
    assert [r["choices"][0]["text"] for r in first] == [r["choices"][0]["text"] for r in second]


def test_allowed_token_ids(gemma_server_url):
    requests = [{"prompt": "Allowed: ", "allowed_token_ids": [start, start + 1, start + 2]} for start in (10, 13)]
    responses = completions(gemma_server_url, requests, token_ids=True)
    for request, response in zip(requests, responses):
        ids = sampled_ids(response)
        assert ids
        assert set(ids) <= set(request["allowed_token_ids"])


def test_frequency_penalty_is_request_local(gemma_server_url):
    prompt = "a " * 100
    penalties = [0.0, 2.0] * 16
    requests = [{"prompt": prompt, "frequency_penalty": penalty} for penalty in penalties]
    mixed = completions(gemma_server_url, requests, token_ids=True)
    baselines = {}
    for penalty in (0.0, 2.0):
        baselines[penalty] = completions(
            gemma_server_url, [{"prompt": prompt, "frequency_penalty": penalty}] * 32, token_ids=True
        )
        for slot, requested_penalty in enumerate(penalties):
            if requested_penalty == penalty:
                assert sampled_ids(mixed[slot]) == sampled_ids(baselines[penalty][slot])
    assert sampled_ids(baselines[0.0][0]) != sampled_ids(baselines[2.0][0])


def test_logprobs(gemma_server_url):
    responses = completions(
        gemma_server_url,
        [
            {"prompt": f"Count from {i}: ", "max_tokens": 10, "logprobs": 5, "return_tokens_as_token_ids": True}
            for i in range(32)
        ],
    )
    for response in responses:
        logprobs = response["choices"][0]["logprobs"]
        assert logprobs["tokens"]
        assert all(5 <= len(row) <= 6 for row in logprobs["top_logprobs"])
        assert all(value is not None for value in logprobs["token_logprobs"])
