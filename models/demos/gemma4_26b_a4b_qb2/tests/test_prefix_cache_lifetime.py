# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Remote scheduler lifetime and growing-context experiments."""

import json
from time import perf_counter
from uuid import uuid4

import pytest
import httpx

from models.demos.gemma4_26b_a4b_qb2.tests.test_api import completions, sampled_ids
from models.demos.gemma4_26b_a4b_qb2.tests.test_prefix_caching import _metric_totals


@pytest.mark.timeout(600)
def test_live_trace_chat_transitions(gemma_server_url, record_property):
    messages = [{"role": "user", "content": (
        "I am learning about compilers. A compiler reads source code, recognizes names, "
        "checks types, and produces executable instructions. It needs to remember information "
        "about each variable and function while processing many references to those names. "
        "In one short sentence, explain why a dictionary or hash table is useful here."
    )}]
    salt = f"trace-chat-{uuid4()}"

    def chat(history, cache_salt):
        response = httpx.post(
            f"{gemma_server_url}/v1/chat/completions",
            json={
                "model": "google/gemma-4-26B-A4B-it", "messages": history,
                "max_tokens": 64, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                "seed": 9472, "cache_salt": cache_salt,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        assert content and any(c.isalpha() for c in content), result
        return result

    cold = chat(messages, salt)
    cached = chat(messages, salt)
    record_property("cold_chat", json.dumps(cold))
    record_property("cached_chat", json.dumps(cached))
    before = _metric_totals(gemma_server_url)
    for repeat in range(3):
        response = chat(messages, salt)
        assert response["choices"][0]["message"]["content"] == cached["choices"][0]["message"]["content"]
        record_property(f"cached_repeat_{repeat}", json.dumps(response))
    after = _metric_totals(gemma_server_url)
    assert after["prefix_cache_hits_total"] > before["prefix_cache_hits_total"]
    growing = messages + [cached["choices"][0]["message"], {
        "role": "user", "content": "Give one specific example of a variable name and the information stored for it."
    }]
    record_property("growing_chat", json.dumps(chat(growing, salt)))
    record_property("independent_cold_chat", json.dumps(chat(messages, salt + "-cold")))


def _request(url, request):
    return completions(url, [request], token_ids=True)[0]


def _probs(response):
    return response["choices"][0]["logprobs"]["top_logprobs"]


@pytest.mark.timeout(600)
def test_live_lifetime_divergence_and_eviction(gemma_server_url, record_property):
    salt = f"lifetime-{uuid4()}"
    prompt = [100 + i % 71 for i in range(1025)]
    request = {"prompt": prompt, "max_tokens": 1, "logprobs": 5, "cache_salt": salt}
    cold = _request(gemma_server_url, request)
    warm = _request(gemma_server_url, request)
    divergent = {**request, "prompt": prompt[:-1] + [999]}
    divergent_first = _request(gemma_server_url, divergent)
    assert _probs(_request(gemma_server_url, divergent)) == _probs(divergent_first)
    assert _probs(_request(gemma_server_url, request)) == _probs(warm)
    # The runner uses max_model_len=4096 for this suite. Distinct content and
    # salts force physical-page reuse without changing another server's cache.
    for index in range(12):
        _request(gemma_server_url, {
            **request, "prompt": [100 + (i + index + 1) % 71 for i in range(1025)],
            "cache_salt": f"{salt}-pressure-{index}",
        })
    before = _metric_totals(gemma_server_url)
    rebuilt = _request(gemma_server_url, request)
    after = _metric_totals(gemma_server_url)
    computed = after["request_prefill_kv_computed_tokens_sum"] - before["request_prefill_kv_computed_tokens_sum"]
    record_property("post_pressure_computed_tokens", computed)
    assert computed == len(prompt), "Pressure did not evict the original conversation"
    assert _probs(rebuilt) == _probs(cold), "Reused pages changed cold reconstruction"
    assert _probs(_request(gemma_server_url, request)) == _probs(warm)


@pytest.mark.timeout(600)
def test_live_lifetime_device_sampling_repeat(gemma_server_url, record_property):
    # No logprobs or penalties: exercise the device sampler, not host fallback.
    request = {
        "prompt": "Explain why a dictionary lookup is useful in a compiler. " * 30,
        "max_tokens": 32, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
        "seed": 9472, "cache_salt": f"device-repeat-{uuid4()}",
    }
    completions(gemma_server_url, [request])  # cold arithmetic is a separate contract
    first = completions(gemma_server_url, [request])[0]
    for repeat in range(3):
        before = _metric_totals(gemma_server_url)
        started = perf_counter()
        second = completions(gemma_server_url, [request])[0]
        wall = perf_counter() - started
        after = _metric_totals(gemma_server_url)
        assert after["prefix_cache_hits_total"] > before["prefix_cache_hits_total"]
        assert first["choices"][0]["text"] == second["choices"][0]["text"]
        assert first["usage"]["completion_tokens"] == second["usage"]["completion_tokens"] == 32
        record_property(f"repeat_{repeat}_wall_seconds", wall)
        for name in before:
            record_property(f"repeat_{repeat}_{name}", after[name] - before[name])
        decode = after["request_decode_time_seconds_sum"] - before["request_decode_time_seconds_sum"]
        assert decode > 0
        record_property(f"repeat_{repeat}_decode_tokens_per_second", 31 / decode)
    record_property("device_response", json.dumps(second))


@pytest.mark.timeout(600)
@pytest.mark.parametrize("prefix_length", [4096, 8192, 16384])
def test_live_growing_127_token_suffix(gemma_server_url, prefix_length, record_property):
    prompt = [100 + i % 71 for i in range(prefix_length + 127)]
    request = {"prompt": prompt, "max_tokens": 1, "logprobs": 5}
    cold_reference = None
    cached_reference = None
    # The first conversation warms both geometries and performs the server's
    # full-logit replay diagnostic. Fresh salts preserve a 127-token suffix on
    # every measured conversation, rather than turning repeats into 31 tokens.
    for repeat in range(4):
        salt = f"growing-{prefix_length}-{repeat}-{uuid4()}"
        _request(gemma_server_url, {**request, "prompt": prompt[:prefix_length + 1], "cache_salt": salt})
        before = _metric_totals(gemma_server_url)
        started = perf_counter()
        cached = _request(gemma_server_url, {**request, "cache_salt": salt})
        cached_wall = perf_counter() - started
        middle = _metric_totals(gemma_server_url)
        started = perf_counter()
        cold = _request(gemma_server_url, {**request, "cache_salt": salt + "-cold"})
        cold_wall = perf_counter() - started
        after = _metric_totals(gemma_server_url)
        assert middle["request_prefill_kv_computed_tokens_sum"] - before["request_prefill_kv_computed_tokens_sum"] == 127
        assert after["request_prefill_kv_computed_tokens_sum"] - middle["request_prefill_kv_computed_tokens_sum"] == len(prompt)
        if repeat == 0:
            cold_reference, cached_reference = _probs(cold), _probs(cached)
            record_property("cold_response", json.dumps(cold))
            record_property("cached_response", json.dumps(cached))
            continue
        assert _probs(cold) == cold_reference and _probs(cached) == cached_reference
        record_property(f"repeat_{repeat}_cold_seconds", cold_wall)
        record_property(f"repeat_{repeat}_cached_seconds", cached_wall)
        record_property(f"repeat_{repeat}_greedy_equal", sampled_ids(cold) == sampled_ids(cached))
        for name in before:
            record_property(f"repeat_{repeat}_cached_{name}", middle[name] - before[name])
            record_property(f"repeat_{repeat}_cold_{name}", after[name] - middle[name])
