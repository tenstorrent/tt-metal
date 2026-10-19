# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focused prefix-cache contracts; execute in remote CI only."""

from time import perf_counter
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
import torch

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tests.test_api import completions, sampled_ids
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator
from models.demos.gemma4_26b_a4b_qb2.tt.generator_vllm import Gemma4ForCausalLM


@pytest.mark.parametrize("start,end", [(0, 31), (32, 33), (32, 63), (96, 129), (992, 1025), (1024, 1057)])
def test_generator_continuation_preserves_absolute_positions(monkeypatch, start, end):
    calls = []
    cache, tables = object(), object()
    generator = object.__new__(Gemma4Generator)

    def forward(ids, **kwargs):
        calls.append((ids.clone(), kwargs))
        return torch.zeros(1, 1, 1, 4)

    generator.model = SimpleNamespace(
        max_seq_len=2048,
        config=SimpleNamespace(vocab_size=16),
        upload=lambda value, *args: value,
        prefill_forward=forward,
    )
    generator._upload_page_tables = lambda value: value
    monkeypatch.setattr(ttnn, "reshape", torch.reshape)
    tokens = torch.arange(end).reshape(1, end)
    generator.prefill_forward(tokens, page_table=tables, kv_cache=cache, prompt_lens=[end], start_pos=[start])
    ids, kwargs = calls[0]
    assert ids.flatten().tolist() == list(range(start, end))
    assert kwargs["start_pos"] == start
    assert kwargs["page_table"] is tables and kwargs["kv_cache"] is cache
    assert kwargs["user_id"] == 0


@pytest.mark.parametrize("starts,ends", [([-1], [33]), ([33], [33]), ([34], [33]), ([0, 32], [33])])
def test_invalid_continuation_fails_before_upload(starts, ends):
    generator = object.__new__(Gemma4Generator)
    generator.model = SimpleNamespace(max_seq_len=2048)
    generator._upload_page_tables = Mock(side_effect=AssertionError("Invalid interval reached upload"))
    with pytest.raises(ValueError):
        generator.prefill_forward(
            torch.zeros(1, 64), page_table=None, kv_cache=None, prompt_lens=ends, start_pos=starts
        )


def test_host_sampling_forwards_cached_prefix():
    generator = Mock()
    adapter = Gemma4ForCausalLM(generator, 1)
    adapter._select_sampling_mode = Mock(return_value=True)
    adapter._read_host_logits = Mock(return_value="host logits")
    tables, cache = [torch.tensor([[7, 9]])], object()
    generator.model.layer_indices = [0]
    result = adapter.prefill_forward(
        torch.arange(33).reshape(1, 33), None, cache, [33], start_pos=[32], page_tables_per_layer=tables
    )
    assert result == "host logits"
    assert generator.prefill_forward.call_args.kwargs["start_pos"] == [32]
    assert generator.prefill_forward.call_args.kwargs["kv_cache"] is cache
    generator.serving_prefill_tokens.assert_not_called()
    generator.configure_sampling.assert_not_called()


def _cache_hits(url):
    response = httpx.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in response.text.splitlines()
        if line.startswith("vllm:prefix_cache_hits_total")
    ]
    assert values, "Prefix-cache hit counter missing"
    return sum(values)


@pytest.mark.parametrize("length", [33, 63, 65, 1023, 1025, 1057])
def test_live_cached_suffix_matches_cold_host_logits(gemma_server_url, length, record_property):
    # A unique salt isolates the cold reference without clearing another request's cache.
    # Raw token IDs provide exact page/window boundaries independently of tokenization.
    prefix = [100 + i % 71 for i in range(length)]
    request = {"prompt": prefix, "max_tokens": 1, "logprobs": 5}
    salt = f"gemma-prefix-contract-{length}-{uuid4()}"
    started = perf_counter()
    cold = completions(gemma_server_url, [{**request, "cache_salt": salt}], token_ids=True)[0]
    record_property("cold_wall_seconds", perf_counter() - started)
    hits_before = _cache_hits(gemma_server_url)
    started = perf_counter()
    warm = completions(gemma_server_url, [{**request, "cache_salt": salt}], token_ids=True)[0]
    record_property("cached_wall_seconds", perf_counter() - started)
    hit_delta = _cache_hits(gemma_server_url) - hits_before
    record_property("prefix_cache_hits", hit_delta)
    record_property("prompt_tokens", length)
    assert hit_delta > 0, "Request did not exercise a cache hit"
    assert sampled_ids(warm) == sampled_ids(cold)
    cold_probs = cold["choices"][0]["logprobs"]["top_logprobs"][0]
    warm_probs = warm["choices"][0]["logprobs"]["top_logprobs"][0]
    assert len(cold_probs) >= 5, "Server did not return the requested top-five logprobs"
    assert warm_probs.keys() == cold_probs.keys()
    for token, logprob in cold_probs.items():
        assert warm_probs[token] == pytest.approx(logprob, abs=0.05)

    # Diverge within the partial final block; compare to a separately salted cold request.
    divergent = {**request, "prompt": prefix[:-1] + [999]}
    cached_divergent = completions(gemma_server_url, [{**divergent, "cache_salt": salt}], token_ids=True)[0]
    cold_divergent = completions(
        gemma_server_url, [{**divergent, "cache_salt": salt + "-divergent-control"}], token_ids=True
    )[0]
    assert sampled_ids(cached_divergent) == sampled_ids(cold_divergent)


def test_live_growing_context(gemma_server_url, record_property):
    salt = f"gemma-growing-{uuid4()}"
    for length in (1025, 1057, 1153, 1281):
        request = {
            "prompt": [100 + i % 71 for i in range(length)],
            "max_tokens": 8,
            "temperature": 0,
        }
        started = perf_counter()
        cached = completions(gemma_server_url, [{**request, "cache_salt": salt}])[0]
        record_property(f"growing_{length}_cached_wall_seconds", perf_counter() - started)
        started = perf_counter()
        cold = completions(gemma_server_url, [{**request, "cache_salt": f"{salt}-cold-{length}"}])[0]
        record_property(f"growing_{length}_cold_wall_seconds", perf_counter() - started)
        assert cached["choices"][0]["text"] == cold["choices"][0]["text"]
