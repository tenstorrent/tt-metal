# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focused prefix-cache contracts; execute in remote CI only."""

import json
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
from models.demos.gemma4_26b_a4b_qb2.tt.multichip_decoder import MultichipDecoder


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


@pytest.mark.parametrize("start,length", [(32, 8), (32, 31), (992, 32), (1008, 32)])
def test_shared_suffix_batch_keeps_absolute_request_mapping(monkeypatch, start, length):
    monkeypatch.setenv("GEMMA4_PREFIX_SHARED_BATCH", "1")
    monkeypatch.setattr(ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(ttnn, "repeat", lambda value, repeats: value.repeat(*repeats))
    layer = SimpleNamespace(
        _validate_kv_cache=Mock(),
        config=SimpleNamespace(max_position_embeddings=2048),
        positions_u32=torch.arange(2048).reshape(1, -1),
        positions_i32=torch.arange(2048).reshape(1, -1),
        decode_forward=Mock(return_value="output"),
    )
    table = torch.tensor([[7, 9, 11], [13, 15, 17]])
    result = MultichipDecoder._prefill_continuation(
        layer, torch.zeros(1, 1, length, 32), rope_mats=(torch.zeros(1, 1, 2048, 32),),
        page_table=table, kv_cache=object(), user_id=1, start_pos=start,
    )
    assert result == "output"
    args = layer.decode_forward.call_args.kwargs
    assert args["current_pos"].flatten().tolist() == list(range(start, start + length))
    assert args["cache_pos"].tolist() == list(range(start, start + length))
    assert args["page_table"].tolist() == [table[1].tolist()] * length


def test_traced_suffix_uses_explicit_tokens_without_sampling(monkeypatch):
    monkeypatch.setenv("GEMMA4_PREFIX_TRACE_SUFFIX", "1")
    monkeypatch.setattr(ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(ttnn, "clone", torch.clone)
    generator = object.__new__(Gemma4Generator)
    generator.model = SimpleNamespace(max_seq_len=2048, config=SimpleNamespace(vocab_size=16))
    generator._upload_page_tables = lambda value: value
    generator.decode_forward = Mock(return_value=torch.zeros(1, 1, 32, 4))
    generator._copy_prefix_logits = lambda value: value[:, :, :1, :].clone()
    cache, table = object(), torch.tensor([[7, 9]])
    generator.prefill_forward(
        torch.arange(63).reshape(1, 63), page_table=table, kv_cache=cache, prompt_lens=[63], start_pos=[32]
    )
    assert generator.decode_forward.call_count == 31
    for position, call in zip(range(32, 63), generator.decode_forward.call_args_list):
        assert call.args[0].item() == position
        assert call.args[1].item() == position
        assert call.kwargs["return_logits"] is True
        assert call.kwargs["page_table"] is table and call.kwargs["kv_cache"] is cache


def _metric_totals(url):
    response = httpx.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    totals = {}
    for metric in (
        "prefix_cache_hits_total",
        "time_to_first_token_seconds_sum",
        "request_prefill_kv_computed_tokens_sum",
        "generation_tokens_total",
        "request_decode_time_seconds_sum",
    ):
        values = [float(line.rsplit(" ", 1)[1]) for line in response.text.splitlines() if line.startswith(f"vllm:{metric}{{")]
        assert values, f"Metric missing: {metric}"
        totals[metric] = sum(values)
    return totals


def test_logits_replay_does_not_sample(monkeypatch):
    generator = object.__new__(Gemma4Generator)
    generator.mesh, generator.trace_id = object(), 42
    generator.trace_logits = object()
    generator.counters = {}
    generator.sampler = Mock()
    execute = Mock()
    monkeypatch.setattr(ttnn, "execute_trace", execute)
    assert generator._replay(return_logits=True) is generator.trace_logits
    execute.assert_called_once_with(generator.mesh, 42, cq_id=0, blocking=False)
    generator.sampler.sample.assert_not_called()
    assert generator.counters["model_replays"] == 1


def _cache_hits(url):
    return _metric_totals(url)["prefix_cache_hits_total"]


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
    record_property("cold_response", json.dumps(cold))
    record_property("cached_response", json.dumps(warm))
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


@pytest.mark.parametrize("length", [33, 63, 65, 1023, 1025, 1057])
def test_live_decode_contract_and_warmed_timing(gemma_server_url, length, record_property):
    """Diagnostic contract, not a replacement for cold-prefill parity above.

    The remote server probe asserts full-vocabulary same-cache decode parity
    on the first cold/warm requests. Later identical requests skip that probe.
    """
    request = {"prompt": [100 + i % 71 for i in range(length)], "max_tokens": 1, "logprobs": 5}
    salt = f"decode-contract-{length}-{uuid4()}"
    cold = completions(gemma_server_url, [{**request, "cache_salt": salt}], token_ids=True)[0]
    hits = _cache_hits(gemma_server_url)
    warm = completions(gemma_server_url, [{**request, "cache_salt": salt}], token_ids=True)[0]
    assert _cache_hits(gemma_server_url) > hits
    record_property("cold_prefill_equal", sampled_ids(cold) == sampled_ids(warm))
    record_property("cold_response", json.dumps(cold))
    record_property("cached_response", json.dumps(warm))
    for repeat in range(3):
        before = _metric_totals(gemma_server_url)
        started = perf_counter()
        cold = completions(gemma_server_url, [{**request, "cache_salt": f"{salt}-cold-{repeat}"}], token_ids=True)[0]
        record_property(f"repeat_{repeat}_cold_seconds", perf_counter() - started)
        middle = _metric_totals(gemma_server_url)
        started = perf_counter()
        cached = completions(gemma_server_url, [{**request, "cache_salt": salt}], token_ids=True)[0]
        record_property(f"repeat_{repeat}_cached_seconds", perf_counter() - started)
        after = _metric_totals(gemma_server_url)
        for name in before:
            record_property(f"repeat_{repeat}_cold_{name}", middle[name] - before[name])
            record_property(f"repeat_{repeat}_cached_{name}", after[name] - middle[name])
        computed = "request_prefill_kv_computed_tokens_sum"
        assert middle[computed] - before[computed] == length
        assert after[computed] - middle[computed] == length - (length - 1) // 32 * 32
        assert sampled_ids(cached) == sampled_ids(warm), "Cached repeat changed greedy output"
