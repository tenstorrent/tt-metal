# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU guards for the exact-signature eager-prefill/decode reuse boundary."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.common.sampling.generator import SamplingParams


class CacheTensor:
    shape = (8192, 2, 32, 256)
    padded_shape = shape
    dtype = "bfloat8_b"
    layout = "tile"

    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address

    def memory_config(self):
        return "DRAM interleaved"


@pytest.mark.parametrize("setting,expected", [(None, True), ("0", False), ("1", True)])
def test_reuse_default_and_explicit_fallback(monkeypatch, setting, expected):
    monkeypatch.delenv("GEMMA4_AUTOPORT_TTFT_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("GEMMA4_BENCHMARK_CONTROL", raising=False)
    if setting is None:
        monkeypatch.delenv("GEMMA4_EAGER_PREFILL_DECODE_REUSE", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_EAGER_PREFILL_DECODE_REUSE", setting)
    adapter = AutoportGemma4ForCausalLM(SimpleNamespace(mesh=None), 32)
    assert adapter.eager_prefill_decode_reuse is expected


def setup():
    gen = Gemma4Generator.__new__(Gemma4Generator)
    gen.mesh = None
    gen.model = SimpleNamespace(max_seq_len=262144, layer_indices=(0, 1))
    gen.host_sampling = False
    gen.prefill_trace_enabled = True
    gen.prefill_prepared = None
    gen.trace_id = 41
    gen.batch = 1
    gen.active_slots = (0,)
    gen._trace_returns_logits = False
    gen._trace_sampled_mode = False
    gen.sampler = SimpleNamespace(_penalties_active=False, _log_probs_active=False)
    adapter = AutoportGemma4ForCausalLM(gen, 32)
    adapter.eager_prefill_decode_reuse = True
    tokens = torch.ones((1, 4096), dtype=torch.long)
    tables = (torch.zeros((1, 8192), dtype=torch.int32),) * 2
    cache = [(CacheTensor(100), CacheTensor(200)), (CacheTensor(300), CacheTensor(400))]
    gen.cache = cache
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    return adapter, gen, tokens, tables, cache, params


def test_key_ignores_token_and_page_values_but_tracks_nested_cache_ownership():
    adapter, _, tokens, tables, cache, params = setup()
    signature = lambda: adapter._eager_prefill_signature(tokens, tables, cache, [4096], params, None)
    key = signature()
    assert key is not None
    tokens.fill_(113)
    tables[0].fill_(127)
    assert signature() == key
    cache[0] = (CacheTensor(100), cache[0][1])
    assert signature() != key
    key = signature()
    cache[0][0].address = 500
    assert signature() != key


@pytest.mark.parametrize("change", ["length", "table_shape", "table_dtype", "token_dtype", "token_padding"])
def test_program_signature_changes_invalidate_key(change):
    adapter, _, tokens, tables, cache, params = setup()
    key = adapter._eager_prefill_signature(tokens, tables, cache, [4096], params, None)
    length = 4096
    if change == "length":
        tokens, length = torch.ones((1, 4097), dtype=torch.long), 4097
    elif change == "table_shape":
        tables = (torch.zeros((1, 8191), dtype=torch.int32),) * 2
    elif change == "table_dtype":
        tables = tuple(t.long() for t in tables)
    elif change == "token_dtype":
        tokens = tokens.int()
    else:
        tokens = torch.ones((1, 4128), dtype=torch.long)
    assert adapter._eager_prefill_signature(tokens, tables, cache, [length], params, None) != key


@pytest.mark.parametrize("change", ["disabled", "host", "sampled", "penalty", "logprobs", "resumed", "batch", "short"])
def test_unsupported_prefills_never_get_reuse_key(change, expect_error):
    adapter, _, tokens, tables, cache, params = setup()
    counts, lengths = None, [4096]
    if change == "disabled":
        adapter.eager_prefill_decode_reuse = False
    elif change == "host":
        params = None
    elif change == "sampled":
        params = replace(params, temperature=0.7)
    elif change == "penalty":
        params = replace(params, presence_penalty=0.5)
    elif change == "logprobs":
        params = replace(params, enable_log_probs=True)
        with expect_error(ValueError, "does not support this TP4 mesh"):
            adapter._eager_prefill_signature(tokens, tables, cache, lengths, params, counts)
        return
    elif change == "resumed":
        counts = torch.tensor([1])
    elif change == "batch":
        tokens, lengths = tokens.repeat(2, 1), [4096, 4096]
    else:
        tokens, lengths = tokens[:, :128], [128]
    assert adapter._eager_prefill_signature(tokens, tables, cache, lengths, params, counts) is None


def test_decode_reuse_requires_explicit_eager_permission_and_matching_live_bundle():
    adapter, gen, _, _, _, params = setup()
    assert not gen.can_reuse_serving_decode(params)
    assert gen.can_reuse_serving_decode(params, allow_eager_prefill=True)
    adapter._eager_decode_trace_id = 41
    assert adapter._has_eager_decode_trace()
    gen.trace_id = 42
    assert not adapter._has_eager_decode_trace()
    gen.trace_id = 41
    gen.active_slots = (1,)
    assert not adapter._has_eager_decode_trace()
    gen.active_slots = (0,)
    gen._trace_returns_logits = True
    assert not gen.can_reuse_serving_decode(params, allow_eager_prefill=True)


@pytest.mark.parametrize("length", [128, 4096])
def test_disabled_prefill_tracing_preserves_original_eager_fallback(length):
    adapter, gen, _, tables, cache, params = setup()
    gen.prefill_trace_enabled = False
    tokens = torch.ones((1, length), dtype=torch.long)
    assert adapter._eager_prefill_signature(tokens, tables, cache, [length], params, None) is None


def test_prefill_and_first_decode_keep_matching_trace_and_release_before_new_shape(monkeypatch):
    adapter, gen, tokens, tables, cache, params = setup()
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda value: [value])
    monkeypatch.setattr(ttnn, "to_torch", lambda value: value)
    gen.trace_id = None
    gen.prefill_trace_id = None
    gen.prefill_sampling_trace_id = None
    gen._release_trace = lambda: setattr(gen, "trace_id", None)
    configured, prefill_traces = [], []

    def configure(*args, _reuse_trace=False, **kwargs):
        configured.append(_reuse_trace)
        if not _reuse_trace:
            gen._release_trace()

    def prefill(*args, **kwargs):
        prefill_traces.append(gen.trace_id)
        return torch.tensor([7], dtype=torch.int32)

    def decode(*args, **kwargs):
        if gen.trace_id is None:
            gen.trace_id = 41
        return torch.tensor([8], dtype=torch.int32)

    gen.configure_sampling = configure
    gen.serving_prefill_tokens = prefill
    gen.decode_forward = decode
    gen.restore_sampling_state = Mock()

    def request(ids):
        output = adapter.prefill_forward(
            ids, tables[0], cache, [ids.shape[-1]], sampling_params=params, page_tables_per_layer=tables
        )
        return adapter.decode_forward(
            output,
            torch.tensor([ids.shape[-1]], dtype=torch.int32),
            tables[0],
            cache,
            sampling_params=params,
            page_tables_per_layer=tables,
            read_from_device=False,
        )

    request(tokens)
    request(tokens + 1)
    assert configured == [False, False, True, True]
    assert prefill_traces == [None, 41]
    request(torch.ones((1, 4097), dtype=torch.long))
    assert configured[-2:] == [False, False]
    assert prefill_traces[-1] is None
