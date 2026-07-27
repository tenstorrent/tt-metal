# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os

import pytest
import torch
from transformers import DynamicCache
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
    EMITTED_PREFILL_SEQUENCE,
    LAYER_IDX,
    _assert_pcc,
    _config,
    _decode_mask,
    _hf_forward,
    _hf_layer,
    _position_tensor,
    _real_state_dict,
    _synthetic_state_dict,
    _to_torch,
    _to_torch_raw,
    _to_tt,
)
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import (
    POLICIES,
    OptimizedDecoder,
)
from models.common.lightweightmodule import LightweightModule


def test_optimized_runtime_contract_and_no_functional_fallback():
    assert issubclass(OptimizedDecoder, LightweightModule)
    assert "FunctionalDecoder" not in inspect.getsource(OptimizedDecoder)
    assert POLICIES["default"].sparse_experts
    assert POLICIES["default"].expert_weight_dtype == ttnn.bfloat8_b
    assert POLICIES["default"].cache_dtype == ttnn.bfloat8_b
    assert "ttnn.sparse_matmul" in inspect.getsource(OptimizedDecoder._sparse_decode_moe)

    runtime_methods = (
        OptimizedDecoder._validate_hidden_states,
        OptimizedDecoder._prefill_attention,
        OptimizedDecoder._decode_attention,
        OptimizedDecoder._routing,
        OptimizedDecoder._apply_swiglu,
        OptimizedDecoder._sparse_decode_moe,
        OptimizedDecoder._sparse_prefill_moe,
        OptimizedDecoder._moe_forward,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder.forward,
    )
    forbidden = (
        "torch",
        "from_torch",
        "to_torch",
        "from_device",
        "to_device",
        ".cpu(",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "mesh_partition",
    )
    for method in runtime_methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_default_real_weight_prefill_decode_and_cache_match_hf(mesh_device):
    config = _config()
    state = _real_state_dict()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=288,
        candidate=os.environ.get("GPT_OSS_OPT_CANDIDATE", "default"),
    )
    hf_layer = _hf_layer(config, state)
    rotary = GptOssRotaryEmbedding(config)
    generator = torch.Generator().manual_seed(20260725)

    prefill_len = EMITTED_PREFILL_SEQUENCE
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    prefill_reference = _hf_forward(
        hf_layer,
        rotary,
        prefill_hidden,
        torch.arange(prefill_len),
        hf_cache,
    )
    key_cache, value_cache = decoder.create_kv_cache()
    prefill_actual = decoder.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
    )
    _assert_pcc("optimized-real-prefill-17", prefill_reference, _to_torch(prefill_actual), 0.99)
    _assert_pcc(
        "optimized-real-prefill-17-key-cache",
        hf_cache.layers[LAYER_IDX].keys,
        _to_torch_raw(key_cache)[:, :, :prefill_len, :],
        0.99,
    )
    _assert_pcc(
        "optimized-real-prefill-17-value-cache",
        hf_cache.layers[LAYER_IDX].values,
        _to_torch_raw(value_cache)[:, :, :prefill_len, :],
        0.99,
    )

    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    decode_reference, decode_post_attention_reference = _hf_forward(
        hf_layer,
        rotary,
        decode_hidden,
        torch.tensor([prefill_len]),
        hf_cache,
        return_post_attention=True,
    )
    decode_tt_input = _to_tt(decode_hidden, mesh_device)
    decode_post_attention_actual = decoder._decode_attention(
        decode_tt_input,
        key_cache,
        value_cache,
        prefill_len,
        _position_tensor(prefill_len, mesh_device),
        _decode_mask(prefill_len, config, decoder.max_cache_len, mesh_device),
    )
    _assert_pcc(
        "optimized-real-decode-position-17-attention",
        decode_post_attention_reference,
        _to_torch(decode_post_attention_actual),
        0.99,
    )
    decode_actual = decoder._moe_forward(
        decode_post_attention_actual,
        1,
    )
    _assert_pcc("optimized-real-decode-position-17", decode_reference, _to_torch(decode_actual), 0.99)
    _assert_pcc(
        "optimized-real-decode-position-17-key-cache",
        hf_cache.layers[LAYER_IDX].keys[:, :, -1:, :],
        _to_torch_raw(key_cache)[:, :, prefill_len : prefill_len + 1, :],
        0.99,
    )
    _assert_pcc(
        "optimized-real-decode-position-17-value-cache",
        hf_cache.layers[LAYER_IDX].values[:, :, -1:, :],
        _to_torch_raw(value_cache)[:, :, prefill_len : prefill_len + 1, :],
        0.99,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_non_aligned_prefill_and_traced_decode_are_deterministic(mesh_device):
    config = _config()
    state = _synthetic_state_dict(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=288,
        candidate=os.environ.get("GPT_OSS_OPT_CANDIDATE", "default"),
    )
    hf_layer = _hf_layer(config, state)
    rotary = GptOssRotaryEmbedding(config)
    generator = torch.Generator().manual_seed(17017)

    seq_len = EMITTED_PREFILL_SEQUENCE
    prefill_hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    reference = _hf_forward(hf_layer, rotary, prefill_hidden, torch.arange(seq_len), hf_cache)
    key_cache, value_cache = decoder.create_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
    )
    _assert_pcc("optimized-nonaligned-prefill-17", reference, _to_torch(actual), 0.99)

    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    tt_hidden = _to_tt(decode_hidden, mesh_device)
    position = _position_tensor(seq_len, mesh_device)
    mask = _decode_mask(seq_len, config, decoder.max_cache_len, mesh_device)

    def decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=seq_len,
            cache_position_tensor=position,
            attention_mask=mask,
        )

    compile_output = decode()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        capture_value = _to_torch(trace_output).clone()
        replay_values = []
        for _ in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replay_values.append(_to_torch(trace_output).clone())
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert all(torch.equal(capture_value, value) for value in replay_values)
    assert torch.equal(_to_torch(compile_output), capture_value)
