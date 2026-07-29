# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder
from models.common.lightweightmodule import LightweightModule


def test_optimized_runtime_is_independent_and_packed_by_default():
    assert issubclass(OptimizedDecoder, LightweightModule)
    assert "FunctionalDecoder" not in inspect.getsource(OptimizedDecoder)
    policy = POLICIES["default"]
    assert policy.packed_qkv
    assert policy.packed_gate_up
    assert policy.attention_weight_dtype == ttnn.bfloat8_b
    assert policy.mlp_gate_up_dtype == ttnn.bfloat4_b
    assert policy.mlp_down_dtype == ttnn.bfloat4_b
    assert policy.cache_dtype == ttnn.bfloat8_b
    assert policy.gate_up_ds_in0_block_w == 2
    assert policy.down_in0_block_w == 17
    assert policy.mlp_l1_chain
    assert '"packed_qkv"' in inspect.getsource(OptimizedDecoder._full_attention_decode)
    assert '"mlp_gate_up"' in inspect.getsource(OptimizedDecoder._mlp)


def test_optimized_runtime_has_no_host_or_functional_fallback():
    runtime_methods = (
        OptimizedDecoder._mlp,
        OptimizedDecoder._linear_attention_prefill,
        OptimizedDecoder._linear_attention_prefill_chunk,
        OptimizedDecoder._linear_attention_decode,
        OptimizedDecoder._full_attention_prefill,
        OptimizedDecoder._full_attention_decode,
        OptimizedDecoder._per_head_norm,
        OptimizedDecoder._per_head_norm_prefill,
        OptimizedDecoder._partial_rope_decode,
        OptimizedDecoder._partial_rope_prefill,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
    )
    forbidden = ("torch.", "ttnn.from_torch", "ttnn.to_torch", "NotImplementedError")
    for method in runtime_methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__
