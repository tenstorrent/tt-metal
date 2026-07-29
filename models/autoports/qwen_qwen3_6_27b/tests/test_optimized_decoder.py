# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder
from models.common.lightweightmodule import LightweightModule


def test_optimized_runtime_is_independent_and_packed_by_default():
    assert issubclass(OptimizedDecoder, LightweightModule)
    source = inspect.getsource(OptimizedDecoder)
    assert "FunctionalDecoder" not in source
    assert POLICIES["default"].packed_qkv
    assert POLICIES["default"].packed_gate_up
    assert POLICIES["default"].attention_weight_dtype == ttnn.bfloat8_b
    assert POLICIES["default"].mlp_gate_up_dtype == ttnn.bfloat4_b
    assert POLICIES["default"].cache_dtype == ttnn.bfloat8_b
    assert "qkv_proj" in inspect.getsource(OptimizedDecoder._full_attention_projections)
    assert "mlp_gate_up" in inspect.getsource(OptimizedDecoder._mlp)


def test_optimized_runtime_has_no_host_or_functional_fallback():
    runtime_methods = (
        OptimizedDecoder._mlp,
        OptimizedDecoder._full_attention_projections,
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
    forbidden = (
        "torch",
        "from_torch",
        "to_torch",
        ".cpu(",
        "FunctionalDecoder",
        "NotImplementedError",
    )
    for method in runtime_methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__
