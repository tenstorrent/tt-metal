# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Advisor-challenger capture target for the shipped Phi-3.5 dense decoder."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import MethodType

import torch
import ttnn


TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # Preserve the advisor environment's ttnn package precedence.
    sys.path.append(TT_METAL_ROOT)

_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None
_PAGE_TABLE = None
_CURRENT_POSITIONS = None


def _capture_safe_decode_rope(self, query, key, current_positions, *, use_long_rope):
    """Shipped manual RoPE with its already-declared output layout made explicit.

    The runtime implementation queries ``query.memory_config()`` and
    ``key.memory_config()``. The tracer cannot resolve those queries before
    layout assignment, while both tensors are explicitly produced in
    ``L1_HEIGHT_SHARDED_MEMORY_CONFIG`` by ``nlp_create_qkv_heads_decode``.
    """

    cos_table = self.long_cos_decode if use_long_rope else self.short_cos_decode
    sin_table = self.long_sin_decode if use_long_rope else self.short_sin_decode
    rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
    cos = ttnn.reshape(
        ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT),
        [1, 1, self.batch, self.head_dim],
    )
    sin = ttnn.reshape(
        ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT),
        [1, 1, self.batch, self.head_dim],
    )
    cos = ttnn.transpose(cos, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin = ttnn.transpose(sin, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
    key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
    return (
        ttnn.to_memory_config(query, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
        ttnn.to_memory_config(key, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
    )


def _build(device):
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
        LAYER_IDX,
        _config,
        _page_table,
        _positions,
        _synthetic_state,
        _to_tt_decode,
    )
    from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
        OptimizationPolicy,
        OptimizedDecoder,
    )

    batch = 1
    max_context = 128
    config = _config()
    # This is the policy observed in the frozen incumbent runs and final
    # tt-perf-report CSV, expressed explicitly so capture cannot drift to a
    # constructor default.
    policy = OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        gate_up_weight_dtype=ttnn.bfloat4_b,
        down_weight_dtype=ttnn.bfloat4_b,
        kv_cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        mlp_math_fidelity=ttnn.MathFidelity.LoFi,
        decode_core_grid=(8, 1),
        qkv_in0_block_w=12,
        o_proj_in0_block_w=12,
        gate_up_in0_block_w=6,
        down_in0_block_w=16,
        gate_up_split_interleaved=True,
        separate_gate_up_projections=False,
        fused_paged_cache_update=True,
        explicit_decode_sdpa=False,
        fused_rope=False,
    )
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=batch,
        max_context=max_context,
        policy=policy,
    )
    decoder._decode_rope = MethodType(_capture_safe_decode_rope, decoder)
    hidden = torch.randn(
        batch,
        1,
        config.hidden_size,
        generator=torch.Generator().manual_seed(20260730),
    ).to(torch.bfloat16)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    return (
        decoder,
        key_cache,
        value_cache,
        _page_table(batch, max_context, device, permute=True),
        _positions([127], device),
        _to_tt_decode(hidden, device),
    )


def decode(hidden):
    # The pinned advisor dialect has handlers for the two constituent
    # paged_update_cache operations but not tt-metal's newer fused wrapper.
    # Lower only that wrapper while tracing; the decoder was constructed with
    # the shipped fused policy and all candidate-bearing layouts, shapes,
    # dtypes, matmuls, attention, and MLP remain identical.
    policy = _DECODER.policy
    is_advisor_proxy = type(hidden).__module__.startswith("ttnn_jit.")
    if is_advisor_proxy:
        _DECODER.policy = replace(policy, fused_paged_cache_update=False)
    try:
        return _DECODER.decode_forward(
            hidden,
            key_cache=_KEY_CACHE,
            value_cache=_VALUE_CACHE,
            page_table=_PAGE_TABLE,
            current_positions=_CURRENT_POSITIONS,
            use_long_rope=False,
        )
    finally:
        _DECODER.policy = policy


def make_inputs(device):
    global _DECODER
    global _KEY_CACHE
    global _VALUE_CACHE
    global _PAGE_TABLE
    global _CURRENT_POSITIONS

    (
        _DECODER,
        _KEY_CACHE,
        _VALUE_CACHE,
        _PAGE_TABLE,
        _CURRENT_POSITIONS,
        hidden,
    ) = _build(device)
    return (hidden,)
