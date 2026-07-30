# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 shard-advisor capture of the shipped Phi-3.5 optimized decoder."""

from __future__ import annotations

import os
import sys
from types import MethodType

import torch
import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

BATCH = 32
MAX_CONTEXT = 128

_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None
_PAGE_TABLE = None
_CURRENT_POSITIONS = None


def _build(device):
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
        LAYER_IDX,
        _config,
        _page_table,
        _positions,
        _real_state,
        _to_tt_decode,
    )
    from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
        OptimizationPolicy,
        OptimizedDecoder,
    )

    config = _config()
    # This is the policy frozen in incumbent.json and selected_candidate_results.csv.
    # It is explicit so advisor-environment TTNN objects are used, rather than
    # serializing constructor defaults from the tt-metal runtime.
    policy = OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        gate_up_weight_dtype=ttnn.bfloat4_b,
        down_weight_dtype=ttnn.bfloat4_b,
        kv_cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_math_fidelity=ttnn.MathFidelity.LoFi,
        down_math_fidelity=ttnn.MathFidelity.LoFi,
        decode_core_count=16,
        in0_block_w_qkv=6,
        in0_block_w_o=6,
        in0_block_w_gate_up=6,
        in0_block_w_down=16,
        use_explicit_prefill_programs=False,
        use_explicit_decode_sdpa=True,
        split_decode_qkv=False,
        split_decode_gate_up=True,
    )
    decoder = OptimizedDecoder.from_state_dict(
        _real_state(),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
        optimization_policy=policy,
    )

    def capture_decode_rope(self, query, key, current_positions, *, use_long_rope):
        """Shipped RoPE with its statically known head-split output layout.

        The runtime method queries ``tensor.memory_config()`` merely to restore
        the L1-height-sharded layout requested from nlp_create_qkv_heads_decode.
        A traced tensor has no assigned layout yet, so state that same declared
        configuration directly for capture.
        """
        cos_table = self.long_cos if use_long_rope else self.short_cos
        sin_table = self.long_sin if use_long_rope else self.short_sin
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, 1, self.batch, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, self.batch, self.head_dim])
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        query = self._apply_rope(query, cos, sin)
        key = self._apply_rope(key, cos, sin)
        return (
            ttnn.to_memory_config(query, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
            ttnn.to_memory_config(key, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
        )

    decoder._decode_rope = MethodType(capture_decode_rope, decoder)
    hidden = torch.randn(
        BATCH,
        1,
        config.hidden_size,
        generator=torch.Generator().manual_seed(20260730),
    ).to(torch.bfloat16)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    return (
        decoder,
        key_cache,
        value_cache,
        _page_table(BATCH, MAX_CONTEXT, device, permute=True),
        _positions([33] * BATCH, device),
        _to_tt_decode(hidden, device),
    )


def decode(hidden):
    return _DECODER.decode_forward(
        hidden,
        key_cache=_KEY_CACHE,
        value_cache=_VALUE_CACHE,
        page_table=_PAGE_TABLE,
        current_positions=_CURRENT_POSITIONS,
        use_long_rope=False,
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE, _PAGE_TABLE, _CURRENT_POSITIONS
    (
        _DECODER,
        _KEY_CACHE,
        _VALUE_CACHE,
        _PAGE_TABLE,
        _CURRENT_POSITIONS,
        hidden,
    ) = _build(device)
    return (hidden,)
