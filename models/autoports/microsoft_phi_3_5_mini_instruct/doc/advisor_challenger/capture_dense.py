# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Batch-32 shard-advisor capture for the shipped Phi-3.5 dense decoder layer."""

from __future__ import annotations

import os
import sys

import torch
import ttnn


TT_METAL_ROOT = os.environ.get("SHARD_ADVISE_TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # Keep the advisor environment's ttnn package ahead of the source tree.
    sys.path.append(TT_METAL_ROOT)

BATCH = 32
MAX_CONTEXT = 128

_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None
_PAGE_TABLE = None
_POSITIONS = None


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

    config = _config()
    # This is the executed policy frozen in incumbent.json, spelled out rather
    # than inherited from constructor defaults.
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
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
        optimization_policy=policy,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden = torch.randn(
        BATCH, 1, config.hidden_size, generator=torch.Generator().manual_seed(20260730)
    ).to(torch.bfloat16)
    return (
        decoder,
        key_cache,
        value_cache,
        _page_table(BATCH, MAX_CONTEXT, device, permute=True),
        _positions([33] * BATCH, device),
        _to_tt_decode(hidden, device),
    )


def decode(hidden):
    """Mirror shipped decode with statically declared RoPE layouts for tracing."""
    decoder = _DECODER
    residual = ttnn.to_memory_config(hidden, decoder.decode_residual_memory_config)
    normalized = decoder._decode_norm(residual, decoder.weights["input_norm"])
    fused = decoder._decode_linear(normalized, "qkv", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
    query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
        fused,
        num_heads=decoder.num_heads,
        num_kv_heads=decoder.num_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    rope_positions = ttnn.typecast(_POSITIONS, ttnn.uint32)
    cos = ttnn.embedding(rope_positions, decoder.short_cos, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(rope_positions, decoder.short_sin, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, 1, BATCH, decoder.head_dim])
    sin = ttnn.reshape(sin, [1, 1, BATCH, decoder.head_dim])
    query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
    key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
    query = decoder._apply_rope(query, cos, sin)
    key = decoder._apply_rope(key, cos, sin)
    query = ttnn.to_memory_config(query, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    key = ttnn.to_memory_config(key, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    ttnn.experimental.paged_update_cache(
        _KEY_CACHE, key, update_idxs_tensor=_POSITIONS, page_table=_PAGE_TABLE
    )
    ttnn.experimental.paged_update_cache(
        _VALUE_CACHE, value, update_idxs_tensor=_POSITIONS, page_table=_PAGE_TABLE
    )
    attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        query,
        _KEY_CACHE,
        _VALUE_CACHE,
        cur_pos_tensor=_POSITIONS,
        page_table_tensor=_PAGE_TABLE,
        scale=decoder.scale,
        program_config=decoder.decode_sdpa_program_config,
        compute_kernel_config=decoder.compute_configs["attention"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attended = ttnn.to_memory_config(attended, decoder._decode_concat_memory_config())
    attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=decoder.num_heads)
    attended = ttnn.to_memory_config(attended, decoder.decode_residual_memory_config)
    projected = decoder._decode_linear(attended, "o_proj", decoder.decode_residual_memory_config)
    return decoder._decode_mlp(
        ttnn.add(residual, projected, memory_config=decoder.decode_residual_memory_config)
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE, _PAGE_TABLE, _POSITIONS
    _DECODER, _KEY_CACHE, _VALUE_CACHE, _PAGE_TABLE, _POSITIONS, hidden = _build(device)
    return (hidden,)
