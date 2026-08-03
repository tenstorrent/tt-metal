# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device CohereLabs/North-Mini-Code-1.0 decoder.

The functional decoder remains the correctness oracle.  This class owns its
weight materialization and every runtime operation.  It keeps the emitted
packed-QKV/paged-attention topology and replaces dense all-expert MoE decode
with routed ``ttnn.sparse_matmul`` execution.  Named construction-time
policies make precision, fidelity, sparse topology, and block geometry A/B
experiments reproducible without runtime host decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Mapping

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import ADVERTISED_CONTEXT, DEFAULT_PAGE_SIZE
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import MODEL_ID as FUNCTIONAL_MODEL_ID
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import (
    FunctionalDecoder,
    _load_expert_weights,
    _require_tensor,
    _rope_output_permutation,
)

MODEL_ID = FUNCTIONAL_MODEL_ID

OPTIMIZED_MOE_CHUNK = 32


@dataclass(frozen=True)
class OptimizationPolicy:
    attention_weight_dtype: object
    dense_gate_up_dtype: object
    dense_down_dtype: object
    cache_dtype: object
    attention_fidelity: object
    dense_gate_up_fidelity: object
    dense_down_fidelity: object
    decode_expert_gate_up_dtype: object
    decode_expert_down_dtype: object
    prefill_expert_gate_up_dtype: object
    prefill_expert_down_dtype: object
    decode_expert_gate_up_fidelity: object
    decode_expert_down_fidelity: object
    prefill_expert_gate_up_fidelity: object
    prefill_expert_down_fidelity: object
    sparse_experts: bool = True
    gate_up_grid: tuple[int, int] = (8, 2)
    gate_up_in0_block_w: int = 16
    gate_up_out_block_w: int = 3
    gate_up_subblock_w: int = 3
    down_grid: tuple[int, int] = (8, 2)
    down_in0_block_w: int = 24
    down_out_block_w: int = 4
    down_subblock_w: int = 4
    prefill_chunk_size: int = OPTIMIZED_MOE_CHUNK
    packed_expert_gate_up: bool = False
    dense_large_prefill_gate_up_grid: tuple[int, int] | None = (8, 8)
    dense_large_prefill_down_grid: tuple[int, int] | None = (8, 8)
    dense_large_prefill_gate_up_in0_block_w: int = 8
    dense_large_prefill_down_in0_block_w: int = 8
    dense_large_prefill_gate_up_subblock_w: int = 3
    dense_large_prefill_down_subblock_w: int = 4
    packed_dense_large_prefill: bool = False
    decode_sharded_residual: bool = False
    attention_dram_sharded: bool = False
    attention_dram_sharded_serving: bool = False
    decode_router_grid: tuple[int, int] | None = None
    decode_router_in0_block_w: int = 8
    decode_router_out_block_w: int = 2
    decode_router_subblock_w: int = 2
    decode_exact_nnz: int | None = None
    prefill_functional_router_compute: bool = False
    dense_large_prefill_chunk_size: int = 1024
    dense_large_prefill_functional_compute: bool = False
    packed_dense_mlp: bool = False
    dense_decode_unpacked_batch1: bool = False
    dense_decode_lofi_batch32: bool = False
    explicit_dense_decode_programs: bool = False
    dense_decode_gate_up_grid: tuple[int, int] = (8, 4)
    dense_decode_gate_up_in0_block_w: int = 2
    dense_decode_gate_up_out_block_w: int = 6
    dense_decode_gate_up_subblock_w: int = 6
    dense_decode_gate_up_interleaved_input: bool = False
    dense_decode_down_grid: tuple[int, int] = (8, 4)
    dense_decode_down_in0_block_w: int = 8
    dense_decode_down_out_block_w: int = 2
    dense_decode_down_subblock_w: int = 2
    dense_decode_down_dram_sharded: bool = False
    dense_decode_down_dram_sharded_batch1: bool = False


POLICIES = {
    "default": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_gate_up_dtype=ttnn.bfloat4_b,
        dense_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        dense_down_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_gate_up_dtype=ttnn.bfloat8_b,
        decode_expert_down_dtype=ttnn.bfloat8_b,
        prefill_expert_gate_up_dtype=ttnn.bfloat8_b,
        prefill_expert_down_dtype=ttnn.bfloat8_b,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        decode_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        packed_expert_gate_up=True,
    ),
    "sparse_bfp8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_gate_up_dtype=ttnn.bfloat8_b,
        dense_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        dense_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        dense_down_fidelity=ttnn.MathFidelity.HiFi2,
        decode_expert_gate_up_dtype=ttnn.bfloat8_b,
        decode_expert_down_dtype=ttnn.bfloat8_b,
        prefill_expert_gate_up_dtype=ttnn.bfloat8_b,
        prefill_expert_down_dtype=ttnn.bfloat8_b,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        decode_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "sparse_bf16": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_gate_up_dtype=ttnn.bfloat16,
        dense_down_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat16,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        dense_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        dense_down_fidelity=ttnn.MathFidelity.HiFi2,
        decode_expert_gate_up_dtype=ttnn.bfloat16,
        decode_expert_down_dtype=ttnn.bfloat16,
        prefill_expert_gate_up_dtype=ttnn.bfloat16,
        prefill_expert_down_dtype=ttnn.bfloat16,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        decode_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_attention": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        dense_gate_up_dtype=ttnn.bfloat4_b,
        dense_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        dense_down_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_gate_up_dtype=ttnn.bfloat4_b,
        decode_expert_down_dtype=ttnn.bfloat4_b,
        prefill_expert_gate_up_dtype=ttnn.bfloat8_b,
        prefill_expert_down_dtype=ttnn.bfloat8_b,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_down": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_gate_up_dtype=ttnn.bfloat4_b,
        dense_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        dense_down_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_gate_up_dtype=ttnn.bfloat8_b,
        decode_expert_down_dtype=ttnn.bfloat4_b,
        prefill_expert_gate_up_dtype=ttnn.bfloat8_b,
        prefill_expert_down_dtype=ttnn.bfloat8_b,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "dense_bf16_baseline": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_gate_up_dtype=ttnn.bfloat16,
        dense_down_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat16,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        dense_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        dense_down_fidelity=ttnn.MathFidelity.HiFi2,
        decode_expert_gate_up_dtype=ttnn.bfloat16,
        decode_expert_down_dtype=ttnn.bfloat16,
        prefill_expert_gate_up_dtype=ttnn.bfloat16,
        prefill_expert_down_dtype=ttnn.bfloat16,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        decode_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        sparse_experts=False,
    ),
    "sparse_bfp4_g8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_gate_up_dtype=ttnn.bfloat4_b,
        dense_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        dense_down_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_gate_up_dtype=ttnn.bfloat4_b,
        decode_expert_down_dtype=ttnn.bfloat4_b,
        prefill_expert_gate_up_dtype=ttnn.bfloat8_b,
        prefill_expert_down_dtype=ttnn.bfloat8_b,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        gate_up_grid=(4, 2),
        gate_up_in0_block_w=32,
        down_grid=(4, 4),
        down_in0_block_w=24,
    ),
    "sparse_bfp4_g32": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_gate_up_dtype=ttnn.bfloat4_b,
        dense_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        dense_down_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_gate_up_dtype=ttnn.bfloat4_b,
        decode_expert_down_dtype=ttnn.bfloat4_b,
        prefill_expert_gate_up_dtype=ttnn.bfloat8_b,
        prefill_expert_down_dtype=ttnn.bfloat8_b,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        gate_up_grid=(8, 4),
        gate_up_in0_block_w=8,
        down_grid=(8, 4),
        down_in0_block_w=12,
    ),
    "prefill_chunk32": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_gate_up_dtype=ttnn.bfloat4_b,
        dense_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        dense_down_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_gate_up_dtype=ttnn.bfloat4_b,
        decode_expert_down_dtype=ttnn.bfloat4_b,
        prefill_expert_gate_up_dtype=ttnn.bfloat8_b,
        prefill_expert_down_dtype=ttnn.bfloat8_b,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_chunk_size=32,
    ),
    "prefill_chunk64": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_gate_up_dtype=ttnn.bfloat4_b,
        dense_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        dense_down_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_gate_up_dtype=ttnn.bfloat4_b,
        decode_expert_down_dtype=ttnn.bfloat4_b,
        prefill_expert_gate_up_dtype=ttnn.bfloat8_b,
        prefill_expert_down_dtype=ttnn.bfloat8_b,
        decode_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_chunk_size=64,
    ),
}

# Each audit candidate is derived from the production policy so its name maps
# to exactly one changed variable (or one coherent program-geometry change).
POLICIES.update(
    {
        "attention_bfp8_lofi": dataclass_replace(POLICIES["default"]),
        "attention_bfp8_hifi2": dataclass_replace(POLICIES["default"], attention_fidelity=ttnn.MathFidelity.HiFi2),
        "cache_bfp8": dataclass_replace(POLICIES["default"], cache_dtype=ttnn.bfloat8_b),
        "cache_bf16": dataclass_replace(POLICIES["default"], cache_dtype=ttnn.bfloat16),
        "bfp4_attention": dataclass_replace(POLICIES["default"], attention_weight_dtype=ttnn.bfloat4_b),
        "prefill_bfp4": dataclass_replace(
            POLICIES["default"],
            prefill_expert_gate_up_dtype=ttnn.bfloat4_b,
            prefill_expert_down_dtype=ttnn.bfloat4_b,
            prefill_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            prefill_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "prefill_bfp8_hifi4": dataclass_replace(
            POLICIES["default"],
            prefill_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi4,
            prefill_expert_down_fidelity=ttnn.MathFidelity.HiFi4,
        ),
        "prefill_chunk1024": dataclass_replace(POLICIES["default"], prefill_chunk_size=1024),
        "packed_gate_up": dataclass_replace(POLICIES["default"]),
        "unpacked_gate_up": dataclass_replace(
            POLICIES["default"],
            packed_expert_gate_up=False,
            # Unpacked gate/up has Nt=24 rather than the packed Nt=48.
            # Use 12 rectangular receiver cores so every core owns exactly
            # two N tiles; sparse_matmul rejects idle multicast receivers.
            gate_up_grid=(6, 2),
            gate_up_out_block_w=2,
            gate_up_subblock_w=2,
        ),
        "bfp4_down": dataclass_replace(
            POLICIES["default"],
            decode_expert_down_dtype=ttnn.bfloat4_b,
            decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "sparse_bfp4": dataclass_replace(
            POLICIES["default"],
            decode_expert_gate_up_dtype=ttnn.bfloat4_b,
            decode_expert_down_dtype=ttnn.bfloat4_b,
            decode_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "sparse_bfp4_bf16_cache": dataclass_replace(
            POLICIES["default"],
            cache_dtype=ttnn.bfloat16,
            decode_expert_gate_up_dtype=ttnn.bfloat4_b,
            decode_expert_down_dtype=ttnn.bfloat4_b,
            decode_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "sparse_bfp8": dataclass_replace(
            POLICIES["default"],
            decode_expert_gate_up_dtype=ttnn.bfloat8_b,
            decode_expert_down_dtype=ttnn.bfloat8_b,
            decode_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
            decode_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        ),
        "sparse_bf16": dataclass_replace(
            POLICIES["default"],
            decode_expert_gate_up_dtype=ttnn.bfloat16,
            decode_expert_down_dtype=ttnn.bfloat16,
            prefill_expert_gate_up_dtype=ttnn.bfloat16,
            prefill_expert_down_dtype=ttnn.bfloat16,
            decode_expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
            decode_expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        ),
        "sparse_bfp4_block16": dataclass_replace(
            POLICIES["default"],
            gate_up_grid=(6, 2),
            gate_up_in0_block_w=16,
            gate_up_out_block_w=2,
            gate_up_subblock_w=2,
            down_grid=(8, 2),
            down_in0_block_w=24,
            down_out_block_w=2,
            down_subblock_w=2,
        ),
        # Final-policy BFP8/HiFi2 expert geometry cross-product.  Core counts
        # divide the packed gate/up Nt=48 and down Nt=64 exactly, avoiding
        # idle cores in sparse_matmul's rectangular multicast receiver set.
        "expert_bfp8_hifi2_geo_g24_g32": dataclass_replace(
            POLICIES["default"],
            gate_up_grid=(6, 4),
            gate_up_in0_block_w=16,
            gate_up_out_block_w=2,
            gate_up_subblock_w=2,
            down_grid=(8, 4),
            down_in0_block_w=24,
            down_out_block_w=2,
            down_subblock_w=2,
        ),
        "expert_bfp8_hifi2_geo_g16": dataclass_replace(
            POLICIES["default"],
            gate_up_grid=(8, 2),
            gate_up_in0_block_w=16,
            gate_up_out_block_w=3,
            gate_up_subblock_w=3,
            down_grid=(8, 2),
            down_in0_block_w=24,
            down_out_block_w=4,
            down_subblock_w=4,
        ),
        "expert_bfp8_hifi2_geo_g8": dataclass_replace(
            POLICIES["default"],
            gate_up_grid=(4, 2),
            gate_up_in0_block_w=32,
            gate_up_out_block_w=6,
            gate_up_subblock_w=3,
            down_grid=(4, 2),
            down_in0_block_w=24,
            down_out_block_w=8,
            down_subblock_w=4,
        ),
        "expert_bfp8_hifi2_geo_g32_safe": dataclass_replace(
            POLICIES["default"],
            gate_up_grid=(8, 4),
            gate_up_in0_block_w=8,
            gate_up_out_block_w=1,
            gate_up_subblock_w=1,
            down_grid=(8, 4),
            down_in0_block_w=12,
            down_out_block_w=1,
            down_subblock_w=1,
        ),
        # Explicit large-prefill 2-D matmul candidates.  Precision stays at
        # the final prefill BFP8/HiFi2 policy and all large intermediates stay
        # DRAM interleaved; only program geometry changes.
        "dense_prefill_2d_g8x4": dataclass_replace(
            POLICIES["default"],
            dense_large_prefill_gate_up_grid=(8, 4),
            dense_large_prefill_down_grid=(8, 4),
            dense_large_prefill_gate_up_in0_block_w=8,
            dense_large_prefill_down_in0_block_w=8,
            dense_large_prefill_gate_up_subblock_w=3,
            dense_large_prefill_down_subblock_w=4,
        ),
        "dense_prefill_2d_g8x8": dataclass_replace(
            POLICIES["default"],
            dense_large_prefill_gate_up_grid=(8, 8),
            dense_large_prefill_down_grid=(8, 8),
            dense_large_prefill_gate_up_in0_block_w=8,
            dense_large_prefill_down_in0_block_w=8,
            dense_large_prefill_gate_up_subblock_w=3,
            dense_large_prefill_down_subblock_w=4,
        ),
        "dense_prefill_packed_2d_g8x8": dataclass_replace(
            POLICIES["default"],
            packed_dense_large_prefill=True,
        ),
        "decode_sharded_residual_chain": dataclass_replace(
            POLICIES["default"],
            decode_sharded_residual=True,
        ),
        "attention_dram_sharded_chain": dataclass_replace(
            POLICIES["default"],
            decode_sharded_residual=True,
            attention_dram_sharded=True,
            attention_dram_sharded_serving=True,
        ),
        "router_decode_g2_block8_subblock2": dataclass_replace(
            POLICIES["default"],
            decode_router_grid=(2, 1),
            decode_router_in0_block_w=8,
            decode_router_out_block_w=2,
            decode_router_subblock_w=2,
        ),
        "selected_decode_chain": dataclass_replace(
            POLICIES["default"],
            dense_gate_up_dtype=ttnn.bfloat4_b,
            dense_down_dtype=ttnn.bfloat8_b,
            dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            dense_down_fidelity=ttnn.MathFidelity.LoFi,
            packed_dense_mlp=True,
            dense_decode_unpacked_batch1=True,
            dense_decode_lofi_batch32=True,
            explicit_dense_decode_programs=True,
            dense_decode_gate_up_grid=(8, 8),
            dense_decode_gate_up_out_block_w=3,
            dense_decode_gate_up_subblock_w=3,
            dense_decode_down_dram_sharded=True,
            decode_sharded_residual=True,
            attention_dram_sharded=True,
            attention_dram_sharded_serving=False,
            decode_exact_nnz=8,
            prefill_functional_router_compute=True,
        ),
    }
)

# Final batch-aware decode policy: the width-sharded residual/norm chain is
# correct and faster at both measured batches.  DRAM-sharded QKV/O is selected
# only at batch 1; serving batch 32 keeps the correct interleaved projections.
POLICIES["default"] = POLICIES["selected_decode_chain"]
POLICIES["bfp4_attention_selected_decode"] = dataclass_replace(
    POLICIES["default"],
    attention_weight_dtype=ttnn.bfloat4_b,
)
POLICIES["cache_bf16_selected_decode"] = dataclass_replace(
    POLICIES["default"],
    cache_dtype=ttnn.bfloat16,
)
POLICIES["sparse_bfp4_bf16_cache_selected_decode"] = dataclass_replace(
    POLICIES["default"],
    cache_dtype=ttnn.bfloat16,
    decode_expert_gate_up_dtype=ttnn.bfloat4_b,
    decode_expert_down_dtype=ttnn.bfloat4_b,
    decode_expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
    decode_expert_down_fidelity=ttnn.MathFidelity.LoFi,
)
POLICIES["batch1_exact_nnz8"] = dataclass_replace(
    POLICIES["default"],
    decode_exact_nnz=8,
)
POLICIES["batch1_dynamic_nnz_control"] = dataclass_replace(
    POLICIES["default"],
    decode_exact_nnz=None,
)
POLICIES["dense_bfp4_lofi_control"] = dataclass_replace(
    POLICIES["default"],
    dense_gate_up_dtype=ttnn.bfloat4_b,
    dense_down_dtype=ttnn.bfloat8_b,
    dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
    dense_down_fidelity=ttnn.MathFidelity.LoFi,
)
POLICIES["dense_bfp8_lofi_control"] = dataclass_replace(
    POLICIES["default"],
    dense_gate_up_dtype=ttnn.bfloat8_b,
    dense_down_dtype=ttnn.bfloat8_b,
    dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
    dense_down_fidelity=ttnn.MathFidelity.LoFi,
)
POLICIES["dense_packed_batch1_control"] = dataclass_replace(
    POLICIES["default"],
    dense_decode_unpacked_batch1=False,
)
POLICIES["dense_hifi2_decode_batch32_control"] = dataclass_replace(
    POLICIES["default"],
    dense_gate_up_dtype=ttnn.bfloat8_b,
    dense_down_dtype=ttnn.bfloat8_b,
    dense_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
    dense_down_fidelity=ttnn.MathFidelity.HiFi2,
    dense_decode_lofi_batch32=False,
)
POLICIES["dense_bfp8_hifi2_control"] = POLICIES["dense_hifi2_decode_batch32_control"]
POLICIES["dense_unpacked_bfp8_hifi2_control"] = dataclass_replace(
    POLICIES["default"],
    dense_gate_up_dtype=ttnn.bfloat8_b,
    dense_down_dtype=ttnn.bfloat8_b,
    dense_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
    dense_down_fidelity=ttnn.MathFidelity.HiFi2,
    dense_decode_lofi_batch32=False,
    packed_dense_mlp=False,
    dense_decode_gate_up_grid=(8, 6),
    dense_decode_gate_up_out_block_w=2,
    dense_decode_gate_up_subblock_w=2,
)
for _in0_block_w in (4, 8, 16, 32):
    POLICIES[f"dense_packed_gate_up_block{_in0_block_w}"] = dataclass_replace(
        POLICIES["default"],
        dense_decode_unpacked_batch1=False,
        dense_decode_gate_up_in0_block_w=_in0_block_w,
        dense_decode_gate_up_interleaved_input=True,
    )
    POLICIES[f"dense_unpacked_gate_up_block{_in0_block_w}"] = dataclass_replace(
        POLICIES["dense_unpacked_bfp8_hifi2_control"],
        dense_decode_gate_up_in0_block_w=_in0_block_w,
        dense_decode_gate_up_interleaved_input=True,
    )
POLICIES["dense_generic_program_control"] = dataclass_replace(
    POLICIES["default"],
    explicit_dense_decode_programs=False,
)
POLICIES["dense_down_g16_sub4"] = dataclass_replace(
    POLICIES["default"],
    dense_decode_down_grid=(8, 2),
    dense_decode_down_in0_block_w=12,
    dense_decode_down_out_block_w=4,
    dense_decode_down_subblock_w=4,
)
POLICIES["dense_down_g32_block8"] = dataclass_replace(
    POLICIES["default"],
    dense_decode_down_in0_block_w=8,
)
POLICIES["dense_down_g32_block12"] = dataclass_replace(
    POLICIES["default"],
    dense_decode_down_in0_block_w=12,
)
POLICIES["dense_down_g32_block16"] = dataclass_replace(
    POLICIES["default"],
    dense_decode_down_in0_block_w=16,
)
POLICIES["dense_down_g32_block24"] = dataclass_replace(
    POLICIES["default"],
    dense_decode_down_in0_block_w=24,
)
POLICIES["dense_down_dram_sharded"] = dataclass_replace(
    POLICIES["default"],
    dense_decode_down_dram_sharded=True,
    dense_decode_down_dram_sharded_batch1=True,
)
POLICIES["dense_gate_up_g64_sub3"] = dataclass_replace(
    POLICIES["default"],
    dense_decode_gate_up_grid=(8, 8),
    dense_decode_gate_up_out_block_w=3,
    dense_decode_gate_up_subblock_w=3,
)
# AutoFix controls for the exact real-weight serving-prefill delta.  The
# first changes only the router's accumulation policy while retaining the
# selected 1024-token expert matmuls.  The second reproduces the functional
# decoder's 32-token default-matmul numerics without calling its runtime.
POLICIES["prefill_functional_router_m1024"] = dataclass_replace(
    POLICIES["default"],
    prefill_functional_router_compute=True,
)
POLICIES["prefill_hifi4_router_control"] = dataclass_replace(
    POLICIES["default"],
    prefill_functional_router_compute=False,
)
POLICIES["prefill_functional_m32_control"] = dataclass_replace(
    POLICIES["default"],
    prefill_functional_router_compute=True,
    dense_large_prefill_chunk_size=OPTIMIZED_MOE_CHUNK,
    dense_large_prefill_functional_compute=True,
)


def _as_device_tensor(
    tensor,
    *,
    mesh_device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


def _compute_config(mesh_device, fidelity, *, fp32_dest_acc_en=False):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=not fp32_dest_acc_en,
    )


def _largest_divisor(value: int, ceiling: int = 8) -> int:
    """Return the largest positive divisor no greater than ``ceiling``."""

    for divisor in range(min(value, ceiling), 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


def _decode_sparse_nnz(policy: OptimizationPolicy, token_count: int, *, prefill: bool) -> int | None:
    """Return a fixed sparse count only for the candidate's exact batch-1 mask."""

    return policy.decode_exact_nnz if not prefill and token_count == 1 else None


def _dram_sharded_weight_memory_config(mesh_device, *, k: int, n: int):
    """Match the common 1-D attention module's single-device weight layout."""

    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        }
    )
    dram_cores = dram_grid_size.x * dram_grid_size.y
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * ttnn.TILE_SIZE * dram_cores
    shard_spec = ttnn.ShardSpec(
        dram_grid,
        (k, padded_n // dram_cores),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def _dram_sharded_decode_program_config(*, m: int, k: int, n: int, num_compute_cores: int):
    """Build the DRAM-sharded decode matmul used by common Attention1D."""

    k_tiles_per_core = k // (ttnn.TILE_SIZE * num_compute_cores)
    if k_tiles_per_core < 1:
        raise ValueError(f"DRAM-sharded matmul needs at least one K tile/core, got {k=} {num_compute_cores=}")
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_largest_divisor(k_tiles_per_core),
        per_core_M=math.ceil(m / ttnn.TILE_SIZE),
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_compute_cores)),
        fused_activation=None,
    )


def _sparse_program_config(
    *,
    grid: tuple[int, int],
    m: int,
    n: int,
    k: int,
    in0_block_w: int,
    out_block_w: int,
    out_subblock_w: int,
):
    """Build and statically validate North-Mini's sparse 1-D matmul geometry."""

    grid_x, grid_y = grid
    num_cores = grid_x * grid_y
    mt = math.ceil(m / ttnn.TILE_SIZE)
    kt = math.ceil(k / ttnn.TILE_SIZE)
    nt = math.ceil(n / ttnn.TILE_SIZE)
    per_core_m = max(1, mt)
    per_core_n = math.ceil(nt / num_cores)
    num_blocks = math.ceil(mt / per_core_m) * math.ceil(nt / per_core_n)
    if num_blocks > num_cores:
        raise ValueError(f"sparse geometry {grid=} has {num_blocks} output blocks for {num_cores} cores")
    if kt % in0_block_w:
        raise ValueError(f"sparse geometry requires Kt={kt} divisible by {in0_block_w=}")
    if per_core_n % out_block_w:
        raise ValueError(f"sparse geometry requires {per_core_n=} divisible by {out_block_w=}")
    if out_block_w % out_subblock_w:
        raise ValueError(f"sparse geometry requires {out_block_w=} divisible by {out_subblock_w=}")
    if out_subblock_w > 4:
        raise ValueError(f"sparse output subblock area must be <= 4 tiles, got {out_subblock_w}")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _dense_large_prefill_program_config(
    *,
    grid: tuple[int, int],
    m: int,
    n: int,
    k: int,
    in0_block_w: int,
    out_subblock_w: int,
):
    """Build an explicit 2-D config for one expert-batched prefill matmul."""

    grid_x, grid_y = grid
    mt = math.ceil(m / ttnn.TILE_SIZE)
    kt = math.ceil(k / ttnn.TILE_SIZE)
    nt = math.ceil(n / ttnn.TILE_SIZE)
    per_core_m = math.ceil(mt / grid_y)
    per_core_n = math.ceil(nt / grid_x)
    if kt % in0_block_w:
        raise ValueError(f"dense prefill geometry requires Kt={kt} divisible by {in0_block_w=}")
    if per_core_n % out_subblock_w:
        raise ValueError(f"dense prefill geometry requires {per_core_n=} divisible by {out_subblock_w=}")
    if out_subblock_w > 4:
        raise ValueError(f"dense prefill output subblock area must be <= 4 tiles, got {out_subblock_w}")
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        # The second input has an expert batch dimension, so fusing batch into
        # M would change the batched-matmul contract.
        fuse_batch=False,
    )


def _dense_decode_program_config(
    *,
    grid: tuple[int, int],
    m: int,
    n: int,
    k: int,
    in0_block_w: int,
    out_block_w: int,
    out_subblock_w: int,
    fuse_batch: bool = False,
):
    """Build a role-specific one-tile-M dense decode matmul program."""

    grid_x, grid_y = grid
    num_cores = grid_x * grid_y
    mt = math.ceil(m / ttnn.TILE_SIZE)
    kt = math.ceil(k / ttnn.TILE_SIZE)
    nt = math.ceil(n / ttnn.TILE_SIZE)
    per_core_n = math.ceil(nt / num_cores)
    per_core_m = mt
    if kt % in0_block_w:
        raise ValueError(f"dense decode requires Kt={kt} divisible by {in0_block_w=}")
    if per_core_n % out_block_w:
        raise ValueError(f"dense decode requires {per_core_n=} divisible by {out_block_w=}")
    if out_block_w % out_subblock_w:
        raise ValueError(f"dense decode requires {out_block_w=} divisible by {out_subblock_w=}")
    if out_subblock_w > 8:
        raise ValueError(f"dense decode output subblock area must be <= 8 tiles, got {out_subblock_w}")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=fuse_batch,
        fused_activation=None,
        mcast_in0=True,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Optimized runtime; inheritance shares only validation/setup utilities."""

    def __init__(self, *, policy: OptimizationPolicy, candidate: str, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy
        self.candidate = candidate
        self.prefill_moe_chunk = policy.prefill_chunk_size
        self.attention_compute_config = _compute_config(self.mesh_device, policy.attention_fidelity)
        self.dense_gate_up_compute_config = _compute_config(self.mesh_device, policy.dense_gate_up_fidelity)
        self.dense_down_compute_config = _compute_config(self.mesh_device, policy.dense_down_fidelity)
        dense_decode_gate_up_fidelity = (
            ttnn.MathFidelity.LoFi
            if self.batch == 32 and policy.dense_decode_lofi_batch32
            else policy.dense_gate_up_fidelity
        )
        dense_decode_down_fidelity = (
            ttnn.MathFidelity.LoFi
            if self.batch == 32 and policy.dense_decode_lofi_batch32
            else policy.dense_down_fidelity
        )
        self.dense_decode_gate_up_compute_config = _compute_config(self.mesh_device, dense_decode_gate_up_fidelity)
        self.dense_decode_down_compute_config = _compute_config(self.mesh_device, dense_decode_down_fidelity)
        self.router_compute_config = _compute_config(self.mesh_device, ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        self.dense_large_prefill_compute_config = _compute_config(
            self.mesh_device, ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )
        self.decode_expert_gate_up_compute_config = _compute_config(
            self.mesh_device, policy.decode_expert_gate_up_fidelity
        )
        self.decode_expert_down_compute_config = _compute_config(self.mesh_device, policy.decode_expert_down_fidelity)
        self.prefill_expert_gate_up_compute_config = _compute_config(
            self.mesh_device, policy.prefill_expert_gate_up_fidelity
        )
        self.prefill_expert_down_compute_config = _compute_config(self.mesh_device, policy.prefill_expert_down_fidelity)
        self.decode_gate_up_grid = policy.gate_up_grid
        self.decode_gate_up_in0_block_w = policy.gate_up_in0_block_w
        self.decode_gate_up_out_block_w = policy.gate_up_out_block_w
        self.decode_gate_up_subblock_w = policy.gate_up_subblock_w
        self.decode_down_grid = policy.down_grid
        self.decode_down_in0_block_w = policy.down_in0_block_w
        self.decode_down_out_block_w = policy.down_out_block_w
        self.decode_down_subblock_w = policy.down_subblock_w
        self.use_unpacked_dense_decode = (
            self.mlp_type == "dense" and self.batch == 1 and policy.dense_decode_unpacked_batch1
        )
        self.dense_decode_gate_up_grid = policy.dense_decode_gate_up_grid
        self.dense_decode_gate_up_in0_block_w = policy.dense_decode_gate_up_in0_block_w
        self.dense_decode_gate_up_out_block_w = policy.dense_decode_gate_up_out_block_w
        self.dense_decode_gate_up_subblock_w = policy.dense_decode_gate_up_subblock_w
        self.dense_decode_gate_up_interleaved_input = policy.dense_decode_gate_up_interleaved_input
        if self.use_unpacked_dense_decode:
            self.dense_decode_gate_up_grid = (8, 6)
            self.dense_decode_gate_up_in0_block_w = 16
            self.dense_decode_gate_up_out_block_w = 2
            self.dense_decode_gate_up_subblock_w = 2
            self.dense_decode_gate_up_interleaved_input = True
        explicit_geometry_candidates = {
            "expert_bfp8_hifi2_geo_g24_g32",
            "expert_bfp8_hifi2_geo_g16",
            "expert_bfp8_hifi2_geo_g8",
            "expert_bfp8_hifi2_geo_g32_safe",
            "sparse_bfp4_block16",
        }
        if self.batch == 32 and candidate not in explicit_geometry_candidates:
            self.decode_gate_up_grid = (8, 4)
            self.decode_gate_up_in0_block_w = 8
            self.decode_gate_up_out_block_w = 1
            self.decode_gate_up_subblock_w = 1
            self.decode_down_grid = (8, 4)
            self.decode_down_in0_block_w = 12
            self.decode_down_out_block_w = 1
            self.decode_down_subblock_w = 1
        if (policy.dense_large_prefill_gate_up_grid is None) != (policy.dense_large_prefill_down_grid is None):
            raise ValueError("dense large-prefill gate/up and down grids must be enabled together")
        self.decode_residual_memory_config = None
        self.decode_norm_program_config = None
        self.decode_norm_compute_config = None
        if policy.decode_sharded_residual:
            # hidden=2048 is 64 tiles wide.  Thirty-two cores leave two tiles
            # per core.  The public [1, batch, 1, hidden] shape gives each
            # logical batch row its own padded tile row, so the physical
            # sharded height and norm block_h must scale with decode batch.
            residual_grid = ttnn.CoreGrid(x=8, y=4)
            residual_shard_height = self.batch * ttnn.TILE_SIZE
            self.decode_residual_memory_config = ttnn.create_sharded_memory_config(
                (residual_shard_height, self.hidden_size // residual_grid.num_cores),
                residual_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.decode_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[residual_grid.x, residual_grid.y],
                subblock_w=2,
                block_h=residual_shard_height // ttnn.TILE_SIZE,
                block_w=2,
                inplace=False,
            )
            self.decode_norm_compute_config = _compute_config(
                self.mesh_device,
                ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,
            )
        self.decode_qkv_program_config = None
        self.decode_o_program_config = None
        self.use_attention_dram_sharded = policy.attention_dram_sharded and (
            self.batch == 1 or policy.attention_dram_sharded_serving
        )
        if self.use_attention_dram_sharded:
            if not policy.decode_sharded_residual:
                raise ValueError("DRAM-sharded attention requires the width-sharded decode residual candidate")
            self.decode_qkv_program_config = _dram_sharded_decode_program_config(
                m=ttnn.TILE_SIZE,
                k=self.hidden_size,
                n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                num_compute_cores=32,
            )
            self.decode_o_program_config = _dram_sharded_decode_program_config(
                m=ttnn.TILE_SIZE,
                k=self.num_heads * self.head_dim,
                n=self.hidden_size,
                num_compute_cores=self.num_heads,
            )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        self.dense_decode_gate_up_program_config = None
        self.dense_decode_down_program_config = None
        self.dense_decode_down_dram_program_config = None
        self.dense_decode_down_input_memory_config = None
        self.use_dense_decode_down_dram_sharded = (
            self.mlp_type == "dense"
            and policy.dense_decode_down_dram_sharded
            and (self.batch == 32 or policy.dense_decode_down_dram_sharded_batch1)
        )
        if self.mlp_type == "dense" and policy.explicit_dense_decode_programs:
            self.dense_decode_gate_up_program_config = _dense_decode_program_config(
                grid=self.dense_decode_gate_up_grid,
                m=self.batch * ttnn.TILE_SIZE,
                n=(1 if self.use_unpacked_dense_decode else 2 if policy.packed_dense_mlp else 1)
                * self.intermediate_size,
                k=self.hidden_size,
                in0_block_w=self.dense_decode_gate_up_in0_block_w,
                out_block_w=self.dense_decode_gate_up_out_block_w,
                out_subblock_w=self.dense_decode_gate_up_subblock_w,
                fuse_batch=True,
            )
            self.dense_decode_down_program_config = _dense_decode_program_config(
                grid=policy.dense_decode_down_grid,
                m=ttnn.TILE_SIZE,
                n=self.hidden_size,
                k=self.intermediate_size,
                in0_block_w=policy.dense_decode_down_in0_block_w,
                out_block_w=policy.dense_decode_down_out_block_w,
                out_subblock_w=policy.dense_decode_down_subblock_w,
            )
        if self.use_dense_decode_down_dram_sharded:
            self.dense_decode_down_dram_program_config = _dram_sharded_decode_program_config(
                m=ttnn.TILE_SIZE,
                k=self.intermediate_size,
                n=self.hidden_size,
                num_compute_cores=32,
            )
            dense_down_grid = ttnn.CoreGrid(x=8, y=4)
            self.dense_decode_down_input_memory_config = ttnn.create_sharded_memory_config(
                (ttnn.TILE_SIZE, self.intermediate_size // dense_down_grid.num_cores),
                dense_down_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

    @staticmethod
    def _configuration_value(value):
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        name = getattr(value, "name", None)
        return name if name is not None else str(value)

    def effective_configuration(self):
        """Return the construction-time policy and derived program knobs as JSON-safe data."""

        policy = {
            field_name: self._configuration_value(getattr(self.policy, field_name))
            for field_name in self.policy.__dataclass_fields__
        }
        return {
            "candidate": self.candidate,
            "policy": policy,
            "sparse_matmul_output_dtype": self._configuration_value(ttnn.bfloat8_b),
            "execution_topology": {
                "decode_experts": (
                    "routed_sparse_packed_gate_up"
                    if self.policy.packed_expert_gate_up
                    else "routed_sparse_separate_gate_up"
                ),
                "small_prefill_experts": "routed_sparse_separate_gate_up",
                "large_prefill_experts": "dense_all_expert_composite",
                "large_prefill_gate_up": (
                    "packed_single_matmul_split"
                    if self.policy.packed_dense_large_prefill
                    else "separate_same_input_matmuls"
                ),
                "large_prefill_threshold_tokens": 1024,
                "large_prefill_chunk_tokens": self.policy.dense_large_prefill_chunk_size,
                "large_prefill_functional_router_compute": self.policy.prefill_functional_router_compute,
                "large_prefill_weights": (
                    "bf16_phase_specific"
                    if self.mlp_type == "sparse" and self.policy.sparse_experts
                    else "phase_native_weights"
                ),
                "decode_prefill_down_storage": (
                    "shared"
                    if self.mlp_type == "sparse"
                    and self.policy.sparse_experts
                    and self.policy.prefill_expert_down_dtype == self.policy.decode_expert_down_dtype
                    else "separate_or_not_applicable"
                ),
                "dense_gate_up": (
                    "batch1_decode_separate_else_packed"
                    if self.policy.packed_dense_mlp and self.policy.dense_decode_unpacked_batch1
                    else (
                        "packed_single_matmul_device_slices"
                        if self.policy.packed_dense_mlp
                        else "separate_same_input_matmuls"
                    )
                ),
                "dense_decode_gate_up_input": (
                    "L1_interleaved_after_explicit_reshard"
                    if self.dense_decode_gate_up_interleaved_input
                    else "residual_chain_native"
                ),
                "dense_decode_compute_fidelity": (
                    "LoFi" if self.batch == 32 and self.policy.dense_decode_lofi_batch32 else "phase_policy"
                ),
                "dense_decode_down": (
                    "dram_width_sharded"
                    if self.use_dense_decode_down_dram_sharded
                    else (
                        "explicit_interleaved_1d"
                        if self.dense_decode_down_program_config is not None
                        else "generic_interleaved"
                    )
                ),
                "cache_initialization": (
                    "lazy_empty" if self.batch == 32 and self.max_cache_len == ADVERTISED_CONTEXT else "zero_filled"
                ),
            },
            "decode_sdpa_program": {
                "compute_with_storage_grid_size": [11, 10],
                "exp_approx_mode": False,
                "q_chunk_size": 0,
                "k_chunk_size": 0,
            },
            "decode_residual_norm": {
                "sharded": self.policy.decode_sharded_residual,
                "layout": "L1_WIDTH_SHARDED" if self.policy.decode_sharded_residual else "DRAM_INTERLEAVED",
                "grid": [8, 4] if self.policy.decode_sharded_residual else None,
                "shard_shape": [self.batch * ttnn.TILE_SIZE, 64] if self.policy.decode_sharded_residual else None,
                "program_class": (
                    "LayerNormShardedMultiCoreProgramConfig" if self.policy.decode_sharded_residual else None
                ),
            },
            "decode_attention_projection": {
                "dram_sharded_weights": self.use_attention_dram_sharded,
                "weight_memory": "DRAM_WIDTH_SHARDED" if self.use_attention_dram_sharded else "DRAM_INTERLEAVED",
                "qkv_program_class": (
                    "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" if self.use_attention_dram_sharded else None
                ),
                "qkv_in0_block_w": 2 if self.use_attention_dram_sharded else None,
                "qkv_per_core_M": 1 if self.use_attention_dram_sharded else None,
                "qkv_per_core_N": 5 if self.use_attention_dram_sharded else None,
                "o_program_class": (
                    "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" if self.use_attention_dram_sharded else None
                ),
                "o_in0_block_w": 4 if self.use_attention_dram_sharded else None,
                "o_per_core_M": 1 if self.use_attention_dram_sharded else None,
                "o_per_core_N": 2 if self.use_attention_dram_sharded else None,
            },
            "decode_router_program": {
                "enabled": self.policy.decode_router_grid is not None,
                "grid": self._configuration_value(self.policy.decode_router_grid),
                "in0_block_w": self.policy.decode_router_in0_block_w,
                "out_block_w": self.policy.decode_router_out_block_w,
                "out_subblock_w": self.policy.decode_router_subblock_w,
            },
            "dense_decode_down_program": {
                "enabled": self.dense_decode_down_program_config is not None,
                "program_class": (
                    "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig"
                    if self.dense_decode_down_dram_program_config is not None
                    else (
                        "MatmulMultiCoreReuseMultiCast1DProgramConfig"
                        if self.dense_decode_down_program_config is not None
                        else None
                    )
                ),
                "weight_memory": (
                    "DRAM_WIDTH_SHARDED"
                    if self.dense_decode_down_dram_program_config is not None
                    else "DRAM_INTERLEAVED"
                ),
                "dram_sharded_selected_for_batch": self.use_dense_decode_down_dram_sharded,
                "grid": self._configuration_value(self.policy.dense_decode_down_grid),
                "in0_block_w": (
                    3
                    if self.dense_decode_down_dram_program_config is not None
                    else self.policy.dense_decode_down_in0_block_w
                ),
                "out_block_w": (
                    None
                    if self.dense_decode_down_dram_program_config is not None
                    else self.policy.dense_decode_down_out_block_w
                ),
                "out_subblock_w": (
                    None
                    if self.dense_decode_down_dram_program_config is not None
                    else self.policy.dense_decode_down_subblock_w
                ),
                "per_core_M": 1,
                "per_core_N": math.ceil(
                    math.ceil(self.hidden_size / ttnn.TILE_SIZE)
                    / (self.policy.dense_decode_down_grid[0] * self.policy.dense_decode_down_grid[1])
                ),
            },
            "dense_decode_gate_up_program": {
                "enabled": self.dense_decode_gate_up_program_config is not None,
                "program_class": (
                    "MatmulMultiCoreReuseMultiCast1DProgramConfig"
                    if self.dense_decode_gate_up_program_config is not None
                    else None
                ),
                "grid": self._configuration_value(self.dense_decode_gate_up_grid),
                "in0_block_w": self.dense_decode_gate_up_in0_block_w,
                "out_block_w": self.dense_decode_gate_up_out_block_w,
                "out_subblock_w": self.dense_decode_gate_up_subblock_w,
                "per_core_M": self.batch,
                "per_core_N": math.ceil(
                    math.ceil(
                        (
                            (1 if self.use_unpacked_dense_decode else 2 if self.policy.packed_dense_mlp else 1)
                            * self.intermediate_size
                        )
                        / ttnn.TILE_SIZE
                    )
                    / (self.dense_decode_gate_up_grid[0] * self.dense_decode_gate_up_grid[1])
                ),
            },
            "expert_program": {
                "decode_gate_up_cores": list(self.decode_gate_up_grid),
                "decode_down_cores": list(self.decode_down_grid),
                "prefill_gate_up_cores": [8, 4],
                "prefill_down_cores": [8, 4],
                "decode_gate_up_in0_block_w": self.decode_gate_up_in0_block_w,
                "decode_down_in0_block_w": self.decode_down_in0_block_w,
                "prefill_gate_up_in0_block_w": 8,
                "prefill_down_in0_block_w": 12,
                "decode_gate_up_out_block_w": self.decode_gate_up_out_block_w,
                "decode_down_out_block_w": self.decode_down_out_block_w,
                "prefill_gate_up_out_block_w": 1,
                "prefill_down_out_block_w": 1,
                "decode_gate_up_subblock_w": self.decode_gate_up_subblock_w,
                "decode_down_subblock_w": self.decode_down_subblock_w,
                "prefill_gate_up_subblock_w": 1,
                "prefill_down_subblock_w": 1,
                "sequence_chunk_size": self.policy.prefill_chunk_size,
                "base_down_split_size": self.policy.prefill_chunk_size,
            },
            "dense_large_prefill_program": {
                "enabled": self.policy.dense_large_prefill_gate_up_grid is not None,
                "program_class": "MatmulMultiCoreReuseMultiCastProgramConfig",
                "gate_up_grid": self._configuration_value(self.policy.dense_large_prefill_gate_up_grid),
                "down_grid": self._configuration_value(self.policy.dense_large_prefill_down_grid),
                "gate_up_in0_block_w": self.policy.dense_large_prefill_gate_up_in0_block_w,
                "down_in0_block_w": self.policy.dense_large_prefill_down_in0_block_w,
                "gate_up_subblock": [1, self.policy.dense_large_prefill_gate_up_subblock_w],
                "down_subblock": [1, self.policy.dense_large_prefill_down_subblock_w],
                "input_memory": "DRAM_INTERLEAVED",
                "weight_memory": "DRAM_INTERLEAVED",
                "output_memory": "DRAM_INTERLEAVED",
                "gate_up_compute_fidelity": "HiFi4",
                "down_compute_fidelity": "HiFi4",
                "fp32_dest_acc_en": True,
                "output_dtype": self._configuration_value(ttnn.bfloat16),
                "fuse_batch": False,
                "packed_gate_up": self.policy.packed_dense_large_prefill,
                "functional_compute": self.policy.dense_large_prefill_functional_compute,
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, object],
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=1,
        max_cache_len=ADVERTISED_CONTEXT,
        page_size=DEFAULT_PAGE_SIZE,
        candidate="default",
        **_kwargs,
    ):
        import torch

        if candidate not in POLICIES:
            raise ValueError(f"unknown candidate {candidate!r}; expected one of {sorted(POLICIES)}")
        policy = POLICIES[candidate]
        if not isinstance(mesh_device, ttnn.MeshDevice) or tuple(mesh_device.shape) != (1, 1):
            raise ValueError("OptimizedDecoder requires a single-device 1x1 MeshDevice")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")
        if batch < 1 or batch > 32:
            raise ValueError(f"optimized decode batch must be in [1, 32], got {batch}")
        top_k = int(hf_config.num_experts_per_tok)
        if policy.decode_exact_nnz is not None and policy.decode_exact_nnz != top_k:
            raise ValueError(f"decode_exact_nnz must equal num_experts_per_tok={top_k}, got {policy.decode_exact_nnz}")
        if policy.decode_exact_nnz is not None and not policy.sparse_experts:
            raise ValueError("decode_exact_nnz requires routed sparse expert execution")
        if not 1 <= max_cache_len <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_cache_len must be in [1, {hf_config.max_position_embeddings}], got {max_cache_len}")
        if page_size < ttnn.TILE_SIZE or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive multiple of {ttnn.TILE_SIZE}, got {page_size}")

        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        num_kv_heads = int(hf_config.num_key_value_heads)
        head_dim = int(hf_config.head_dim)
        if (hidden_size, num_heads, num_kv_heads, head_dim) != (2048, 32, 4, 128):
            raise ValueError("North-Mini dimensions must be hidden=2048, heads=32, kv_heads=4, head_dim=128")

        q = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        o = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        norm = _require_tensor(state_dict, layer_idx, "input_layernorm.weight")
        q = q.index_select(0, _rope_output_permutation(num_heads, head_dim))
        k = k.index_select(0, _rope_output_permutation(num_kv_heads, head_dim))
        qkv = torch.cat((q, k, v), dim=0).transpose(-2, -1).to(torch.bfloat16)
        o_transposed = o.transpose(-2, -1).to(torch.bfloat16)
        weights = {
            "qkv": _as_device_tensor(qkv, mesh_device=mesh_device, dtype=policy.attention_weight_dtype),
            "o": _as_device_tensor(o_transposed, mesh_device=mesh_device, dtype=policy.attention_weight_dtype),
            "norm": _as_device_tensor(
                norm.reshape(1, 1, 1, hidden_size).to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
        }
        use_attention_dram_sharded = policy.attention_dram_sharded and (
            batch == 1 or policy.attention_dram_sharded_serving
        )
        if use_attention_dram_sharded:
            qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
            weights["qkv_decode"] = _as_device_tensor(
                qkv,
                mesh_device=mesh_device,
                dtype=policy.attention_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(
                    mesh_device,
                    k=hidden_size,
                    n=qkv_size,
                ),
            )
            weights["o_decode"] = _as_device_tensor(
                o_transposed,
                mesh_device=mesh_device,
                dtype=policy.attention_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(
                    mesh_device,
                    k=num_heads * head_dim,
                    n=hidden_size,
                ),
            )

        mlp_type = hf_config.mlp_layer_types[layer_idx]
        if mlp_type == "dense":
            gate = _require_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
            up = _require_tensor(state_dict, layer_idx, "mlp.up_proj.weight")
            down = _require_tensor(state_dict, layer_idx, "mlp.down_proj.weight")
            if policy.packed_dense_mlp:
                weights["gate_up_proj"] = _as_device_tensor(
                    torch.cat((gate, up), dim=0).transpose(-2, -1).to(torch.bfloat16),
                    mesh_device=mesh_device,
                    dtype=policy.dense_gate_up_dtype,
                )
            if not policy.packed_dense_mlp or (batch == 1 and policy.dense_decode_unpacked_batch1):
                weights["gate_proj"] = _as_device_tensor(
                    gate.transpose(-2, -1).to(torch.bfloat16),
                    mesh_device=mesh_device,
                    dtype=policy.dense_gate_up_dtype,
                )
                weights["up_proj"] = _as_device_tensor(
                    up.transpose(-2, -1).to(torch.bfloat16),
                    mesh_device=mesh_device,
                    dtype=policy.dense_gate_up_dtype,
                )
            weights["down_proj"] = _as_device_tensor(
                down.transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=policy.dense_down_dtype,
            )
            use_dense_decode_down_dram_sharded = policy.dense_decode_down_dram_sharded and (
                batch == 32 or policy.dense_decode_down_dram_sharded_batch1
            )
            if use_dense_decode_down_dram_sharded:
                weights["down_proj_decode"] = _as_device_tensor(
                    down.transpose(-2, -1).to(torch.bfloat16),
                    mesh_device=mesh_device,
                    dtype=policy.dense_down_dtype,
                    memory_config=_dram_sharded_weight_memory_config(
                        mesh_device,
                        k=int(down.shape[-1]),
                        n=hidden_size,
                    ),
                )
        elif mlp_type == "sparse":
            gate, up, down = _load_expert_weights(
                state_dict, layer_idx, int(hf_config.num_experts), int(hf_config.intermediate_size)
            )
            dense_gate, dense_up, dense_down = gate, up, down
            weights["router"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.gate.weight").transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
            )
            if policy.sparse_experts:
                if policy.packed_dense_large_prefill:
                    weights["expert_gate_up_dense_prefill"] = _as_device_tensor(
                        torch.cat((dense_gate, dense_up), dim=-1),
                        mesh_device=mesh_device,
                        dtype=ttnn.bfloat16,
                    )
                else:
                    weights["expert_gate_dense_prefill"] = _as_device_tensor(
                        dense_gate, mesh_device=mesh_device, dtype=ttnn.bfloat16
                    )
                    weights["expert_up_dense_prefill"] = _as_device_tensor(
                        dense_up, mesh_device=mesh_device, dtype=ttnn.bfloat16
                    )
                weights["expert_down_dense_prefill"] = _as_device_tensor(
                    dense_down, mesh_device=mesh_device, dtype=ttnn.bfloat16
                )
                gate = gate.unsqueeze(0)
                up = up.unsqueeze(0)
                down = down.unsqueeze(0)
            if policy.packed_expert_gate_up:
                weights["expert_gate_up"] = _as_device_tensor(
                    torch.cat((gate, up), dim=-1),
                    mesh_device=mesh_device,
                    dtype=policy.decode_expert_gate_up_dtype,
                )
            else:
                weights["expert_gate"] = _as_device_tensor(
                    gate, mesh_device=mesh_device, dtype=policy.decode_expert_gate_up_dtype
                )
                weights["expert_up"] = _as_device_tensor(
                    up, mesh_device=mesh_device, dtype=policy.decode_expert_gate_up_dtype
                )
            weights["expert_down"] = _as_device_tensor(
                down, mesh_device=mesh_device, dtype=policy.decode_expert_down_dtype
            )
            if policy.sparse_experts:
                # Packing wins for one-tile decode, while large-M prefill split
                # kernels conflict with reserved dispatch cores. Keep the
                # prefill pair separate so large program configs remain legal.
                weights["expert_gate_prefill"] = _as_device_tensor(
                    gate, mesh_device=mesh_device, dtype=policy.prefill_expert_gate_up_dtype
                )
                weights["expert_up_prefill"] = _as_device_tensor(
                    up, mesh_device=mesh_device, dtype=policy.prefill_expert_gate_up_dtype
                )
                # The default policy uses the same BFP8 representation in both
                # phases.  Keep one physical down-projection allocation and
                # alias it across the two names.  Candidates that intentionally
                # sweep a distinct prefill dtype still receive their own tensor.
                if policy.prefill_expert_down_dtype == policy.decode_expert_down_dtype:
                    weights["expert_down_prefill"] = weights["expert_down"]
                else:
                    weights["expert_down_prefill"] = _as_device_tensor(
                        down, mesh_device=mesh_device, dtype=policy.prefill_expert_down_dtype
                    )
        else:
            raise ValueError(f"unsupported North-Mini MLP kind {mlp_type!r}")

        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            page_size=page_size,
            weights=weights,
            policy=policy,
            candidate=candidate,
        )

    def create_paged_kv_cache(self, *, num_blocks: int | None = None):
        min_blocks = self.batch * math.ceil(self.max_cache_len / self.page_size)
        num_blocks = min_blocks if num_blocks is None else int(num_blocks)
        if num_blocks < min_blocks:
            raise ValueError(f"num_blocks={num_blocks} cannot cover required {min_blocks} blocks")
        shape = (num_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        kwargs = dict(
            dtype=self.policy.cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Serving allocates the batch-32 advertised cache before it is filled.
        # Avoid touching the entire 32+ GiB allocation at construction time;
        # smaller/test caches stay zeroed so unwritten masked tiles are benign.
        constructor = ttnn.empty if self.batch == 32 and self.max_cache_len == ADVERTISED_CONTEXT else ttnn.zeros
        return constructor(shape, **kwargs), constructor(shape, **kwargs)

    def _qkv_prefill(self, normalized, seq_len, position_cos, position_sin):
        fused = ttnn.linear(
            normalized,
            self.weights["qkv"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused = ttnn.reshape(fused, (self.batch, seq_len, -1))
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.use_rope:
            query = ttnn.experimental.rotary_embedding(
                query, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            key = ttnn.experimental.rotary_embedding(
                key, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            query = ttnn.slice(query, (0, 0, 0, 0), (self.batch, self.num_heads, seq_len, self.head_dim))
            key = ttnn.slice(key, (0, 0, 0, 0), (self.batch, self.num_kv_heads, seq_len, self.head_dim))
        return query, key, value

    def _attention_prefill(
        self, normalized, *, key_cache, value_cache, page_table, position_cos, position_sin, seq_len
    ):
        query, key, value = self._qkv_prefill(normalized, seq_len, position_cos, position_sin)
        cache_dtype = self.policy.cache_dtype
        fill_key = ttnn.typecast(key, cache_dtype) if key.dtype != cache_dtype else key
        fill_value = ttnn.typecast(value, cache_dtype) if value.dtype != cache_dtype else value
        for user in range(self.batch):
            key_user = ttnn.slice(fill_key, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            value_user = ttnn.slice(fill_value, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            ttnn.experimental.paged_fill_cache(key_cache, key_user, page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(value_cache, value_user, page_table, batch_idx=user)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _qkv_decode(self, normalized, position_cos, position_sin):
        qkv_weight = self.weights["qkv_decode"] if self.use_attention_dram_sharded else self.weights["qkv"]
        qkv_kwargs = {}
        qkv_output_memory = ttnn.DRAM_MEMORY_CONFIG
        qkv_input = normalized
        if self.use_attention_dram_sharded:
            # Put the decode tokens on matmul's M axis.  Batch <= 32 occupies
            # one tile, so this legal DRAM-sharded family uses per_core_M=1
            # at both batch 1 and serving batch 32.
            qkv_input = ttnn.reshape(normalized, (1, 1, self.batch, self.hidden_size))
            qkv_kwargs["program_config"] = self.decode_qkv_program_config
            qkv_output_memory = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        fused = ttnn.linear(
            qkv_input,
            qkv_weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.attention_compute_config,
            memory_config=qkv_output_memory,
            **qkv_kwargs,
        )
        if self.use_attention_dram_sharded:
            # Head creation requires interleaved BF16.  Keep the unavoidable
            # conversion in L1, as common Attention1D does.
            fused = ttnn.sharded_to_interleaved(fused, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        fused = ttnn.reshape(fused, (1, 1, self.batch, -1))
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        if self.use_rope:
            position_cos = ttnn.interleaved_to_sharded(position_cos, self.decode_rope_memory_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, self.decode_rope_memory_config)
            query = ttnn.experimental.rotary_embedding_hf(
                query,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            key = ttnn.experimental.rotary_embedding_hf(
                key,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
        return query, key, value

    def _attention_decode(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        position_cos,
        position_sin,
    ):
        query, key, value = self._qkv_decode(normalized, position_cos, position_sin)
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
        )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_positions,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self.decode_concat_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(
            attended, num_heads=self.num_heads, sub_core_grids=self.decode_sub_core_grids
        )
        o_weight = self.weights["o_decode"] if self.use_attention_dram_sharded else self.weights["o"]
        o_kwargs = {}
        o_output_memory = ttnn.DRAM_MEMORY_CONFIG
        if self.use_attention_dram_sharded:
            o_kwargs["program_config"] = self.decode_o_program_config
            o_output_memory = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        projected = ttnn.linear(
            attended,
            o_weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.attention_compute_config,
            memory_config=o_output_memory,
            **o_kwargs,
        )
        if self.use_attention_dram_sharded:
            # The public layout is [1, batch, 1, hidden], while head concat
            # retains a logical padded batch.  Slice/permute are interleaved,
            # so return to L1 and reshard once at the residual boundary.
            projected = ttnn.sharded_to_interleaved(projected, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        return ttnn.permute(projected, (0, 2, 1, 3))

    def _dense_mlp(self, normalized, *, prefill=False):
        projection_memory = ttnn.DRAM_MEMORY_CONFIG if prefill else ttnn.L1_MEMORY_CONFIG
        gate_up_input = normalized
        if not prefill and self.dense_decode_gate_up_interleaved_input:
            gate_up_input = ttnn.sharded_to_interleaved(normalized, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
            # The batch-32 packed output does not coexist with the wider
            # matmul circular buffers in L1.  Keep this audit-only retry in
            # DRAM so wider legal K blocks are measured instead of rejected
            # at their first allocation error.
            projection_memory = ttnn.DRAM_MEMORY_CONFIG
        use_packed_dense_mlp = self.policy.packed_dense_mlp and not (not prefill and self.use_unpacked_dense_decode)
        gate_up_compute_config = (
            self.dense_gate_up_compute_config if prefill else self.dense_decode_gate_up_compute_config
        )
        down_compute_config = self.dense_down_compute_config if prefill else self.dense_decode_down_compute_config
        if use_packed_dense_mlp:
            gate_up_kwargs = {}
            if not prefill and self.dense_decode_gate_up_program_config is not None:
                gate_up_kwargs["program_config"] = self.dense_decode_gate_up_program_config
            gate_up = ttnn.linear(
                gate_up_input,
                self.weights["gate_up_proj"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=gate_up_compute_config,
                memory_config=projection_memory,
                **gate_up_kwargs,
            )
            prefix_shape = tuple(gate_up.shape[index] for index in range(len(gate_up.shape) - 1))
            gate = ttnn.slice(
                gate_up,
                (0,) * len(prefix_shape) + (0,),
                prefix_shape + (self.intermediate_size,),
            )
            up = ttnn.slice(
                gate_up,
                (0,) * len(prefix_shape) + (self.intermediate_size,),
                prefix_shape + (2 * self.intermediate_size,),
            )
        else:
            gate_up_kwargs = {}
            if not prefill and self.dense_decode_gate_up_program_config is not None:
                gate_up_kwargs["program_config"] = self.dense_decode_gate_up_program_config
            gate = ttnn.linear(
                gate_up_input,
                self.weights["gate_proj"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=gate_up_compute_config,
                memory_config=projection_memory,
                **gate_up_kwargs,
            )
            up = ttnn.linear(
                gate_up_input,
                self.weights["up_proj"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=gate_up_compute_config,
                memory_config=projection_memory,
                **gate_up_kwargs,
            )
        activated = ttnn.multiply(ttnn.silu(gate), up, memory_config=projection_memory)
        down_kwargs = {}
        down_weight = self.weights["down_proj"]
        down_input = activated
        down_output_memory = ttnn.DRAM_MEMORY_CONFIG
        if not prefill and self.dense_decode_down_dram_program_config is not None:
            down_weight = self.weights["down_proj_decode"]
            down_input = ttnn.reshape(activated, (1, 1, self.batch, self.intermediate_size))
            down_input = ttnn.to_memory_config(down_input, self.dense_decode_down_input_memory_config)
            down_kwargs["program_config"] = self.dense_decode_down_dram_program_config
            down_output_memory = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        elif not prefill and self.dense_decode_down_program_config is not None:
            down_kwargs["program_config"] = self.dense_decode_down_program_config
        down_output = ttnn.linear(
            down_input,
            down_weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=down_compute_config,
            memory_config=down_output_memory,
            **down_kwargs,
        )
        if not prefill and self.dense_decode_down_dram_program_config is not None:
            down_output = ttnn.sharded_to_interleaved(down_output, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
            down_output = ttnn.reshape(down_output, (1, self.batch, 1, self.hidden_size))
        return down_output

    def _routing(self, normalized, token_count, *, prefill=False, exact_presence=False):
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))

        # The target router has close top-k logits.  Keep its public-M
        # accumulation behavior consistent with decode by evaluating large
        # prefill inputs in one-tile groups; a single M=1024 router matmul
        # changes expert membership on official weights.
        def project(chunk):
            kwargs = {}
            if self.policy.decode_router_grid is not None and token_count <= ttnn.TILE_SIZE:
                kwargs["program_config"] = _sparse_program_config(
                    grid=self.policy.decode_router_grid,
                    m=chunk.shape[0],
                    n=self.num_experts,
                    k=self.hidden_size,
                    in0_block_w=self.policy.decode_router_in0_block_w,
                    out_block_w=self.policy.decode_router_out_block_w,
                    out_subblock_w=self.policy.decode_router_subblock_w,
                )
            if not (prefill and self.policy.prefill_functional_router_compute):
                kwargs["compute_kernel_config"] = self.router_compute_config
            return ttnn.linear(
                chunk,
                self.weights["router"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **kwargs,
            )

        logits = (
            ttnn.concat([project(chunk) for chunk in ttnn.split(flat, ttnn.TILE_SIZE, dim=0)], dim=0)
            if token_count > ttnn.TILE_SIZE
            else project(flat)
        )
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        route_presence = None
        if exact_presence:
            # `nnz` is an exact count, not an upper bound.  Build its mask
            # from topk's unique indices and exactly representable ones rather
            # than from sigmoid scores, which Blackhole may flush to zero.
            route_presence = ttnn.scatter(
                ttnn.zeros_like(logits),
                dim=-1,
                index=top_indices,
                src=ttnn.ones_like(top_values),
            )
        return flat, routing, route_presence

    def _dense_large_prefill_matmul_kwargs(self, role, token_count):
        if self.policy.dense_large_prefill_functional_compute:
            return {}
        if self.policy.dense_large_prefill_gate_up_grid is None:
            return {"compute_kernel_config": self.dense_large_prefill_compute_config}
        if role == "gate_up":
            gate_up_n = self.intermediate_size * (2 if self.policy.packed_dense_large_prefill else 1)
            program_config = _dense_large_prefill_program_config(
                grid=self.policy.dense_large_prefill_gate_up_grid,
                m=token_count,
                n=gate_up_n,
                k=self.hidden_size,
                in0_block_w=self.policy.dense_large_prefill_gate_up_in0_block_w,
                out_subblock_w=self.policy.dense_large_prefill_gate_up_subblock_w,
            )
            compute_kernel_config = self.dense_large_prefill_compute_config
        elif role == "down":
            program_config = _dense_large_prefill_program_config(
                grid=self.policy.dense_large_prefill_down_grid,
                m=token_count,
                n=self.hidden_size,
                k=self.intermediate_size,
                in0_block_w=self.policy.dense_large_prefill_down_in0_block_w,
                out_subblock_w=self.policy.dense_large_prefill_down_subblock_w,
            )
            compute_kernel_config = self.dense_large_prefill_compute_config
        else:
            raise ValueError(f"unknown dense large-prefill matmul role {role!r}")
        return {
            "program_config": program_config,
            "compute_kernel_config": compute_kernel_config,
        }

    def _dense_expert_moe(self, flat, routing, token_count, *, prefill=False):
        if prefill and self.policy.sparse_experts:
            down_weight = self.weights["expert_down_dense_prefill"]
            if self.policy.packed_dense_large_prefill:
                gate_up_weight = self.weights["expert_gate_up_dense_prefill"]
                gate_weight = None
                up_weight = None
            else:
                gate_up_weight = None
                gate_weight = self.weights["expert_gate_dense_prefill"]
                up_weight = self.weights["expert_up_dense_prefill"]
        else:
            gate_up_weight = None
            gate_weight = self.weights["expert_gate"]
            up_weight = self.weights["expert_up"]
            down_weight = self.weights["expert_down"]
        expert_input = ttnn.reshape(flat, (1, token_count, self.hidden_size))
        expert_input = ttnn.repeat(expert_input, ttnn.Shape((self.num_experts, 1, 1)))
        gate_up_kwargs = self._dense_large_prefill_matmul_kwargs("gate_up", token_count) if prefill else {}
        down_kwargs = self._dense_large_prefill_matmul_kwargs("down", token_count) if prefill else {}
        if gate_up_weight is not None:
            gate_up = ttnn.matmul(
                expert_input,
                gate_up_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **gate_up_kwargs,
            )
            gate, up = ttnn.split(gate_up, self.intermediate_size, dim=-1)
        else:
            gate = ttnn.matmul(
                expert_input,
                gate_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **gate_up_kwargs,
            )
            up = ttnn.matmul(
                expert_input,
                up_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **gate_up_kwargs,
            )
        expert_output = ttnn.matmul(
            ttnn.multiply(ttnn.silu(gate), up),
            down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **down_kwargs,
        )
        routing = ttnn.reshape(ttnn.permute(routing, (1, 0)), (self.num_experts, token_count, 1))
        return ttnn.sum(ttnn.multiply(expert_output, routing), dim=0)

    def _sparse_expert_moe(self, flat, routing, token_count, *, prefill, route_presence=None):
        normalized = ttnn.reshape(flat, (1, 1, token_count, self.hidden_size))
        normalized = ttnn.to_memory_config(normalized, ttnn.L1_MEMORY_CONFIG)
        # sparse_matmul consumes one expert-presence vector for the whole M
        # group.  Union the active routes on device, then retain the original
        # per-token sigmoid scores for the exact weighted reduction below.
        nnz = _decode_sparse_nnz(self.policy, token_count, prefill=prefill)
        if nnz is not None and route_presence is None:
            raise ValueError("fixed sparse nnz requires an exact top-k presence mask")
        sparsity = ttnn.sum(route_presence if route_presence is not None else routing, dim=0)
        sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(sparsity), ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])
        expert_output_memory = ttnn.L1_MEMORY_CONFIG if token_count <= ttnn.TILE_SIZE else ttnn.DRAM_MEMORY_CONFIG
        weight_suffix = "_prefill" if prefill else ""
        gate_up_compute_config = (
            self.prefill_expert_gate_up_compute_config if prefill else self.decode_expert_gate_up_compute_config
        )
        down_compute_config = (
            self.prefill_expert_down_compute_config if prefill else self.decode_expert_down_compute_config
        )
        if self.policy.packed_expert_gate_up and not prefill:
            gate_up = ttnn.sparse_matmul(
                normalized,
                self.weights["expert_gate_up" + weight_suffix],
                sparsity=sparsity,
                nnz=nnz,
                memory_config=expert_output_memory,
                output_tile=output_tile,
                program_config=_sparse_program_config(
                    grid=self.decode_gate_up_grid,
                    m=token_count,
                    n=2 * self.intermediate_size,
                    k=self.hidden_size,
                    in0_block_w=self.decode_gate_up_in0_block_w,
                    out_block_w=self.decode_gate_up_out_block_w,
                    out_subblock_w=self.decode_gate_up_subblock_w,
                ),
                compute_kernel_config=gate_up_compute_config,
                dtype=ttnn.bfloat8_b,
            )
            gate, up = ttnn.split(gate_up, self.intermediate_size, dim=-1)
        else:
            gate_up_grid = (8, 4) if prefill else self.decode_gate_up_grid
            gate_up_in0_block_w = 8 if prefill else self.decode_gate_up_in0_block_w
            gate_up_out_block_w = 1 if prefill else self.decode_gate_up_out_block_w
            gate_up_subblock_w = 1 if prefill else self.decode_gate_up_subblock_w
            gate = ttnn.sparse_matmul(
                normalized,
                self.weights["expert_gate" + weight_suffix],
                sparsity=sparsity,
                nnz=nnz,
                memory_config=expert_output_memory,
                output_tile=output_tile,
                program_config=_sparse_program_config(
                    grid=gate_up_grid,
                    m=token_count,
                    n=self.intermediate_size,
                    k=self.hidden_size,
                    in0_block_w=gate_up_in0_block_w,
                    out_block_w=gate_up_out_block_w,
                    out_subblock_w=gate_up_subblock_w,
                ),
                compute_kernel_config=gate_up_compute_config,
                dtype=ttnn.bfloat8_b,
            )
            up = ttnn.sparse_matmul(
                normalized,
                self.weights["expert_up" + weight_suffix],
                sparsity=sparsity,
                nnz=nnz,
                memory_config=expert_output_memory,
                output_tile=output_tile,
                program_config=_sparse_program_config(
                    grid=gate_up_grid,
                    m=token_count,
                    n=self.intermediate_size,
                    k=self.hidden_size,
                    in0_block_w=gate_up_in0_block_w,
                    out_block_w=gate_up_out_block_w,
                    out_subblock_w=gate_up_subblock_w,
                ),
                compute_kernel_config=gate_up_compute_config,
                dtype=ttnn.bfloat8_b,
            )
        gate = ttnn.reshape(gate, (self.num_experts, token_count, self.intermediate_size))
        gate = ttnn.permute(gate, (1, 0, 2), memory_config=expert_output_memory)
        up = ttnn.reshape(up, (self.num_experts, token_count, self.intermediate_size))
        up = ttnn.permute(up, (1, 0, 2), memory_config=expert_output_memory)
        down_input = ttnn.reshape(
            ttnn.permute(ttnn.multiply(ttnn.silu(gate), up), (1, 0, 2)),
            (1, self.num_experts, token_count, self.intermediate_size),
        )
        down = ttnn.sparse_matmul(
            down_input,
            self.weights["expert_down" + weight_suffix],
            sparsity=sparsity,
            nnz=nnz,
            memory_config=expert_output_memory,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=_sparse_program_config(
                grid=(8, 4) if prefill else self.decode_down_grid,
                m=token_count,
                n=self.hidden_size,
                k=self.intermediate_size,
                in0_block_w=12 if prefill else self.decode_down_in0_block_w,
                out_block_w=1 if prefill else self.decode_down_out_block_w,
                out_subblock_w=1 if prefill else self.decode_down_subblock_w,
            ),
            compute_kernel_config=down_compute_config,
            dtype=ttnn.bfloat8_b,
        )
        output = ttnn.reshape(down, (self.num_experts, token_count, self.hidden_size))
        output = ttnn.permute(output, (1, 0, 2), memory_config=expert_output_memory)
        routing = ttnn.reshape(routing, (token_count, self.num_experts, 1))
        return ttnn.sum(ttnn.multiply(output, routing), dim=1)

    def _sparse_moe_chunk(self, normalized, token_count, *, prefill):
        nnz = _decode_sparse_nnz(self.policy, token_count, prefill=prefill)
        flat, routing, route_presence = self._routing(
            normalized,
            token_count,
            prefill=prefill,
            exact_presence=nnz is not None,
        )
        if self.policy.sparse_experts:
            return self._sparse_expert_moe(
                flat,
                routing,
                token_count,
                prefill=prefill,
                route_presence=route_presence,
            )
        return self._dense_expert_moe(flat, routing, token_count, prefill=prefill)

    def _sparse_moe(self, normalized, seq_len, *, prefill):
        total_tokens = self.batch * seq_len
        flat = ttnn.reshape(normalized, (1, 1, total_tokens, self.hidden_size))
        if prefill and total_tokens >= 1024:
            chunks = ttnn.split(flat, self.policy.dense_large_prefill_chunk_size, dim=2)
            outputs = []
            for chunk in chunks:
                token_count = chunk.shape[2]
                chunk_flat, routing, _ = self._routing(chunk, token_count, prefill=True)
                outputs.append(self._dense_expert_moe(chunk_flat, routing, token_count, prefill=True))
            return ttnn.reshape(ttnn.concat(outputs, dim=0), (1, self.batch, seq_len, self.hidden_size))
        if total_tokens <= self.prefill_moe_chunk:
            result = self._sparse_moe_chunk(flat, total_tokens, prefill=prefill)
        else:
            chunks = ttnn.split(flat, self.prefill_moe_chunk, dim=2)
            outputs = [self._sparse_moe_chunk(chunk, chunk.shape[2], prefill=prefill) for chunk in chunks]
            result = ttnn.concat(outputs, dim=0)
        return ttnn.reshape(result, (1, self.batch, seq_len, self.hidden_size))

    def _mlp(self, normalized, seq_len, *, prefill):
        return (
            self._dense_mlp(normalized, prefill=prefill)
            if self.mlp_type == "dense"
            else self._sparse_moe(normalized, seq_len, prefill=prefill)
        )

    def prefill_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos=None,
        position_sin=None,
    ):
        seq_len = self._validate_hidden(hidden_states, decode=False)
        if self.use_rope and (position_cos is None or position_sin is None):
            raise ValueError("this layer kind requires position_cos and position_sin")
        normalized = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.weights["norm"])
        attention = self._attention_prefill(
            normalized,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            position_cos=position_cos,
            position_sin=position_sin,
            seq_len=seq_len,
        )
        mlp = self._mlp(normalized, seq_len, prefill=True)
        return ttnn.add(ttnn.add(hidden_states, attention), mlp)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        position_cos=None,
        position_sin=None,
    ):
        self._validate_hidden(hidden_states, decode=True)
        if self.use_rope and (position_cos is None or position_sin is None):
            raise ValueError("this layer kind requires position_cos and position_sin")
        residual = hidden_states
        if self.policy.decode_sharded_residual:
            residual = ttnn.to_memory_config(hidden_states, self.decode_residual_memory_config)
            normalized = ttnn.rms_norm(
                residual,
                epsilon=self.eps,
                weight=self.weights["norm"],
                program_config=self.decode_norm_program_config,
                memory_config=self.decode_residual_memory_config,
                compute_kernel_config=self.decode_norm_compute_config,
            )
        else:
            normalized = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.weights["norm"])
        attention = self._attention_decode(
            normalized,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        mlp = self._mlp(normalized, 1, prefill=False)
        if self.policy.decode_sharded_residual:
            attention = ttnn.to_memory_config(attention, self.decode_residual_memory_config)
            mlp = ttnn.to_memory_config(mlp, self.decode_residual_memory_config)
            return ttnn.add(
                ttnn.add(residual, attention, memory_config=self.decode_residual_memory_config),
                mlp,
                memory_config=self.decode_residual_memory_config,
            )
        return ttnn.add(ttnn.add(residual, attention), mlp)

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
