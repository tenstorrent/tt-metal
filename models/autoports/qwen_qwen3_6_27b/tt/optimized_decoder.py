# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device Qwen3.6-27B decoder layer.

The functional decoder remains the correctness oracle.  This class reuses its
proven cache/state and shape helpers, but owns weight materialization and both
public runtime entry points.  In particular, the measured decode path uses
packed projections, phase-specific weights, explicit program/compute configs,
DRAM-sharded decode matmuls, and an L1 width-sharded residual/MLP contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    LINEAR_PREFILL_CHUNK_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    REPRESENTATIVE_LAYERS,
    FunctionalDecoder,
    _require_tensor,
    _to_device,
)


@dataclass(frozen=True)
class OptimizationPolicy:
    attention_weight_dtype: object
    mlp_gate_up_dtype: object
    mlp_down_dtype: object
    cache_dtype: object
    attention_fidelity: object
    mlp_fidelity: object
    activation_residual_dtype: object = ttnn.bfloat16
    qkv_fidelity: object | None = None
    o_fidelity: object | None = None
    packed_qkv: bool = True
    packed_gate_up: bool = True
    decode_in0_block_w: int = 20
    qkv_decode_in0_block_w: int | None = None
    o_decode_in0_block_w: int = 3
    mlp_gate_decode_in0_block_w: int | None = None
    mlp_up_decode_in0_block_w: int | None = None
    mlp_down_in0_block_w: int = 17
    prefill_in0_block_w: int = 4
    prefill_grid_y: int = 10
    linear_packed_decode: bool = False
    linear_outer_product: bool = False
    linear_packed_in0_block_w: int = 2
    linear_out_in0_block_w: int = 3
    linear_input_weight_dtype: object = ttnn.bfloat16
    linear_output_weight_dtype: object = ttnn.bfloat16
    linear_input_fidelity: object = ttnn.MathFidelity.HiFi2
    linear_output_fidelity: object = ttnn.MathFidelity.HiFi2
    linear_recurrent_program: str = "auto"
    linear_recurrent_fidelity: object = ttnn.MathFidelity.HiFi2
    linear_recurrent_state_dtype: object = ttnn.float32
    decode_storage_cores: int = 8
    # Advisor-challenger placement bundle.  This remains default-off in the
    # shipped decoder; experiment policies are constructed with
    # dataclasses.replace from the frozen incumbent.
    advisor_plan: str = "off"


POLICIES = {
    # The default is the strongest expected batch-1 decode policy.  It is not
    # accepted until real-weight PCC and final default timing reproduce it.
    "default": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        packed_gate_up=False,
        decode_in0_block_w=2,
    ),
    "bfp8_hifi2": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_gate_up_dtype=ttnn.bfloat8_b,
        mlp_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        mlp_fidelity=ttnn.MathFidelity.HiFi2,
        decode_in0_block_w=1,
    ),
    "bfp8_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_gate_up_dtype=ttnn.bfloat8_b,
        mlp_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        decode_in0_block_w=1,
    ),
    "bfp4_mlp": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        decode_in0_block_w=1,
    ),
    "bfp4_gate_up": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi2,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        decode_in0_block_w=1,
    ),
    "split_gate_up": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        packed_gate_up=False,
        decode_in0_block_w=2,
    ),
    "bf16_cache": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat16,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        decode_in0_block_w=2,
    ),
    "bf16_hifi4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        mlp_gate_up_dtype=ttnn.bfloat16,
        mlp_down_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat16,
        attention_fidelity=ttnn.MathFidelity.HiFi4,
        mlp_fidelity=ttnn.MathFidelity.HiFi4,
        packed_gate_up=False,
        decode_in0_block_w=1,
        mlp_down_in0_block_w=1,
        prefill_in0_block_w=2,
    ),
    "bf16_attention_bfp4_mlp": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat16,
        attention_fidelity=ttnn.MathFidelity.HiFi4,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        packed_gate_up=False,
        decode_in0_block_w=1,
        prefill_in0_block_w=2,
    ),
    "bf16_attention_bfp4_mlp_bfp8_cache": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.HiFi4,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        packed_gate_up=False,
        decode_in0_block_w=1,
        prefill_in0_block_w=2,
    ),
    "geometry_w10": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        decode_in0_block_w=10,
    ),
    "geometry_w5": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        decode_in0_block_w=5,
    ),
    "geometry_w4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        decode_in0_block_w=4,
    ),
    "geometry_w2": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        decode_in0_block_w=2,
    ),
    "prefill_w2": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        packed_gate_up=False,
        decode_in0_block_w=2,
        prefill_in0_block_w=2,
    ),
    "prefill_grid8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_dtype=ttnn.bfloat4_b,
        mlp_down_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        packed_gate_up=False,
        decode_in0_block_w=2,
        prefill_grid_y=8,
    ),
}

# Precision-locked, per-role candidates for the selected full-attention
# BF16-attention/BFP4-MLP/BFP8-cache family.  Keeping these as independent
# policies makes whole-layer B1/B32 A/B runs cumulative and reproducible.
_FINAL_FULL = POLICIES["bf16_attention_bfp4_mlp_bfp8_cache"]
_FINAL_CUMULATIVE = replace(
    _FINAL_FULL,
    qkv_fidelity=ttnn.MathFidelity.HiFi2,
    o_fidelity=ttnn.MathFidelity.HiFi2,
    qkv_decode_in0_block_w=2,
    mlp_gate_decode_in0_block_w=5,
    mlp_up_decode_in0_block_w=5,
    advisor_plan="mlp_product_only",
)
_LINEAR_FP32_FINAL = replace(
    POLICIES["default"],
    linear_packed_decode=True,
    linear_outer_product=True,
    linear_recurrent_program="grid4_w4",
)
_LINEAR_PROJECTION_BASELINE = replace(
    _LINEAR_FP32_FINAL,
    linear_recurrent_state_dtype=ttnn.bfloat8_b,
)
_LINEAR_PRECISION_FINAL = replace(
    _LINEAR_PROJECTION_BASELINE,
    linear_input_weight_dtype=ttnn.bfloat4_b,
    linear_input_fidelity=ttnn.MathFidelity.LoFi,
    linear_output_weight_dtype=ttnn.bfloat4_b,
    linear_output_fidelity=ttnn.MathFidelity.LoFi,
)
_LINEAR_FINAL = replace(
    _LINEAR_PRECISION_FINAL,
    linear_packed_in0_block_w=5,
    linear_out_in0_block_w=12,
    advisor_plan="mlp_product_only",
)
POLICIES.update(
    {
        "final_qkv_w2": replace(_FINAL_FULL, qkv_decode_in0_block_w=2),
        "final_qkv_w4": replace(_FINAL_FULL, qkv_decode_in0_block_w=4),
        "final_o_w6": replace(_FINAL_FULL, o_decode_in0_block_w=6),
        "final_o_w4": replace(_FINAL_FULL, o_decode_in0_block_w=4),
        "final_o_w8": replace(_FINAL_FULL, o_decode_in0_block_w=8),
        "final_o_w12": replace(_FINAL_FULL, o_decode_in0_block_w=12),
        "final_gate_up_w2": replace(
            _FINAL_FULL,
            mlp_gate_decode_in0_block_w=2,
            mlp_up_decode_in0_block_w=2,
        ),
        "final_gate_up_w4": replace(
            _FINAL_FULL,
            mlp_gate_decode_in0_block_w=4,
            mlp_up_decode_in0_block_w=4,
        ),
        "final_gate_up_w5": replace(
            _FINAL_FULL,
            mlp_gate_decode_in0_block_w=5,
            mlp_up_decode_in0_block_w=5,
        ),
        "final_gate_w5": replace(_FINAL_FULL, mlp_gate_decode_in0_block_w=5),
        "final_up_w5": replace(_FINAL_FULL, mlp_up_decode_in0_block_w=5),
        "final_gate_w2": replace(_FINAL_FULL, mlp_gate_decode_in0_block_w=2),
        "final_gate_w4": replace(_FINAL_FULL, mlp_gate_decode_in0_block_w=4),
        "final_gate_w10": replace(_FINAL_FULL, mlp_gate_decode_in0_block_w=10),
        "final_up_w2": replace(_FINAL_FULL, mlp_up_decode_in0_block_w=2),
        "final_up_w4": replace(_FINAL_FULL, mlp_up_decode_in0_block_w=4),
        "final_up_w10": replace(_FINAL_FULL, mlp_up_decode_in0_block_w=10),
        "final_mlp_hifi2": replace(_FINAL_FULL, mlp_fidelity=ttnn.MathFidelity.HiFi2),
        "final_down_w4": replace(_FINAL_FULL, mlp_down_in0_block_w=4),
        "final_down_w34": replace(_FINAL_FULL, mlp_down_in0_block_w=34),
        "final_down_w68": replace(_FINAL_FULL, mlp_down_in0_block_w=68),
        "final_attention_hifi2": replace(_FINAL_FULL, attention_fidelity=ttnn.MathFidelity.HiFi2),
        "final_qkv_hifi2": replace(_FINAL_FULL, qkv_fidelity=ttnn.MathFidelity.HiFi2),
        "final_o_hifi2": replace(_FINAL_FULL, o_fidelity=ttnn.MathFidelity.HiFi2),
        "final_cumulative": _FINAL_CUMULATIVE,
        "advisor_mlp_product_full_b32": replace(_FINAL_CUMULATIVE, advisor_plan="mlp_product_only"),
        "final_cumulative_o4": replace(
            _FINAL_CUMULATIVE,
            o_decode_in0_block_w=4,
        ),
        "final_cum_qkv_w4": replace(_FINAL_CUMULATIVE, qkv_decode_in0_block_w=4),
        "final_cum_o_w4": replace(_FINAL_CUMULATIVE, o_decode_in0_block_w=4),
        "final_cum_o_w6": replace(_FINAL_CUMULATIVE, o_decode_in0_block_w=6),
        "final_cum_o_w8": replace(_FINAL_CUMULATIVE, o_decode_in0_block_w=8),
        "final_cum_o_w12": replace(_FINAL_CUMULATIVE, o_decode_in0_block_w=12),
        "final_cum_gate_w2": replace(_FINAL_CUMULATIVE, mlp_gate_decode_in0_block_w=2),
        "final_cum_gate_w4": replace(_FINAL_CUMULATIVE, mlp_gate_decode_in0_block_w=4),
        "final_cum_gate_w10": replace(_FINAL_CUMULATIVE, mlp_gate_decode_in0_block_w=10),
        "final_cum_gate_w20": replace(_FINAL_CUMULATIVE, mlp_gate_decode_in0_block_w=20),
        "final_cum_up_w2": replace(_FINAL_CUMULATIVE, mlp_up_decode_in0_block_w=2),
        "final_cum_up_w4": replace(_FINAL_CUMULATIVE, mlp_up_decode_in0_block_w=4),
        "final_cum_up_w10": replace(_FINAL_CUMULATIVE, mlp_up_decode_in0_block_w=10),
        "final_cum_up_w20": replace(_FINAL_CUMULATIVE, mlp_up_decode_in0_block_w=20),
        "final_cum_down_w4": replace(_FINAL_CUMULATIVE, mlp_down_in0_block_w=4),
        "final_cum_down_w34": replace(_FINAL_CUMULATIVE, mlp_down_in0_block_w=34),
        "final_cum_down_w68": replace(_FINAL_CUMULATIVE, mlp_down_in0_block_w=68),
        "final_cum_mlp_hifi2": replace(_FINAL_CUMULATIVE, mlp_fidelity=ttnn.MathFidelity.HiFi2),
        "final_cum_grid4": replace(_FINAL_CUMULATIVE, decode_storage_cores=4),
        # Gated-delta decode has four projections of the same normalized
        # hidden state. Materialize them as one DRAM-sharded projection and
        # keep its output projection DRAM-sharded as well. Independent dtype
        # and fidelity controls retain the measured BF16 baseline and BFP4
        # winner.
        "linear_packed_dram": replace(
            POLICIES["default"],
            linear_packed_decode=True,
        ),
        "linear_outer_product": replace(
            POLICIES["default"],
            linear_outer_product=True,
        ),
        "linear_final": _LINEAR_FINAL,
        "advisor_mlp_product_linear_b32": replace(_LINEAR_FINAL, advisor_plan="mlp_product_only"),
        "linear_state_fp32": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_recurrent_state_dtype=ttnn.float32,
        ),
        "linear_packed_w4": replace(
            POLICIES["default"],
            linear_packed_decode=True,
            linear_packed_in0_block_w=4,
        ),
        "linear_out_w4": replace(
            POLICIES["default"],
            linear_packed_decode=True,
            linear_out_in0_block_w=4,
        ),
        # Precision-locked geometry controls for the selected BFP4/LoFi
        # projections and BFP8 recurrent state. These must remain derived
        # from _LINEAR_PRECISION_FINAL so block-width results are not
        # confounded with the earlier BF16/FP32 policy.
        "linear_final_input_w1": replace(_LINEAR_PRECISION_FINAL, linear_packed_in0_block_w=1),
        "linear_final_input_w4": replace(_LINEAR_PRECISION_FINAL, linear_packed_in0_block_w=4),
        "linear_final_input_w5": replace(_LINEAR_PRECISION_FINAL, linear_packed_in0_block_w=5),
        "linear_final_input_w10": replace(_LINEAR_PRECISION_FINAL, linear_packed_in0_block_w=10),
        "linear_final_input_w20": replace(_LINEAR_PRECISION_FINAL, linear_packed_in0_block_w=20),
        "linear_final_output_w1": replace(_LINEAR_PRECISION_FINAL, linear_out_in0_block_w=1),
        "linear_final_output_w2": replace(_LINEAR_PRECISION_FINAL, linear_out_in0_block_w=2),
        "linear_final_output_w4": replace(_LINEAR_PRECISION_FINAL, linear_out_in0_block_w=4),
        "linear_final_output_w6": replace(_LINEAR_PRECISION_FINAL, linear_out_in0_block_w=6),
        "linear_final_output_w8": replace(_LINEAR_PRECISION_FINAL, linear_out_in0_block_w=8),
        "linear_final_output_w12": replace(_LINEAR_PRECISION_FINAL, linear_out_in0_block_w=12),
        "linear_final_output_w24": replace(_LINEAR_PRECISION_FINAL, linear_out_in0_block_w=24),
        "linear_final_input_w5_output_w8": replace(
            _LINEAR_PRECISION_FINAL,
            linear_packed_in0_block_w=5,
            linear_out_in0_block_w=8,
        ),
        "linear_final_input_w5_output_w12": replace(
            _LINEAR_PRECISION_FINAL,
            linear_packed_in0_block_w=5,
            linear_out_in0_block_w=12,
        ),
        "linear_final_input_w5_output_w24": replace(
            _LINEAR_PRECISION_FINAL,
            linear_packed_in0_block_w=5,
            linear_out_in0_block_w=24,
        ),
        "linear_final_grid4": replace(_LINEAR_PRECISION_FINAL, decode_storage_cores=4),
        "linear_recurrent_explicit_w1": replace(_LINEAR_PROJECTION_BASELINE, linear_recurrent_program="grid4_w1"),
        "linear_recurrent_explicit_w2": replace(_LINEAR_PROJECTION_BASELINE, linear_recurrent_program="grid4_w2"),
        "linear_recurrent_explicit_w4": replace(_LINEAR_PROJECTION_BASELINE, linear_recurrent_program="grid4_w4"),
        "linear_recurrent_subblock2": replace(_LINEAR_PROJECTION_BASELINE, linear_recurrent_program="grid2_n2"),
        "linear_recurrent_hifi4": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_recurrent_fidelity=ttnn.MathFidelity.HiFi4,
        ),
        # The functional cache is FP32 even though its affine prefill scan
        # consumes and produces BF16 states.  Keep the selected linear
        # topology fixed while independently sweeping the persistent state
        # boundary. These retained controls keep the BF16 projection baseline
        # while the final policy independently selects BFP8 state storage.
        "linear_state_bf16": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_recurrent_state_dtype=ttnn.bfloat16,
        ),
        "linear_state_bfp8": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_recurrent_state_dtype=ttnn.bfloat8_b,
        ),
        "linear_state_bfp4": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_recurrent_state_dtype=ttnn.bfloat4_b,
        ),
        "linear_proj_bf16_hifi2": _LINEAR_PROJECTION_BASELINE,
        "linear_input_bf16_lofi": replace(_LINEAR_PROJECTION_BASELINE, linear_input_fidelity=ttnn.MathFidelity.LoFi),
        "linear_output_bf16_lofi": replace(_LINEAR_PROJECTION_BASELINE, linear_output_fidelity=ttnn.MathFidelity.LoFi),
        "linear_both_bf16_lofi": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_input_fidelity=ttnn.MathFidelity.LoFi,
            linear_output_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "linear_input_bfp8_hifi2": replace(_LINEAR_PROJECTION_BASELINE, linear_input_weight_dtype=ttnn.bfloat8_b),
        "linear_input_bfp8_lofi": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_input_weight_dtype=ttnn.bfloat8_b,
            linear_input_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "linear_output_bfp8_hifi2": replace(_LINEAR_PROJECTION_BASELINE, linear_output_weight_dtype=ttnn.bfloat8_b),
        "linear_output_bfp8_lofi": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_output_weight_dtype=ttnn.bfloat8_b,
            linear_output_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "linear_input_bfp4_lofi": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_input_weight_dtype=ttnn.bfloat4_b,
            linear_input_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "linear_output_bfp4_lofi": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_output_weight_dtype=ttnn.bfloat4_b,
            linear_output_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "linear_both_bfp4_lofi": replace(
            _LINEAR_PROJECTION_BASELINE,
            linear_input_weight_dtype=ttnn.bfloat4_b,
            linear_input_fidelity=ttnn.MathFidelity.LoFi,
            linear_output_weight_dtype=ttnn.bfloat4_b,
            linear_output_fidelity=ttnn.MathFidelity.LoFi,
        ),
    }
)


def resolve_policy(candidate: str, layer_kind: str) -> OptimizationPolicy:
    """Resolve the evidence-selected policy for a representative layer kind."""
    if candidate == "default" and layer_kind == "full_attention":
        return POLICIES["final_cumulative"]
    if candidate == "default" and layer_kind == "linear_attention":
        return POLICIES["linear_final"]
    return POLICIES[candidate]


def _dram_grid(mesh_device):
    size = mesh_device.dram_grid_size()
    return ttnn.CoreRangeSet(
        # The decode matmul streams one shard per DRAM-bank column.  Extending
        # this range through ``size.y`` makes the program infer 80 logical
        # workers on p300c and over-allocates each circular-buffer family.
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(size.x - 1, 0))}
    )


def _dram_weight_memory_config(mesh_device, *, k: int, n: int):
    cores = mesh_device.dram_grid_size().x
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * cores)) * ttnn.TILE_SIZE * cores
    shard = ttnn.ShardSpec(
        _dram_grid(mesh_device),
        (k, padded_n // cores),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)


def _l1_width_memory_config(*, rows: int, width: int, cores: int = 8):
    return ttnn.create_sharded_memory_config(
        shape=(rows, math.ceil(width / cores / ttnn.TILE_SIZE) * ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(x=cores, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _advisor_core_range_set(cores: int):
    """Row-major worker set matching shard-advise's partial final row."""
    grid_x = 11
    full_rows, tail = divmod(cores, grid_x)
    ranges = set()
    if full_rows:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if tail:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(tail - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _advisor_width_memory_config(*, rows: int, width: int, cores: int, shard_width: int):
    """Reconstruct an exact width-sharded layout from advised_plan.ops."""
    return ttnn.create_sharded_memory_config(
        shape=(rows, shard_width),
        core_grid=_advisor_core_range_set(cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _advisor_norm_memory_config():
    """B32 hidden-5120 norm layout: L1 block-sharded, 11 cores, [32, 480]."""
    return ttnn.create_sharded_memory_config(
        shape=(32, 480),
        core_grid=ttnn.CoreGrid(x=11, y=1),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _decode_program(*, k: int, n: int, in0_block_w: int, cores: int = 8, fused_activation=None):
    k_tiles_per_core = k // ttnn.TILE_SIZE // cores
    if k_tiles_per_core % in0_block_w:
        raise ValueError(f"in0_block_w={in0_block_w} must divide {k_tiles_per_core} K tiles/core for K={k}")
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=math.ceil(n / ttnn.TILE_SIZE / cores),
        fused_activation=fused_activation,
    )


def _prefill_program(*, rows: int, k: int, n: int, in0_block_w_limit: int, grid_y: int, fused_activation=None):
    grid_x = 8
    per_core_n = math.ceil(n / ttnn.TILE_SIZE / grid_x)
    out_subblock_w = 4
    while out_subblock_w > 1 and per_core_n % out_subblock_w:
        out_subblock_w -= 1
    k_tiles = k // ttnn.TILE_SIZE
    # Eight K tiles overcommits L1 for the packed 2x-intermediate MLP
    # projection (1.676 MB on Blackhole).  Four tiles keeps the same 2-D
    # output decomposition while fitting the 1.5 MiB worker-L1 contract.
    in0_block_w = min(in0_block_w_limit, k_tiles)
    while k_tiles % in0_block_w:
        in0_block_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=max(1, math.ceil(rows / ttnn.TILE_SIZE / grid_y)),
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=True,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Qwen3.6 decoder with an independently selectable optimized runtime."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=1,
        max_context=ADVERTISED_CONTEXT,
        page_size=64,
        candidate="default",
        policy_override=None,
        **kwargs,
    ):
        if candidate not in POLICIES:
            raise ValueError(f"unknown candidate {candidate!r}; expected one of {sorted(POLICIES)}")
        # Invoke the proven loader with cls=OptimizedDecoder.  The returned
        # runtime object is never a FunctionalDecoder fallback.
        decoder = FunctionalDecoder.from_state_dict.__func__(
            cls,
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_context=max_context,
            page_size=page_size,
            **kwargs,
        )
        decoder.policy = policy_override or resolve_policy(candidate, decoder.layer_kind)
        decoder.candidate = candidate
        decoder._configure_optimized_runtime(state_dict)
        policy = decoder.policy
        print(
            "OPTIMIZED_POLICY",
            f"candidate={candidate}",
            f"layer_kind={decoder.layer_kind}",
            f"attention_dtype={policy.attention_weight_dtype}",
            f"cache_dtype={policy.cache_dtype}",
            f"sdpa_fidelity={policy.attention_fidelity}",
            f"qkv_fidelity={policy.qkv_fidelity or policy.attention_fidelity}",
            f"o_fidelity={policy.o_fidelity or policy.attention_fidelity}",
            f"qkv_w={policy.qkv_decode_in0_block_w or policy.decode_in0_block_w}",
            f"o_w={policy.o_decode_in0_block_w}",
            f"gate_w={policy.mlp_gate_decode_in0_block_w or policy.decode_in0_block_w}",
            f"up_w={policy.mlp_up_decode_in0_block_w or policy.decode_in0_block_w}",
            f"down_w={policy.mlp_down_in0_block_w}",
            f"linear_packed={policy.linear_packed_decode}",
            f"linear_outer={policy.linear_outer_product}",
            f"linear_recurrent={policy.linear_recurrent_program}",
            f"linear_recurrent_fidelity={policy.linear_recurrent_fidelity}",
            f"linear_recurrent_state_dtype={policy.linear_recurrent_state_dtype}",
            f"linear_input_dtype={policy.linear_input_weight_dtype}",
            f"linear_input_fidelity={policy.linear_input_fidelity}",
            f"linear_input_w={policy.linear_packed_in0_block_w}",
            f"linear_output_dtype={policy.linear_output_weight_dtype}",
            f"linear_output_fidelity={policy.linear_output_fidelity}",
            f"linear_output_w={policy.linear_out_in0_block_w}",
            f"storage_cores={policy.decode_storage_cores}",
            f"advisor_plan={policy.advisor_plan}",
        )
        return decoder

    def _configure_optimized_runtime(self, state_dict):
        import torch

        policy = self.policy
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        rows = ttnn.TILE_SIZE

        if self.layer_kind == "linear_attention" and policy.linear_recurrent_state_dtype != ttnn.float32:
            # FunctionalDecoder constructs an FP32 zero state.  Convert it
            # once during setup so the persistent physical allocation—not
            # merely a transient decode operand—uses the candidate dtype.
            functional_state = self.caches["recurrent"]
            self.caches["recurrent"] = ttnn.typecast(
                functional_state,
                policy.linear_recurrent_state_dtype,
            )
            ttnn.deallocate(functional_state)

        self.attention_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=policy.attention_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.qkv_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=policy.qkv_fidelity or policy.attention_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.o_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=policy.o_fidelity or policy.attention_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.mlp_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=policy.mlp_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # Input packing and output projection are separate numerical and
        # performance boundaries. Keep their compute policies independent so
        # each retained sweep candidate is attributable.
        self.linear_input_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=policy.linear_input_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.linear_output_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=policy.linear_output_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.linear_recurrent_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=policy.linear_recurrent_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.norm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        storage_cores = policy.decode_storage_cores
        advisor_enabled = policy.advisor_plan in ("apply_all", "apply_all_minus_norm", "residual_only")
        advisor_norm_enabled = policy.advisor_plan in ("apply_all", "norm_only")
        if advisor_enabled:
            self.decode_residual_memory_config = _advisor_width_memory_config(
                rows=rows, width=hidden, cores=80, shard_width=64
            )
        else:
            self.decode_residual_memory_config = _l1_width_memory_config(rows=rows, width=hidden, cores=storage_cores)
        if advisor_norm_enabled:
            self.decode_norm_memory_config = _advisor_norm_memory_config()
            self.decode_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(11, 1),
                subblock_w=3,
                block_h=1,
                block_w=15,
                inplace=False,
            )
        else:
            self.decode_norm_memory_config = _l1_width_memory_config(rows=rows, width=hidden, cores=storage_cores)
            self.decode_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(storage_cores, 1),
                subblock_w=4,
                block_h=1,
                block_w=hidden // ttnn.TILE_SIZE // storage_cores,
                inplace=False,
            )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

        def host(suffix, *, transpose=False, add_one=False):
            value = _require_tensor(state_dict, self.layer_idx, suffix).to(torch.bfloat16)
            if transpose:
                value = value.transpose(-2, -1)
            if add_one:
                value = value + 1
            return value.contiguous()

        def upload(value, *, dtype, memory_config):
            # Direct BF16 host->DRAM-sharded tilize builds a 2.34 MiB CB on a
            # single worker for the 17k-wide projections.  Stage through
            # interleaved DRAM so the subsequent device reshard is distributed.
            target_memory_config = memory_config
            initial_memory_config = (
                ttnn.DRAM_MEMORY_CONFIG if dtype == ttnn.bfloat16 and memory_config.is_sharded() else memory_config
            )
            result = ttnn.from_torch(
                value,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=initial_memory_config,
            )
            if initial_memory_config != target_memory_config:
                sharded = ttnn.to_memory_config(result, target_memory_config)
                ttnn.deallocate(result)
                return sharded
            return result

        def replace(name, value):
            old = self.weights.get(name)
            self.weights[name] = value
            if old is not None:
                ttnn.deallocate(old, force=True)

        # Sharded norm weights use the common 1D RMSNorm row-major contract.
        norm_shape = (1, 1, hidden // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
        for name, suffix in (
            ("input_norm", "input_layernorm.weight"),
            ("post_attention_norm", "post_attention_layernorm.weight"),
        ):
            replace(
                name,
                _to_device(
                    host(suffix, add_one=True).reshape(norm_shape),
                    mesh_device=self.mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            )

        gate = host("mlp.gate_proj.weight", transpose=True)
        up = host("mlp.up_proj.weight", transpose=True)
        down = host("mlp.down_proj.weight", transpose=True)
        if policy.packed_gate_up:
            gate_up = torch.cat([gate, up], dim=-1)
            replace(
                "mlp_gate_up_decode",
                upload(
                    gate_up,
                    dtype=policy.mlp_gate_up_dtype,
                    memory_config=_dram_weight_memory_config(self.mesh_device, k=hidden, n=2 * intermediate),
                ),
            )
            replace(
                "mlp_gate_up_prefill",
                upload(
                    gate_up,
                    dtype=policy.mlp_gate_up_dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        else:
            for name, value in (("mlp_gate", gate), ("mlp_up", up)):
                replace(
                    f"{name}_decode",
                    upload(
                        value,
                        dtype=policy.mlp_gate_up_dtype,
                        memory_config=_dram_weight_memory_config(self.mesh_device, k=hidden, n=intermediate),
                    ),
                )
                replace(
                    f"{name}_prefill",
                    upload(value, dtype=policy.mlp_gate_up_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                )
        replace(
            "mlp_down_decode",
            upload(
                down,
                dtype=policy.mlp_down_dtype,
                memory_config=_dram_weight_memory_config(self.mesh_device, k=intermediate, n=hidden),
            ),
        )
        replace(
            "mlp_down_prefill",
            upload(down, dtype=policy.mlp_down_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )

        for stale in ("mlp_gate", "mlp_up", "mlp_down"):
            old = self.weights.pop(stale, None)
            if old is not None:
                ttnn.deallocate(old, force=True)

        if self.layer_kind == "full_attention":
            q_width = self.num_heads * self.head_dim
            kv_width = self.num_kv_heads * self.head_dim
            q_and_gate = host("self_attn.q_proj.weight", transpose=True)
            # HF emits [q_head_0, gate_head_0, q_head_1, gate_head_1, ...].
            # Repack that per-head order into the projection family's
            # [all-q, k, v, all-gate] contract.
            q_and_gate = q_and_gate.reshape(hidden, self.num_heads, 2 * self.head_dim)
            q = q_and_gate[..., : self.head_dim].reshape(hidden, q_width)
            gate = q_and_gate[..., self.head_dim :].reshape(hidden, q_width)
            k = host("self_attn.k_proj.weight", transpose=True)
            v = host("self_attn.v_proj.weight", transpose=True)
            # Q/K/V are contiguous for create-heads; gate is the tail.
            packed = torch.cat([q, k, v, gate], dim=-1)
            packed_width = 2 * q_width + 2 * kv_width
            for phase, memcfg in (
                (
                    "decode",
                    _dram_weight_memory_config(self.mesh_device, k=hidden, n=packed_width),
                ),
                ("prefill", ttnn.DRAM_MEMORY_CONFIG),
            ):
                replace(
                    f"qkv_gate_{phase}",
                    upload(packed, dtype=policy.attention_weight_dtype, memory_config=memcfg),
                )
            # Long-context prefill projects one component at a time so packed
            # Q/K/V/gate activations do not consume most of device DRAM.
            for name, value in (("q", q), ("k", k), ("v", v), ("gate", gate)):
                replace(
                    f"{name}_prefill_long",
                    upload(
                        value,
                        dtype=policy.attention_weight_dtype,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                )
            o = host("self_attn.o_proj.weight", transpose=True)
            replace(
                "o_proj_decode",
                upload(
                    o,
                    dtype=policy.attention_weight_dtype,
                    memory_config=_dram_weight_memory_config(self.mesh_device, k=q_width, n=hidden),
                ),
            )
            replace(
                "o_proj_prefill",
                upload(o, dtype=policy.attention_weight_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            )
            for stale in ("q_proj", "k_proj", "v_proj", "o_proj"):
                old = self.weights.pop(stale, None)
                if old is not None:
                    ttnn.deallocate(old, force=True)

            if policy.cache_dtype != ttnn.bfloat16:
                cache_shape = tuple(self.caches["key"].shape)
                for name in ("key", "value"):
                    old = self.caches[name]
                    self.caches[name] = ttnn.zeros(
                        cache_shape,
                        dtype=policy.cache_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    ttnn.deallocate(old, force=True)
        elif policy.linear_packed_decode:
            key_width = int(self.hf_config.linear_num_key_heads) * int(self.hf_config.linear_key_head_dim)
            value_width = int(self.hf_config.linear_num_value_heads) * int(self.hf_config.linear_value_head_dim)
            value_heads = int(self.hf_config.linear_num_value_heads)
            conv_width = 2 * key_width + value_width
            packed = torch.cat(
                [
                    host("linear_attn.in_proj_qkv.weight", transpose=True),
                    host("linear_attn.in_proj_z.weight", transpose=True),
                    host("linear_attn.in_proj_b.weight", transpose=True),
                    host("linear_attn.in_proj_a.weight", transpose=True),
                ],
                dim=-1,
            )
            packed_width = conv_width + value_width + 2 * value_heads
            replace(
                "linear_packed_decode",
                upload(
                    packed,
                    dtype=policy.linear_input_weight_dtype,
                    memory_config=_dram_weight_memory_config(self.mesh_device, k=hidden, n=packed_width),
                ),
            )
            replace(
                "linear_out_decode",
                upload(
                    host("linear_attn.out_proj.weight", transpose=True),
                    dtype=policy.linear_output_weight_dtype,
                    memory_config=_dram_weight_memory_config(self.mesh_device, k=value_width, n=hidden),
                ),
            )

    def _decode_linear(
        self,
        hidden_states,
        weight_name,
        *,
        k,
        n,
        in0_block_w,
        fused_activation=None,
        compute_kernel_config=None,
    ):
        storage_cores = self.policy.decode_storage_cores
        input_memcfg = _l1_width_memory_config(rows=ttnn.TILE_SIZE, width=k, cores=storage_cores)
        if hidden_states.memory_config() != input_memcfg:
            hidden_states = ttnn.to_memory_config(hidden_states, input_memcfg)
        output_memcfg = _l1_width_memory_config(rows=ttnn.TILE_SIZE, width=n, cores=storage_cores)
        if self.policy.advisor_plan == "apply_all":
            advised_outputs = {
                "linear_packed_decode": (103, 160),
                "linear_out_decode": (80, 64),
                "qkv_gate_decode": (90, 160),
                "o_proj_decode": (80, 64),
                "mlp_gate_decode": (109, 160),
                "mlp_up_decode": (109, 160),
                "mlp_down_decode": (80, 64),
            }
            if weight_name in advised_outputs:
                cores, shard_width = advised_outputs[weight_name]
                output_memcfg = _advisor_width_memory_config(
                    rows=ttnn.TILE_SIZE,
                    width=n,
                    cores=cores,
                    shard_width=shard_width,
                )
        return ttnn.linear(
            hidden_states,
            self.weights[weight_name],
            memory_config=output_memcfg,
            program_config=_decode_program(
                k=k,
                n=n,
                in0_block_w=in0_block_w,
                cores=storage_cores,
                fused_activation=fused_activation,
            ),
            compute_kernel_config=compute_kernel_config
            or (
                self.qkv_compute_kernel_config
                if weight_name.startswith("qkv")
                else (
                    self.o_compute_kernel_config if weight_name.startswith("o_proj") else self.mlp_compute_kernel_config
                )
            ),
            dtype=ttnn.bfloat16,
        )

    def _prefill_linear(self, hidden_states, weight_name, *, k, n, fused_activation=None):
        # Program M is based on physical tiles.  For non-aligned sequences,
        # especially serving batch 32, using logical rows underprovisions the
        # grid because every batch row has its own sequence padding.
        rows = math.prod(tuple(hidden_states.padded_shape)[:-1])
        kwargs = dict(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=(
                self.attention_compute_kernel_config
                if weight_name.startswith(("qkv", "o_proj"))
                else self.mlp_compute_kernel_config
            ),
            dtype=ttnn.bfloat16,
        )
        # The serving-size 2-D config's per-core M grows with the full prompt
        # and eventually makes its circular buffers exceed worker L1.  TTNN's
        # general program selection tiles large M without that static-CB
        # growth.  Retain the measured explicit path through B32/S65.
        if rows <= 2048:
            kwargs["program_config"] = _prefill_program(
                rows=rows,
                k=k,
                n=n,
                in0_block_w_limit=self.policy.prefill_in0_block_w,
                grid_y=self.policy.prefill_grid_y,
                fused_activation=fused_activation,
            )
        output = ttnn.linear(hidden_states, self.weights[weight_name], **kwargs)
        if rows > 2048 and fused_activation == ttnn.UnaryOpType.SILU:
            output = ttnn.silu(output)
        return output

    def _rms_norm_decode(self, hidden_states, name):
        if hidden_states.memory_config() != self.decode_norm_memory_config:
            hidden_states = ttnn.to_memory_config(hidden_states, self.decode_norm_memory_config)
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights[name],
            memory_config=self.decode_norm_memory_config,
            program_config=self.decode_norm_program_config,
            compute_kernel_config=self.norm_compute_kernel_config,
        )

    def _partial_rope_prefill(self, tensor, current_positions):
        """Apply partial RoPE while retaining per-row sequence padding.

        Embedding produces a tiled ``[batch, sequence, rotary]`` tensor.  At
        serving batch 32 the physical sequence dimension is padded separately
        for every row, so a logical-only reshape loses volume.  Supplying the
        physical shape keeps non-aligned public sequence lengths valid.
        """
        rotary_dim = int(self.head_dim * float(self.hf_config.partial_rotary_factor))
        rotary = tensor[..., :rotary_dim]
        passthrough = tensor[..., rotary_dim:]
        cos = ttnn.embedding(current_positions, self.rope["cos"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(current_positions, self.rope["sin"], layout=ttnn.TILE_LAYOUT)
        logical_shape = (self.batch, 1, tensor.shape[2], rotary_dim)
        padded_sequence = cos.padded_shape[-2]
        padded_shape = (self.batch, 1, padded_sequence, rotary_dim)
        cos = ttnn.reshape(cos, logical_shape, padded_shape)
        sin = ttnn.reshape(sin, logical_shape, padded_shape)
        heads = tensor.shape[1]
        cos = ttnn.repeat(cos, ttnn.Shape([1, heads, 1, 1]))
        sin = ttnn.repeat(sin, ttnn.Shape([1, heads, 1, 1]))
        rotary = ttnn.add(
            ttnn.multiply(rotary, cos),
            ttnn.multiply(self._rotate_half(rotary), sin),
        )
        return ttnn.concat([rotary, passthrough], dim=-1)

    def _mlp_decode(self, hidden_states):
        if self.policy.packed_gate_up:
            packed = self._decode_linear(
                hidden_states,
                "mlp_gate_up_decode",
                k=self.hidden_size,
                n=2 * self.intermediate_size,
                in0_block_w=self.policy.decode_in0_block_w,
            )
            gate, up = ttnn.split(packed, (self.intermediate_size, self.intermediate_size), dim=-1)
            ttnn.deallocate(packed)
            gate = ttnn.silu(gate)
        else:
            gate = self._decode_linear(
                hidden_states,
                "mlp_gate_decode",
                k=self.hidden_size,
                n=self.intermediate_size,
                in0_block_w=(self.policy.mlp_gate_decode_in0_block_w or self.policy.decode_in0_block_w),
                fused_activation=ttnn.UnaryOpType.SILU,
            )
            up = self._decode_linear(
                hidden_states,
                "mlp_up_decode",
                k=self.hidden_size,
                n=self.intermediate_size,
                in0_block_w=(self.policy.mlp_up_decode_in0_block_w or self.policy.decode_in0_block_w),
            )
        product = ttnn.multiply(
            gate,
            up,
            memory_config=(
                _advisor_width_memory_config(
                    rows=ttnn.TILE_SIZE,
                    width=self.intermediate_size,
                    cores=109,
                    shard_width=160,
                )
                if self.batch == 32
                and self.policy.advisor_plan in ("apply_all", "apply_all_minus_norm", "mlp_product_only")
                else _l1_width_memory_config(
                    rows=ttnn.TILE_SIZE,
                    width=self.intermediate_size,
                    cores=self.policy.decode_storage_cores,
                )
            ),
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return self._decode_linear(
            product,
            "mlp_down_decode",
            k=self.intermediate_size,
            n=self.hidden_size,
            in0_block_w=self.policy.mlp_down_in0_block_w,
        )

    def _mlp_prefill(self, hidden_states):
        sequence = hidden_states.shape[2]
        rows = math.prod(tuple(hidden_states.padded_shape)[:-1])
        if rows > 2048:
            # Bound the otherwise multi-gigabyte gate/up/product live set.
            chunk_sequence = max(
                ttnn.TILE_SIZE,
                (2048 // self.batch // ttnn.TILE_SIZE) * ttnn.TILE_SIZE,
            )
            chunks = []
            start = 0
            while start < sequence:
                end = min(sequence, start + chunk_sequence)
                chunk = ttnn.slice(
                    hidden_states,
                    (0, 0, start, 0),
                    (1, self.batch, end, self.hidden_size),
                )
                chunks.append(self._mlp_prefill(chunk))
                start = end
            return chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2)
        if self.policy.packed_gate_up:
            packed = self._prefill_linear(
                hidden_states,
                "mlp_gate_up_prefill",
                k=self.hidden_size,
                n=2 * self.intermediate_size,
            )
            gate, up = ttnn.split(packed, (self.intermediate_size, self.intermediate_size), dim=-1)
            gate = ttnn.silu(gate)
        else:
            gate = self._prefill_linear(
                hidden_states,
                "mlp_gate_prefill",
                k=self.hidden_size,
                n=self.intermediate_size,
                fused_activation=ttnn.UnaryOpType.SILU,
            )
            up = self._prefill_linear(
                hidden_states,
                "mlp_up_prefill",
                k=self.hidden_size,
                n=self.intermediate_size,
            )
        return self._prefill_linear(
            ttnn.multiply(gate, up),
            "mlp_down_prefill",
            k=self.intermediate_size,
            n=self.hidden_size,
        )

    def _linear_attention_prefill_chunk(self, hidden_states):
        """Preserve the inherited affine scan across a compressed state boundary.

        The correctness-first prefill implementation explicitly expects an
        FP32 destination for its final-state copy.  Reduced-precision
        candidates therefore expand the persistent state before each chunk,
        run the proven implementation unchanged, then compress the completed
        state back into a physical candidate-dtype allocation.  This also
        handles non-aligned prefills spanning multiple 64-token chunks.
        """
        state_dtype = self.policy.linear_recurrent_state_dtype
        if state_dtype == ttnn.float32:
            return FunctionalDecoder._linear_attention_prefill_chunk(self, hidden_states)

        persistent_state = self.caches["recurrent"]
        working_state = ttnn.typecast(persistent_state, ttnn.float32)
        self.caches["recurrent"] = working_state
        ttnn.deallocate(persistent_state)
        output = FunctionalDecoder._linear_attention_prefill_chunk(self, hidden_states)
        completed_state = self.caches["recurrent"]
        self.caches["recurrent"] = ttnn.typecast(completed_state, state_dtype)
        ttnn.deallocate(completed_state)
        return output

    def _linear_attention_decode(self, hidden_states):
        """Decode gated-delta attention with one same-input projection family."""
        if not (self.policy.linear_packed_decode or self.policy.linear_outer_product):
            return FunctionalDecoder._linear_attention_decode(self, hidden_states)

        key_heads = int(self.hf_config.linear_num_key_heads)
        value_heads = int(self.hf_config.linear_num_value_heads)
        key_dim = int(self.hf_config.linear_key_head_dim)
        value_dim = int(self.hf_config.linear_value_head_dim)
        key_width = key_heads * key_dim
        value_width = value_heads * value_dim
        conv_width = 2 * key_width + value_width
        packed_width = conv_width + value_width + 2 * value_heads

        if self.policy.linear_packed_decode:
            packed = self._decode_linear(
                hidden_states,
                "linear_packed_decode",
                k=self.hidden_size,
                n=packed_width,
                in0_block_w=self.policy.linear_packed_in0_block_w,
                compute_kernel_config=self.linear_input_compute_kernel_config,
            )
            # Conv, reshape, and recurrent composite kernels currently require
            # interleaved tensors; cross that boundary once after the packed
            # projection instead of four times before four independent matmuls.
            packed = ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG)
            mixed, z, beta, decay = ttnn.split(
                packed,
                (conv_width, value_width, value_heads, value_heads),
                dim=-1,
            )
            ttnn.deallocate(packed)
        else:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            mixed = ttnn.linear(
                hidden_states,
                self.weights["in_qkv"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            z = ttnn.linear(
                hidden_states,
                self.weights["in_z"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            beta = ttnn.linear(
                hidden_states,
                self.weights["in_b"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            decay = ttnn.linear(
                hidden_states,
                self.weights["in_a"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        mixed = ttnn.permute(mixed, (0, 2, 3, 1))
        next_conv_state = ttnn.concat(
            [self.caches["conv"][..., 1:], mixed],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mixed = ttnn.sum(
            ttnn.multiply(next_conv_state, self.weights["conv"]),
            dim=-1,
            keepdim=True,
        )
        mixed = ttnn.silu(mixed)
        ttnn.copy(next_conv_state, self.caches["conv"])
        mixed = ttnn.permute(mixed, (0, 3, 1, 2))

        query = mixed[..., :key_width]
        key = mixed[..., key_width : 2 * key_width]
        value = mixed[..., 2 * key_width :]
        query = ttnn.reshape(query, (self.batch, 1, key_heads, key_dim))
        key = ttnn.reshape(key, (self.batch, 1, key_heads, key_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.reshape(value, (self.batch, 1, value_heads, value_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))
        repeats = value_heads // key_heads
        query = ttnn.repeat_interleave(query, repeats, dim=1)
        key = ttnn.repeat_interleave(key, repeats, dim=1)
        query = self._l2_norm(query)
        key = self._l2_norm(key)
        query = ttnn.multiply(query, key_dim**-0.5)

        beta = ttnn.sigmoid(beta)
        decay = ttnn.multiply(
            self.weights["a"],
            ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"])),
        )
        beta = ttnn.reshape(beta, (self.batch, value_heads, 1, 1))
        decay = ttnn.reshape(decay, (self.batch, value_heads, 1, 1))
        decay = ttnn.exp(decay)

        state_dtype = self.policy.linear_recurrent_state_dtype
        recurrent_state = self.caches["recurrent"]
        if state_dtype != ttnn.float32:
            # The recurrent matmuls are BF16.  Make the reduced-precision
            # storage boundary explicit and avoid relying on mixed FP32/BFP
            # elementwise promotion rules.
            recurrent_state = ttnn.typecast(recurrent_state, ttnn.bfloat16)
            decay = ttnn.typecast(decay, ttnn.bfloat16)
            beta = ttnn.typecast(beta, ttnn.bfloat16)
        recurrent = ttnn.multiply(recurrent_state, decay)
        memory_value = self._linear_recurrent_matmul(key, recurrent)
        delta = ttnn.multiply(ttnn.subtract(value, memory_value), beta)
        key_transposed = ttnn.transpose(key, -2, -1)
        if self.policy.linear_outer_product:
            # This is a K=1 outer product.  Elementwise broadcast avoids the
            # matmul's tile-padded reduction and is mathematically identical.
            update = ttnn.multiply(
                key_transposed,
                delta,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            update = ttnn.matmul(key_transposed, delta)
        recurrent = ttnn.add(recurrent, update)
        output = self._linear_recurrent_matmul(query, recurrent)
        if state_dtype == ttnn.float32:
            ttnn.copy(recurrent, self.caches["recurrent"])
        else:
            stored_recurrent = ttnn.typecast(recurrent, state_dtype)
            ttnn.copy(stored_recurrent, self.caches["recurrent"])

        output = ttnn.rms_norm(
            output,
            epsilon=self.eps,
            weight=self.weights["gated_norm"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = ttnn.reshape(z, (self.batch, value_heads, 1, value_dim))
        output = ttnn.multiply(output, ttnn.silu(z))
        output = ttnn.permute(output, (2, 0, 1, 3))
        output = ttnn.reshape(output, (1, 1, self.batch, value_width))
        if self.policy.linear_packed_decode:
            return self._decode_linear(
                output,
                "linear_out_decode",
                k=value_width,
                n=self.hidden_size,
                in0_block_w=self.policy.linear_out_in0_block_w,
                compute_kernel_config=self.linear_output_compute_kernel_config,
            )
        return ttnn.linear(
            output,
            self.weights["out_proj"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _linear_recurrent_matmul(self, left, right):
        mode = self.policy.linear_recurrent_program
        kwargs = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "compute_kernel_config": self.linear_recurrent_compute_kernel_config,
            "dtype": ttnn.bfloat16,
        }
        if mode != "auto":
            if mode == "grid2_n2":
                grid = (2, 1)
                block_w = 2
                per_core_n = 2
                subblock_w = 2
            else:
                grid = (4, 1)
                block_w = int(mode.rsplit("w", 1)[1])
                per_core_n = 1
                subblock_w = 1
            kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=block_w,
                out_subblock_h=1,
                out_subblock_w=subblock_w,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
        return ttnn.matmul(left, right, **kwargs)

    def _full_attention_decode(self, hidden_states, page_table, current_positions):
        cache_positions = ttnn.typecast(current_positions, ttnn.int32)
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        packed = self._decode_linear(
            hidden_states,
            "qkv_gate_decode",
            k=self.hidden_size,
            n=2 * q_width + 2 * kv_width,
            in0_block_w=(self.policy.qkv_decode_in0_block_w or self.policy.decode_in0_block_w),
        )
        qkv, gate = ttnn.split(packed, (q_width + 2 * kv_width, q_width), dim=-1)
        ttnn.deallocate(packed)
        # nlp_create_qkv_heads_decode interprets its packed Q/K/V axis as an
        # interleaved row.  Passing the DRAM-matmul's width-sharded output is
        # accepted by the API but silently assigns the wrong values to heads
        # for dense real weights (the diagonal synthetic fixture masked it).
        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=self.decode_attention_memory_config,
        )
        ttnn.deallocate(qkv)
        q = self._per_head_norm(q, "q_norm")
        k = self._per_head_norm(k, "k_norm")
        q = self._partial_rope_decode(q, current_positions)
        k = self._partial_rope_decode(k, current_positions)
        ttnn.experimental.paged_update_cache(
            self.caches["key"], k, update_idxs_tensor=cache_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            self.caches["value"], v, update_idxs_tensor=cache_positions, page_table=page_table
        )
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.caches["key"],
            self.caches["value"],
            cur_pos_tensor=cache_positions,
            page_table_tensor=page_table,
            scale=self.head_dim**-0.5,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.attention_compute_kernel_config,
            # GQA decode currently rejects a sharded SDPA output.  Keep the
            # explicit program config, then cross this narrow helper boundary.
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        attention = ttnn.to_memory_config(attention, self.decode_attention_memory_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=self.num_heads)
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        ttnn.deallocate(gate)
        attention = self._decode_linear(
            attention,
            "o_proj_decode",
            k=q_width,
            n=self.hidden_size,
            in0_block_w=self.policy.o_decode_in0_block_w,
        )
        return ttnn.reshape(
            attention,
            (1, 1, self.batch, self.hidden_size),
            (1, 1, ttnn.TILE_SIZE, self.hidden_size),
        )

    def _full_attention_prefill(self, hidden_states, page_table, current_positions):
        sequence = hidden_states.shape[2]
        if sequence > 32768:
            return self._full_attention_prefill_long(hidden_states, page_table, current_positions)
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        packed = self._prefill_linear(
            hidden_states,
            "qkv_gate_prefill",
            k=self.hidden_size,
            n=2 * q_width + 2 * kv_width,
        )
        qkv, gate = ttnn.split(packed, (q_width + 2 * kv_width, q_width), dim=-1)
        q, k, v = ttnn.split(qkv, (q_width, kv_width, kv_width), dim=-1)
        q = ttnn.permute(ttnn.reshape(q, (self.batch, sequence, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (self.batch, sequence, self.num_kv_heads, self.head_dim)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (self.batch, sequence, self.num_kv_heads, self.head_dim)), (0, 2, 1, 3))
        q = self._partial_rope_prefill(self._per_head_norm_prefill(q, "q_norm"), current_positions)
        k = self._partial_rope_prefill(self._per_head_norm_prefill(k, "k_norm"), current_positions)
        cache_k = ttnn.typecast(k, self.policy.cache_dtype)
        cache_v = ttnn.typecast(v, self.policy.cache_dtype)
        ttnn.experimental.paged_fill_cache(
            self.caches["key"], cache_k, page_table, batch_idx_tensor=self.caches["batch_indices"]
        )
        ttnn.experimental.paged_fill_cache(
            self.caches["value"], cache_v, page_table, batch_idx_tensor=self.caches["batch_indices"]
        )
        attention = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=64,
                k_chunk_size=64,
            ),
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.experimental.nlp_concat_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.permute(attention, (1, 0, 2, 3))
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        return self._prefill_linear(
            attention,
            "o_proj_prefill",
            k=q_width,
            n=self.hidden_size,
        )

    def _full_attention_prefill_long(self, hidden_states, page_table, current_positions):
        """Memory-bounded paged attention for prompts above ordinary SDPA's limit."""
        sequence = hidden_states.shape[2]
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        chunk_size = 32768
        page_size = self.caches["key"].shape[2]

        # Populate paged K/V first. Chunk boundaries are page-aligned, so a
        # sliced page table maps each local fill directly to its global blocks.
        start = 0
        while start < sequence:
            end = min(sequence, start + chunk_size)
            logical_chunk = end - start
            hidden_chunk = ttnn.slice(
                hidden_states,
                (0, 0, start, 0),
                (1, self.batch, end, self.hidden_size),
            )
            position_chunk = ttnn.slice(
                current_positions,
                (0, start),
                (self.batch, end),
            )
            page_chunk = ttnn.slice(
                page_table,
                (0, start // page_size),
                (self.batch, math.ceil(end / page_size)),
            )
            k = self._prefill_linear(
                hidden_chunk,
                "k_prefill_long",
                k=self.hidden_size,
                n=kv_width,
            )
            k = ttnn.permute(
                ttnn.reshape(
                    k,
                    (self.batch, logical_chunk, self.num_kv_heads, self.head_dim),
                ),
                (0, 2, 1, 3),
            )
            k = self._partial_rope_prefill(self._per_head_norm_prefill(k, "k_norm"), position_chunk)
            k = ttnn.typecast(k, self.policy.cache_dtype)
            v = self._prefill_linear(
                hidden_chunk,
                "v_prefill_long",
                k=self.hidden_size,
                n=kv_width,
            )
            v = ttnn.permute(
                ttnn.reshape(
                    v,
                    (self.batch, logical_chunk, self.num_kv_heads, self.head_dim),
                ),
                (0, 2, 1, 3),
            )
            v = ttnn.typecast(v, self.policy.cache_dtype)
            ttnn.experimental.paged_fill_cache(
                self.caches["key"],
                k,
                page_chunk,
                batch_idx_tensor=self.caches["batch_indices"],
            )
            ttnn.experimental.paged_fill_cache(
                self.caches["value"],
                v,
                page_chunk,
                batch_idx_tensor=self.caches["batch_indices"],
            )
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            start = end

        outputs = []
        start = 0
        while start < sequence:
            end = min(sequence, start + chunk_size)
            logical_chunk = end - start
            hidden_chunk = ttnn.slice(
                hidden_states,
                (0, 0, start, 0),
                (1, self.batch, end, self.hidden_size),
            )
            position_chunk = ttnn.slice(
                current_positions,
                (0, start),
                (self.batch, end),
            )
            q = self._prefill_linear(
                hidden_chunk,
                "q_prefill_long",
                k=self.hidden_size,
                n=q_width,
            )
            q = ttnn.permute(
                ttnn.reshape(
                    q,
                    (self.batch, logical_chunk, self.num_heads, self.head_dim),
                ),
                (0, 2, 1, 3),
            )
            q = self._partial_rope_prefill(self._per_head_norm_prefill(q, "q_norm"), position_chunk)
            padding = (-logical_chunk) % ttnn.TILE_SIZE
            if padding:
                q = ttnn.pad(
                    q,
                    ((0, 0), (0, 0), (0, padding), (0, 0)),
                    value=0.0,
                )
            attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                self.caches["key"],
                self.caches["value"],
                page_table,
                chunk_start_idx=start,
                scale=self.head_dim**-0.5,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            if padding:
                attention = ttnn.slice(
                    attention,
                    (0, 0, 0, 0),
                    (self.batch, self.num_heads, logical_chunk, self.head_dim),
                )
            attention = ttnn.experimental.nlp_concat_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attention = ttnn.permute(attention, (1, 0, 2, 3))
            gate = self._prefill_linear(
                hidden_chunk,
                "gate_prefill_long",
                k=self.hidden_size,
                n=q_width,
            )
            attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
            outputs.append(
                self._prefill_linear(
                    attention,
                    "o_proj_prefill",
                    k=q_width,
                    n=self.hidden_size,
                )
            )
            start = end
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def prefill_forward(
        self,
        *,
        hidden_states,
        page_table,
        current_positions,
        sequence_mask=None,
        conv_state_selectors=None,
        cache_page_table=None,
    ):
        # Mixed-length serving metadata is consumed by the stateful linear
        # mixer.  Keep it out of the established TP projection signatures.
        self._sequence_masks = sequence_mask if isinstance(sequence_mask, (list, tuple)) else None
        self._conv_state_selector_chunks = (
            conv_state_selectors
            if conv_state_selectors and isinstance(conv_state_selectors[0], (list, tuple))
            else None
        )
        self._sequence_mask = sequence_mask if self._sequence_masks is None else None
        self._conv_state_selectors = conv_state_selectors if self._conv_state_selector_chunks is None else None
        # Attention reads retain the caller's ordinary page table.  Cache
        # fills may use -1 entries to skip inactive fixed slots without
        # disturbing a live peer during slot reuse.
        self._cache_page_table = page_table if cache_page_table is None else cache_page_table
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "input_norm")
        hidden_states = self._token_mixer_prefill(hidden_states, page_table, current_positions)
        hidden_states = ttnn.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "post_attention_norm")
        hidden_states = self._mlp_prefill(hidden_states)
        return ttnn.add(residual, hidden_states)

    def decode_forward(self, *, hidden_states, page_table, current_positions, active_mask=None):
        self._active_mask = active_mask
        residual = ttnn.to_memory_config(hidden_states, self.decode_residual_memory_config)
        hidden_states = self._rms_norm_decode(residual, "input_norm")
        if self.layer_kind == "linear_attention" and not self.policy.linear_packed_decode:
            # Gated-delta's stateful composite currently requires interleaved
            # tensors at its conv/recurrent boundary.  The conversion is
            # measured as part of the coherent candidate.
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._token_mixer_decode(hidden_states, page_table, current_positions)
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memory_config)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=self.decode_residual_memory_config)
        residual = hidden_states
        hidden_states = self._rms_norm_decode(hidden_states, "post_attention_norm")
        hidden_states = self._mlp_decode(hidden_states)
        return ttnn.add(residual, hidden_states, memory_config=self.decode_residual_memory_config)


__all__ = [
    "ADVERTISED_CONTEXT",
    "LINEAR_PREFILL_CHUNK_SIZE",
    "MODEL_ID",
    "MODEL_REVISION",
    "OptimizationPolicy",
    "OptimizedDecoder",
    "POLICIES",
    "REPRESENTATIVE_LAYERS",
    "resolve_policy",
]
