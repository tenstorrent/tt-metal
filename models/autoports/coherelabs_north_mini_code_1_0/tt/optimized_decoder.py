# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device North-Mini decoder layer.

The implementation starts from :class:`FusedDecoder` and keeps its packed QKV,
packed gate/up, fused cache update, and non-aligned prefill contracts.  The
batch-1 MoE decode path additionally carries the exact top-8 routing mask
through both sparse projections, so down projection evaluates active experts
instead of the fused stage's exact-but-expensive all-128-expert fallback.

Candidate policies are construction-time controls used by the stage's
precision/fidelity and sparse-geometry sweeps.  Runtime forwards contain no
host conversions or functional-decoder fallback.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.fused_decoder import FusedDecoder


@dataclass(frozen=True)
class OptimizationPolicy:
    attention_weight_dtype: object = ttnn.bfloat16
    dense_weight_dtype: object = ttnn.bfloat16
    expert_weight_dtype: object = ttnn.bfloat16
    router_weight_dtype: object = ttnn.bfloat16
    activation_dtype: object = ttnn.bfloat16
    attention_fidelity: object = ttnn.MathFidelity.HiFi2
    dense_decode_attention_fidelity: object | None = None
    dense_fidelity: object = ttnn.MathFidelity.HiFi2
    expert_fidelity: object = ttnn.MathFidelity.LoFi
    expert_fp32_dest_acc: bool = False
    gate_up_cores: int = 24
    gate_up_in0_block_w: int = 16
    down_cores: int = 32
    down_in0_block_w: int = 24
    cache_dtype: object = ttnn.bfloat16
    explicit_decode_sdpa: bool = False
    explicit_prefill_sdpa: bool = False
    dram_sharded_dense_decode: bool = False
    sharded_dense_residual: bool = False
    dram_sharded_moe_attention: bool = False
    dram_k_block_cap: int = 32
    sparse_output_tile_height: int = 32
    sparse_l1_chain: bool = False
    prefill_moe_grid_scale: int = 0
    prefill_gate_up_in0_block_w: int = 2
    prefill_down_in0_block_w: int = 2
    advisor_moe_norm_cores: int = 0


POLICIES = {
    "default": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        dense_decode_attention_fidelity=ttnn.MathFidelity.LoFi,
        dense_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded_dense_decode=True,
        sharded_dense_residual=True,
        sparse_l1_chain=True,
        prefill_moe_grid_scale=8,
        prefill_gate_up_in0_block_w=8,
        prefill_down_in0_block_w=8,
        advisor_moe_norm_cores=32,
    ),
    "bf16_control": OptimizationPolicy(),
    "bf16_hifi2_control": OptimizationPolicy(expert_fidelity=ttnn.MathFidelity.HiFi2),
    "bf16_hifi4_fp32_control": OptimizationPolicy(
        expert_fidelity=ttnn.MathFidelity.HiFi4,
        expert_fp32_dest_acc=True,
    ),
    "fused_baseline": OptimizationPolicy(gate_up_cores=48, gate_up_in0_block_w=8, down_cores=64, down_in0_block_w=8),
    "expert_bfp8": OptimizationPolicy(expert_weight_dtype=ttnn.bfloat8_b),
    "expert_bfp8_hifi2": OptimizationPolicy(
        expert_weight_dtype=ttnn.bfloat8_b,
        expert_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "expert_bfp8_geometry_48_64": OptimizationPolicy(
        expert_weight_dtype=ttnn.bfloat8_b,
        gate_up_cores=48,
        gate_up_in0_block_w=16,
        down_cores=64,
        down_in0_block_w=24,
    ),
    "expert_bfp8_geometry_16_16": OptimizationPolicy(
        expert_weight_dtype=ttnn.bfloat8_b,
        gate_up_cores=16,
        gate_up_in0_block_w=16,
        down_cores=16,
        down_in0_block_w=24,
    ),
    "expert_bfp4": OptimizationPolicy(expert_weight_dtype=ttnn.bfloat4_b),
    "expert_bfp4_geometry_48_64": OptimizationPolicy(
        expert_weight_dtype=ttnn.bfloat4_b,
        gate_up_cores=48,
        gate_up_in0_block_w=16,
        down_cores=64,
        down_in0_block_w=24,
    ),
    "expert_bfp4_geometry_16_16": OptimizationPolicy(
        expert_weight_dtype=ttnn.bfloat4_b,
        gate_up_cores=16,
        gate_up_in0_block_w=16,
        down_cores=16,
        down_in0_block_w=24,
    ),
    "attention_bfp8_hifi2": OptimizationPolicy(attention_weight_dtype=ttnn.bfloat8_b),
    "attention_bfp8_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "attention_bfp4_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "dense_bfp8_hifi2": OptimizationPolicy(dense_weight_dtype=ttnn.bfloat8_b),
    "dense_bfp8_lofi": OptimizationPolicy(
        dense_weight_dtype=ttnn.bfloat8_b,
        dense_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "dense_bfp4_lofi": OptimizationPolicy(
        dense_weight_dtype=ttnn.bfloat4_b,
        dense_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "cache_bfp8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
    ),
    "sdpa_explicit": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        explicit_decode_sdpa=True,
        explicit_prefill_sdpa=True,
    ),
    "dram_sharded_dense_decode": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        dram_sharded_dense_decode=True,
    ),
    "dram_sharded_moe_attention": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        dram_sharded_moe_attention=True,
    ),
    "dram_sharded_dense_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        dense_decode_attention_fidelity=ttnn.MathFidelity.LoFi,
        dense_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded_dense_decode=True,
        sharded_dense_residual=True,
    ),
    "dram_sharded_dense_bfp4_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        dense_weight_dtype=ttnn.bfloat4_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        dense_decode_attention_fidelity=ttnn.MathFidelity.LoFi,
        dense_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded_dense_decode=True,
        sharded_dense_residual=True,
    ),
    "dram_sharded_dense_small_k": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        dram_sharded_dense_decode=True,
        sharded_dense_residual=True,
        dram_k_block_cap=4,
    ),
    "router_bfp8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
    ),
    "sharded_dense_residual": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        dram_sharded_dense_decode=True,
        sharded_dense_residual=True,
    ),
    "activation_bfp8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat8_b,
    ),
    "sparse_tile_h1": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        sparse_output_tile_height=1,
    ),
    "sparse_tile_h16": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        sparse_output_tile_height=16,
    ),
    "sparse_l1_chain": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        sparse_l1_chain=True,
    ),
    "prefill_moe_grid_48_64": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        prefill_moe_grid_scale=4,
    ),
    "prefill_moe_grid_24_32": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        prefill_moe_grid_scale=8,
    ),
    "prefill_moe_grid_24_32_block4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        prefill_moe_grid_scale=8,
        prefill_gate_up_in0_block_w=4,
        prefill_down_in0_block_w=4,
    ),
    "prefill_moe_grid_24_32_block8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        prefill_moe_grid_scale=8,
        prefill_gate_up_in0_block_w=8,
        prefill_down_in0_block_w=8,
    ),
    "prefill_moe_grid_24_32_block16_12": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        router_weight_dtype=ttnn.bfloat8_b,
        prefill_moe_grid_scale=8,
        prefill_gate_up_in0_block_w=16,
        prefill_down_in0_block_w=12,
    ),
    "geometry_48_64": OptimizationPolicy(
        gate_up_cores=48,
        gate_up_in0_block_w=16,
        down_cores=64,
        down_in0_block_w=24,
    ),
    "geometry_16_16": OptimizationPolicy(
        gate_up_cores=16,
        gate_up_in0_block_w=16,
        down_cores=16,
        down_in0_block_w=24,
    ),
}
POLICIES.update(
    {f"advisor_moe_norm_{cores}": replace(POLICIES["default"], advisor_moe_norm_cores=cores) for cores in (22, 32, 64)}
)


def _rectangular_grid(num_cores: int, grid) -> ttnn.CoreCoord:
    for x in range(min(grid.x, num_cores), 0, -1):
        if num_cores % x == 0 and num_cores // x <= grid.y:
            return ttnn.CoreCoord(x, num_cores // x)
    raise ValueError(f"{num_cores} cores do not form a rectangle on {grid}")


def _sparse_program_config(*, m: int, k: int, n: int, num_cores: int, in0_block_w: int, grid):
    k_tiles = math.ceil(k / ttnn.TILE_SIZE)
    n_tiles = math.ceil(n / ttnn.TILE_SIZE)
    if k_tiles % in0_block_w:
        raise ValueError(f"K tiles {k_tiles} must divide in0_block_w={in0_block_w}")
    if n_tiles % num_cores:
        raise ValueError(f"N tiles {n_tiles} must divide num_cores={num_cores}")
    per_core_n = n_tiles // num_cores
    out_subblock_w = next(value for value in (4, 3, 2, 1) if per_core_n % value == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=_rectangular_grid(num_cores, grid),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=max(ttnn.TILE_SIZE, m) // ttnn.TILE_SIZE,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _replace_dtype(weights, name, dtype):
    old = weights[name]
    if old.dtype == dtype:
        return
    replacement = ttnn.typecast(old, dtype, memory_config=old.memory_config())
    old.deallocate(True)
    weights[name] = replacement


def _dram_grid(mesh_device):
    dram = mesh_device.dram_grid_size()
    if dram.y != 1:
        raise RuntimeError(f"DRAM-sharded decode expects a one-row DRAM grid, got {dram}")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, 0))})


def _dram_weight_memory_config(mesh_device, *, k: int, n: int):
    grid = _dram_grid(mesh_device)
    cores = grid.num_cores()
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * cores)) * ttnn.TILE_SIZE * cores
    shard = ttnn.ShardSpec(grid, (k, padded_n // cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)


def _dram_input_memory_config(mesh_device, *, k: int):
    grid = _dram_grid(mesh_device)
    shard = ttnn.ShardSpec(grid, (ttnn.TILE_SIZE, k // grid.num_cores()), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)


def _dram_matmul_program_config(mesh_device, *, k: int, n: int, k_block_cap: int):
    cores = _dram_grid(mesh_device).num_cores()
    k_tiles_per_core = k // (ttnn.TILE_SIZE * cores)
    in0_block_w = next(
        value
        for value in (32, 24, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1)
        if value <= k_block_cap and k_tiles_per_core % value == 0
    )
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * cores)),
        fused_activation=None,
    )


def _prefill_moe_program_config(*, m: int, n: int, grid_x: int, in0_block_w: int):
    m_tiles = math.ceil(m / ttnn.TILE_SIZE)
    grid_y = next(value for value in range(min(8, m_tiles), 0, -1) if m_tiles % value == 0)
    n_tiles = math.ceil(n / ttnn.TILE_SIZE)
    if n_tiles % grid_x:
        raise ValueError(f"N tiles {n_tiles} must divide grid_x={grid_x}")
    per_core_n = n_tiles // grid_x
    out_subblock_w = next(value for value in (4, 2, 1) if per_core_n % value == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=m_tiles // grid_y,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


class OptimizedDecoder(FusedDecoder):
    """Fused North-Mini decoder with active-expert batch-1 decode."""

    @classmethod
    def from_state_dict(cls, state_dict, *, candidate=None, **kwargs):
        candidate = candidate or os.environ.get("NORTH_MINI_OPTIMIZATION_CANDIDATE", "default")
        if candidate not in POLICIES:
            raise ValueError(f"unknown optimized candidate {candidate!r}; expected one of {sorted(POLICIES)}")
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.candidate = candidate
        decoder.policy = POLICIES[candidate]
        decoder.optimized_batch1_moe_calls = 0

        if decoder.mlp_type == "dense":
            _replace_dtype(decoder.weights, "gate_up", decoder.policy.dense_weight_dtype)
            _replace_dtype(decoder.weights, "down_proj", decoder.policy.dense_weight_dtype)
        else:
            _replace_dtype(decoder.weights, "expert_gate_up", decoder.policy.expert_weight_dtype)
            _replace_dtype(decoder.weights, "expert_down", decoder.policy.expert_weight_dtype)
            _replace_dtype(decoder.weights, "router", decoder.policy.router_weight_dtype)
        _replace_dtype(decoder.weights, "qkv", decoder.policy.attention_weight_dtype)
        _replace_dtype(decoder.weights, "o", decoder.policy.attention_weight_dtype)
        decoder.dram_sharded_decode_configs = {}
        decoder.advisor_moe_norm_memory_config = None
        decoder.advisor_moe_norm_program_config = None
        if decoder.mlp_type == "sparse" and decoder.policy.advisor_moe_norm_cores:
            cores = decoder.policy.advisor_moe_norm_cores
            grid = _rectangular_grid(cores, decoder.mesh_device.compute_with_storage_grid_size())
            core_ranges = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
            )
            decoder.advisor_moe_norm_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    core_ranges,
                    (ttnn.TILE_SIZE, math.ceil(decoder.hidden_size / (ttnn.TILE_SIZE * cores)) * ttnn.TILE_SIZE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            decoder.advisor_moe_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid,
                subblock_w=1,
                block_h=1,
                block_w=math.ceil(decoder.hidden_size / ttnn.TILE_SIZE / cores),
                inplace=False,
            )
        decoder.use_dram_sharded_dense_decode = decoder.batch == 1 and (
            (decoder.policy.dram_sharded_dense_decode and decoder.mlp_type == "dense")
            or (decoder.policy.dram_sharded_moe_attention and decoder.mlp_type == "sparse")
        )
        decoder.use_sharded_dense_residual = (
            decoder.policy.sharded_dense_residual and decoder.use_dram_sharded_dense_decode
        )
        if decoder.use_dram_sharded_dense_decode:
            names_and_shapes = {
                "qkv": (decoder.hidden_size, decoder.weights["qkv"].shape[-1]),
                "o": (decoder.weights["o"].shape[-2], decoder.hidden_size),
            }
            if decoder.mlp_type == "dense":
                names_and_shapes.update(
                    {
                        "gate_up": (decoder.hidden_size, 2 * decoder.intermediate_size),
                        "down_proj": (decoder.intermediate_size, decoder.hidden_size),
                    }
                )
            for name, (k, n) in names_and_shapes.items():
                decoder.weights[f"{name}_decode"] = ttnn.to_memory_config(
                    decoder.weights[name],
                    _dram_weight_memory_config(decoder.mesh_device, k=k, n=n),
                )
                decoder.dram_sharded_decode_configs[name] = (
                    _dram_input_memory_config(decoder.mesh_device, k=k),
                    _dram_matmul_program_config(
                        decoder.mesh_device,
                        k=k,
                        n=n,
                        k_block_cap=decoder.policy.dram_k_block_cap,
                    ),
                    _dram_input_memory_config(decoder.mesh_device, k=n),
                )
            decoder.dense_residual_memory_config = decoder.dram_sharded_decode_configs["qkv"][0]
            decoder.dense_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=decoder.dense_residual_memory_config.shard_spec.grid.bounding_box().grid_size(),
                subblock_w=4,
                block_h=1,
                block_w=decoder.hidden_size
                // decoder.dense_residual_memory_config.shard_spec.grid.num_cores()
                // ttnn.TILE_SIZE,
                inplace=False,
            )

        decoder.attention_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            decoder.mesh_device.arch(),
            math_fidelity=(
                decoder.policy.dense_decode_attention_fidelity
                if decoder.mlp_type == "dense"
                and decoder.batch == 1
                and decoder.policy.dense_decode_attention_fidelity is not None
                else decoder.policy.attention_fidelity
            ),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.prefill_attention_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            decoder.mesh_device.arch(),
            math_fidelity=decoder.policy.attention_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.dense_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            decoder.mesh_device.arch(),
            math_fidelity=decoder.policy.dense_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.prefill_dense_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            decoder.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            decoder.mesh_device.arch(),
            math_fidelity=decoder.policy.expert_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=decoder.policy.expert_fp32_dest_acc,
            packer_l1_acc=True,
        )
        decoder.prefill_expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            decoder.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.decode_sdpa_program_config = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )
            if decoder.policy.explicit_decode_sdpa
            else None
        )
        if decoder.mlp_type == "sparse":
            grid = decoder.mesh_device.compute_with_storage_grid_size()
            decoder.optimized_sparse_gate_up_program_config = _sparse_program_config(
                m=1,
                k=decoder.hidden_size,
                n=2 * decoder.intermediate_size,
                num_cores=decoder.policy.gate_up_cores,
                in0_block_w=decoder.policy.gate_up_in0_block_w,
                grid=grid,
            )
            decoder.optimized_sparse_down_program_config = _sparse_program_config(
                m=1,
                k=decoder.intermediate_size,
                n=decoder.hidden_size,
                num_cores=decoder.policy.down_cores,
                in0_block_w=decoder.policy.down_in0_block_w,
                grid=grid,
            )
        return decoder

    def create_paged_kv_cache(self, *, num_blocks: int | None = None):
        min_blocks = self.batch * math.ceil(self.max_cache_len / self.page_size)
        num_blocks = min_blocks if num_blocks is None else int(num_blocks)
        if num_blocks < min_blocks:
            raise ValueError(f"num_blocks={num_blocks} cannot cover required {min_blocks} blocks")
        shape = (num_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        return tuple(
            ttnn.zeros(
                shape,
                dtype=self.policy.cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

    def _attention_prefill(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos,
        position_sin,
        seq_len,
    ):
        query, key, value = self._qkv_prefill(normalized, seq_len, position_cos, position_sin)
        cache_key = key if self.policy.cache_dtype == ttnn.bfloat16 else ttnn.typecast(key, self.policy.cache_dtype)
        cache_value = (
            value if self.policy.cache_dtype == ttnn.bfloat16 else ttnn.typecast(value, self.policy.cache_dtype)
        )
        for user in range(self.batch):
            key_user = ttnn.slice(
                cache_key,
                (user, 0, 0, 0),
                (user + 1, self.num_kv_heads, seq_len, self.head_dim),
            )
            value_user = ttnn.slice(
                cache_value,
                (user, 0, 0, 0),
                (user + 1, self.num_kv_heads, seq_len, self.head_dim),
            )
            ttnn.experimental.paged_fill_cache(key_cache, key_user, page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(value_cache, value_user, page_table, batch_idx=user)

        program_config = None
        if self.policy.explicit_prefill_sdpa:
            chunk = 256 if seq_len >= 2048 else 64
            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                exp_approx_mode=False,
                q_chunk_size=chunk,
                k_chunk_size=chunk,
            )
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            program_config=program_config,
            compute_kernel_config=self.prefill_attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.prefill_attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _qkv_decode(self, normalized, position_cos, position_sin):
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        program_config = None
        if self.use_dram_sharded_dense_decode:
            input_memory_config, program_config, memory_config = self.dram_sharded_decode_configs["qkv"]
            if os.environ.get("NORTH_MINI_ADVISOR_CAPTURE") == "1" or normalized.memory_config() != input_memory_config:
                normalized = ttnn.to_memory_config(normalized, input_memory_config)
        fused = ttnn.linear(
            normalized,
            self.weights["qkv_decode"] if self.use_dram_sharded_dense_decode else self.weights["qkv"],
            dtype=ttnn.bfloat16,
            program_config=program_config,
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=memory_config,
        )
        if self.use_dram_sharded_dense_decode and not self.use_sharded_dense_residual:
            fused = ttnn.to_memory_config(fused, ttnn.DRAM_MEMORY_CONFIG)
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

    def _dense_mlp(self, normalized):
        gate_up_memory_config = ttnn.DRAM_MEMORY_CONFIG
        gate_up_program_config = None
        if self.use_dram_sharded_dense_decode and normalized.shape[-2] == 1:
            input_memory_config, gate_up_program_config, gate_up_memory_config = self.dram_sharded_decode_configs[
                "gate_up"
            ]
            if os.environ.get("NORTH_MINI_ADVISOR_CAPTURE") == "1" or normalized.memory_config() != input_memory_config:
                normalized = ttnn.to_memory_config(normalized, input_memory_config)
        gate_up = ttnn.linear(
            normalized,
            self.weights["gate_up_decode"] if gate_up_program_config is not None else self.weights["gate_up"],
            dtype=ttnn.bfloat16,
            program_config=gate_up_program_config,
            compute_kernel_config=(
                self.dense_compute_kernel_config
                if gate_up_program_config is not None
                else self.prefill_dense_compute_kernel_config
            ),
            memory_config=gate_up_memory_config,
        )
        gate, up = self._split_gate_up(gate_up, self.intermediate_size)
        activated = self._fused_swiglu(gate, up)
        down_memory_config = ttnn.DRAM_MEMORY_CONFIG
        down_program_config = None
        if self.use_dram_sharded_dense_decode and normalized.shape[-2] == 1:
            input_memory_config, down_program_config, down_memory_config = self.dram_sharded_decode_configs["down_proj"]
            activated = ttnn.to_memory_config(activated, input_memory_config)
        result = ttnn.linear(
            activated,
            self.weights["down_proj_decode"] if down_program_config is not None else self.weights["down_proj"],
            dtype=ttnn.bfloat16,
            program_config=down_program_config,
            compute_kernel_config=(
                self.dense_compute_kernel_config
                if down_program_config is not None
                else self.prefill_dense_compute_kernel_config
            ),
            memory_config=down_memory_config,
        )
        return (
            ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
            if self.use_dram_sharded_dense_decode and normalized.shape[-2] == 1 and not self.use_sharded_dense_residual
            else result
        )

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
        value = ttnn.to_memory_config(value, self.decode_value_memory_config)
        ttnn.experimental.paged_fused_update_cache(
            key_cache,
            key,
            value_cache,
            value,
            update_idxs_tensor=current_positions,
            page_table=page_table,
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
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self.decode_concat_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(
            attended,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_sub_core_grids,
        )
        projected = ttnn.linear(
            (
                ttnn.to_memory_config(attended, self.dram_sharded_decode_configs["o"][0])
                if self.use_dram_sharded_dense_decode
                else attended
            ),
            self.weights["o_decode"] if self.use_dram_sharded_dense_decode else self.weights["o"],
            dtype=ttnn.bfloat16,
            program_config=(self.dram_sharded_decode_configs["o"][1] if self.use_dram_sharded_dense_decode else None),
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=(
                self.dram_sharded_decode_configs["o"][2]
                if self.use_dram_sharded_dense_decode
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        if self.use_dram_sharded_dense_decode and not self.use_sharded_dense_residual:
            projected = ttnn.to_memory_config(projected, ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        return ttnn.permute(projected, (0, 2, 1, 3))

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
        if self.advisor_moe_norm_memory_config is not None:
            self._validate_hidden(hidden_states, decode=True)
            if tuple(current_positions.shape) != (self.batch,):
                raise ValueError(
                    f"current_positions must have shape ({self.batch},), got {tuple(current_positions.shape)}"
                )
            if self.use_rope and (position_cos is None or position_sin is None):
                raise ValueError("this layer kind requires position_cos and position_sin")
            normalized = ttnn.rms_norm(
                ttnn.to_memory_config(hidden_states, self.advisor_moe_norm_memory_config),
                epsilon=self.eps,
                weight=self.weights["norm"],
                program_config=self.advisor_moe_norm_program_config,
                memory_config=self.advisor_moe_norm_memory_config,
            )
            attention = self._attention_decode(
                normalized,
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=page_table,
                current_positions=current_positions,
                position_cos=position_cos,
                position_sin=position_sin,
            )
            mlp = self._mlp(normalized, 1)
            return ttnn.add(ttnn.add(hidden_states, attention), mlp)
        if not self.use_sharded_dense_residual:
            return super().decode_forward(
                hidden_states,
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=page_table,
                current_positions=current_positions,
                position_cos=position_cos,
                position_sin=position_sin,
            )
        self._validate_hidden(hidden_states, decode=True)
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape ({self.batch},), got {tuple(current_positions.shape)}")
        if self.use_rope and (position_cos is None or position_sin is None):
            raise ValueError("this layer kind requires position_cos and position_sin")
        hidden_states = ttnn.to_memory_config(hidden_states, self.dense_residual_memory_config)
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["norm"],
            program_config=self.dense_norm_program_config,
            memory_config=self.dense_residual_memory_config,
        )
        attention = self._attention_decode(
            normalized,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        mlp = self._dense_mlp(normalized)
        return ttnn.add(
            ttnn.add(hidden_states, attention, memory_config=self.dense_residual_memory_config),
            mlp,
            memory_config=self.dense_residual_memory_config,
        )

    def _sparse_moe_chunk(self, normalized, token_count):
        if token_count != 1:
            return super()._sparse_moe_chunk(normalized, token_count)
        self.optimized_batch1_moe_calls += 1
        flat = ttnn.reshape(normalized, (1, self.hidden_size))
        logits = ttnn.linear(
            flat,
            self.weights["router"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if self.layer_idx == 4 else ttnn.L1_MEMORY_CONFIG,
        )
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        exact_mask = ttnn.scatter(
            ttnn.zeros_like(logits),
            dim=-1,
            index=top_indices,
            src=ttnn.ones_like(top_values),
        )
        sparsity = ttnn.to_layout(
            ttnn.reshape(exact_mask, (1, 1, 1, self.num_experts)),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        expert_input = ttnn.reshape(flat, (1, 1, 1, self.hidden_size))
        gate_up = ttnn.sparse_matmul(
            expert_input,
            self.weights["expert_gate_up"],
            sparsity=sparsity,
            nnz=self.top_k,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([self.policy.sparse_output_tile_height, 32]),
            program_config=self.optimized_sparse_gate_up_program_config,
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=self.policy.activation_dtype,
        )
        gate_up = ttnn.reshape(gate_up, (1, self.num_experts, 2 * self.intermediate_size))
        gate, up = self._split_gate_up(gate_up, self.intermediate_size)
        if self.policy.sparse_l1_chain:
            activated = ttnn.multiply(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            activated = self._fused_swiglu(gate, up)
        activated = ttnn.reshape(
            ttnn.transpose(
                activated,
                1,
                0,
                memory_config=ttnn.L1_MEMORY_CONFIG if self.policy.sparse_l1_chain else ttnn.DRAM_MEMORY_CONFIG,
            ),
            (1, self.num_experts, 1, self.intermediate_size),
            memory_config=ttnn.L1_MEMORY_CONFIG if self.policy.sparse_l1_chain else ttnn.DRAM_MEMORY_CONFIG,
        )
        expert_output = ttnn.sparse_matmul(
            activated,
            self.weights["expert_down"],
            sparsity=sparsity,
            nnz=self.top_k,
            is_input_a_sparse=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([self.policy.sparse_output_tile_height, 32]),
            program_config=self.optimized_sparse_down_program_config,
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=self.policy.activation_dtype,
        )
        expert_output = ttnn.reshape(
            ttnn.permute(expert_output, (0, 2, 1, 3)),
            (1, self.num_experts, self.hidden_size),
        )
        routing = ttnn.reshape(routing, (1, self.num_experts, 1))
        expert_output = ttnn.multiply(expert_output, routing, output_tensor=expert_output)
        return ttnn.sum(expert_output, dim=1)

    def _packed_all_expert_moe(self, flat, routing, token_count):
        if not self.policy.prefill_moe_grid_scale:
            return super()._packed_all_expert_moe(flat, routing, token_count)
        expert_input = ttnn.repeat(
            ttnn.reshape(flat, (1, token_count, self.hidden_size)),
            ttnn.Shape((self.num_experts, 1, 1)),
        )
        gate_up_grid_x = (2 * self.intermediate_size // ttnn.TILE_SIZE) // self.policy.prefill_moe_grid_scale
        down_grid_x = (self.hidden_size // ttnn.TILE_SIZE) // self.policy.prefill_moe_grid_scale
        gate_up = ttnn.matmul(
            expert_input,
            ttnn.reshape(
                self.weights["expert_gate_up"],
                (self.num_experts, self.hidden_size, 2 * self.intermediate_size),
            ),
            dtype=ttnn.bfloat16,
            program_config=_prefill_moe_program_config(
                m=token_count,
                n=2 * self.intermediate_size,
                grid_x=gate_up_grid_x,
                in0_block_w=self.policy.prefill_gate_up_in0_block_w,
            ),
            compute_kernel_config=self.prefill_expert_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate, up = self._split_gate_up(gate_up, self.intermediate_size)
        activated = self._fused_swiglu(gate, up)
        expert_output = ttnn.matmul(
            activated,
            ttnn.reshape(
                self.weights["expert_down"],
                (self.num_experts, self.intermediate_size, self.hidden_size),
            ),
            dtype=ttnn.bfloat16,
            program_config=_prefill_moe_program_config(
                m=token_count,
                n=self.hidden_size,
                grid_x=down_grid_x,
                in0_block_w=self.policy.prefill_down_in0_block_w,
            ),
            compute_kernel_config=self.prefill_expert_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if token_count != ttnn.TILE_SIZE:
            routing = ttnn.reshape(
                ttnn.permute(routing, (1, 0)),
                (self.num_experts, token_count, 1),
            )
            return ttnn.sum(ttnn.multiply(expert_output, routing), dim=0)

        expert_output = ttnn.reshape(
            expert_output,
            (self.num_experts, 1, token_count, self.hidden_size),
        )
        expert_output = ttnn.to_memory_config(expert_output, ttnn.L1_MEMORY_CONFIG)
        scores = ttnn.to_layout(
            ttnn.reshape(routing, (token_count, 1, 1, self.num_experts)),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        indices = ttnn.slice(
            self.fused_reduce_indices,
            (0, 0, 0, 0),
            (token_count, 1, 1, self.num_experts),
        )
        outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            expert_output,
            indices,
            self.fused_reduce_mapping,
            reduce_dim=0,
            split_size=self.hidden_size,
            cluster_axis=0,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            scores_tensor=scores,
        )
        return ttnn.reshape(outputs[0], (token_count, self.hidden_size))
