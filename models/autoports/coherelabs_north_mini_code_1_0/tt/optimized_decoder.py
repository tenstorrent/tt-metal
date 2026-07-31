# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device Cohere North-Mini decoder layer.

Runtime methods are independent of :class:`FunctionalDecoder`.  Setup reuses
only the functional stage's canonical state-dict lookup and Cohere RoPE row
permutation helpers.  Candidate policies are construction-time choices so a
captured decode graph cannot switch to a functional fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    DEFAULT_PAGE_SIZE,
    PREFILL_MOE_CHUNK,
    _load_expert_weights,
    _require_tensor,
    _rope_output_permutation,
)
from models.common.lightweightmodule import LightweightModule
from models.demos.gpt_oss.tt.experts.config import ProgramConfig


@dataclass(frozen=True)
class OptimizationPolicy:
    attention_weight_dtype: object
    expert_weight_dtype: object
    cache_dtype: object
    attention_math_fidelity: object
    expert_output_dtype: object = ttnn.bfloat8_b
    sparse_experts: bool = True
    sparse_prefill: bool = False
    static_sparse_nnz: int | None = None
    gate_up_in0_block_w: int = 16
    down_in0_block_w: int = 24
    gate_up_cores: tuple[int, int] = (4, 4)
    down_cores: tuple[int, int] = (8, 8)
    gate_up_subblock_w: int = 1
    down_subblock_w: int = 1
    dram_sharded_attention: bool = False
    large_prefill_configs: bool = False
    packed_decode_config: str | None = None
    dram_sharded_experts: str | None = None
    packed_sparse_gate_up: bool = False
    precise_router: bool = False


POLICIES = {
    "default": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_in0_block_w=8,
        gate_up_cores=(6, 4),
        packed_sparse_gate_up=True,
        precise_router=True,
    ),
    "bfp8_hifi2": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp8_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "bfp4_attention": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "bf16_cache": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat16,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "bf16_reference": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat16,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        expert_output_dtype=ttnn.bfloat16,
    ),
    "dense_bfp4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        sparse_experts=False,
    ),
    "geometry_12x30": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_in0_block_w=16,
        down_in0_block_w=12,
        gate_up_cores=(3, 4),
        down_cores=(4, 4),
    ),
    "geometry_24x24": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_in0_block_w=8,
        down_in0_block_w=12,
        gate_up_cores=(6, 4),
        down_cores=(4, 4),
    ),
    "down_block24": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_in0_block_w=8,
        gate_up_cores=(6, 4),
        down_in0_block_w=24,
        down_cores=(4, 4),
    ),
    "down_8x4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_in0_block_w=8,
        gate_up_cores=(6, 4),
        down_in0_block_w=24,
        down_cores=(8, 4),
        down_subblock_w=2,
    ),
    "down_8x8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_in0_block_w=8,
        gate_up_cores=(6, 4),
        down_in0_block_w=24,
        down_cores=(8, 8),
    ),
    "down_6x4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_in0_block_w=8,
        gate_up_cores=(6, 4),
        down_in0_block_w=8,
        down_cores=(6, 4),
        down_subblock_w=3,
    ),
    "packed_sparse_gate_up": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        gate_up_in0_block_w=8,
        gate_up_cores=(6, 4),
        packed_sparse_gate_up=True,
    ),
    "packed_interleaved_48_64": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        sparse_experts=False,
        dram_sharded_attention=True,
        packed_decode_config="48_64",
    ),
    "packed_interleaved_32": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        sparse_experts=False,
        dram_sharded_attention=True,
        packed_decode_config="32",
    ),
    "packed_dram_gate_up": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        sparse_experts=False,
        dram_sharded_attention=True,
        dram_sharded_experts="gate_up",
    ),
    "packed_dram_down": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        sparse_experts=False,
        dram_sharded_attention=True,
        dram_sharded_experts="down",
    ),
    "packed_dram_both": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        sparse_experts=False,
        dram_sharded_attention=True,
        dram_sharded_experts="both",
    ),
    "dram_sharded_attention": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded_attention=True,
    ),
    "dram_sharded_attention_bfp8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded_attention=True,
    ),
    "large_prefill": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        large_prefill_configs=True,
    ),
}


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


def _expert_program_config(policy: OptimizationPolicy) -> ProgramConfig:
    return ProgramConfig(
        decode_gate_up_cores=policy.gate_up_cores,
        decode_down_cores=policy.down_cores,
        prefill_gate_up_cores=policy.gate_up_cores,
        prefill_down_cores=policy.down_cores,
        decode_gate_up_in0_block_w=policy.gate_up_in0_block_w,
        decode_down_in0_block_w=policy.down_in0_block_w,
        decode_gate_up_subblock_w=policy.gate_up_subblock_w,
        decode_down_subblock_w=policy.down_subblock_w,
        prefill_gate_up_in0_block_w=policy.gate_up_in0_block_w,
        prefill_down_in0_block_w=policy.down_in0_block_w,
        sequence_chunk_size=PREFILL_MOE_CHUNK,
        base_down_split_size=PREFILL_MOE_CHUNK,
    )


def _largest_tile_divisor(tile_count: int, limit: int) -> int:
    return max(divisor for divisor in range(1, limit + 1) if tile_count % divisor == 0)


def _dram_weight_memory_config(mesh_device, k: int, n: int):
    dram_grid = mesh_device.dram_grid_size()
    dram_cores = dram_grid.x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))})
    shard_width = math.ceil(n / dram_cores / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _dram_batched_weight_memory_config(mesh_device, batch: int, k: int, n: int):
    dram_grid = mesh_device.dram_grid_size()
    dram_cores = dram_grid.x * dram_grid.y
    if batch % dram_cores:
        raise ValueError(f"expert batch {batch} must divide evenly over {dram_cores} DRAM banks")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (batch // dram_cores * k, n), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _l1_batched_memory_config(mesh_device, batch: int, m: int, width: int):
    workers = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    if batch % len(workers):
        raise ValueError(f"expert batch {batch} must divide evenly over {len(workers)} worker cores")
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(core.x, core.y), ttnn.CoreCoord(core.x, core.y)) for core in workers]
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (batch // len(workers) * m, width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _l1_width_memory_config(mesh_device, width: int, core_count: int):
    storage_grid = mesh_device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(
        core_count,
        ttnn.CoreCoord(storage_grid.x, storage_grid.y),
        row_wise=True,
    )
    return ttnn.create_sharded_memory_config_(
        shape=(ttnn.TILE_SIZE, width // core_count),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dram_decode_matmul_config(k: int, n: int, input_cores: int, output_cores: int):
    k_tiles = k // ttnn.TILE_SIZE
    n_tiles = n // ttnn.TILE_SIZE
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_largest_tile_divisor(k_tiles // input_cores, 8),
        per_core_M=1,
        per_core_N=n_tiles // output_cores,
        fused_activation=None,
    )


def _prefill_matmul_config(m: int, k: int, n: int):
    grid_x = 8
    grid_y = min(8, max(1, math.ceil(m / ttnn.TILE_SIZE)))
    per_core_m = math.ceil(m / (ttnn.TILE_SIZE * grid_y))
    per_core_n = math.ceil(n / (ttnn.TILE_SIZE * grid_x))
    out_subblock_w = max(width for width in range(1, 5) if per_core_n % width == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=_largest_tile_divisor(k // ttnn.TILE_SIZE, 4 if m > 1024 else 8),
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _packed_decode_matmul_config(kind: str, projection: str):
    if (kind, projection) == ("48_64", "gate_up"):
        grid, block_w, per_core_n, subblock_w = (8, 6), 16, 1, 1
    elif (kind, projection) == ("48_64", "down"):
        grid, block_w, per_core_n, subblock_w = (8, 8), 24, 1, 1
    elif (kind, projection) == ("32", "gate_up"):
        grid, block_w, per_core_n, subblock_w = (8, 4), 16, 2, 2
    elif (kind, projection) == ("32", "down"):
        grid, block_w, per_core_n, subblock_w = (8, 4), 24, 2, 2
    else:
        raise ValueError(f"unsupported packed decode config {kind!r}/{projection!r}")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=block_w,
        out_subblock_h=1,
        out_subblock_w=subblock_w,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _packed_dram_program_config(k: int, n: int):
    return ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=_largest_tile_divisor(k // ttnn.TILE_SIZE, 2),
        per_core_M=1,
        per_core_N=n // ttnn.TILE_SIZE,
        fused_activation=None,
    )


class OptimizedDecoder(LightweightModule):
    """North-Mini decoder with packed QKV and routed active-expert execution."""

    def __init__(
        self,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int,
        max_cache_len: int,
        page_size: int,
        weights: dict[str, ttnn.Tensor],
        policy: OptimizationPolicy,
        candidate: str,
        prefill_sparsity: ttnn.Tensor | None,
    ):
        super().__init__()
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.page_size = page_size
        self.weights = weights
        self.policy = policy
        self.candidate = candidate
        self.prefill_sparsity = prefill_sparsity

        self.hidden_size = int(hf_config.hidden_size)
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = int(hf_config.head_dim)
        self.scale = self.head_dim**-0.5
        self.eps = float(hf_config.rms_norm_eps)
        self.layer_type = hf_config.layer_types[layer_idx]
        self.mlp_type = hf_config.mlp_layer_types[layer_idx]
        self.sliding_window = int(hf_config.sliding_window) if self.layer_type == "sliding_attention" else None
        self.use_rope = self.sliding_window is not None or (
            self.mlp_type == "dense" and int(hf_config.prefix_dense_sliding_window_pattern) == 1
        )
        self.intermediate_size = (
            int(hf_config.prefix_dense_intermediate_size)
            if self.mlp_type == "dense"
            else int(hf_config.intermediate_size)
        )
        self.num_experts = int(hf_config.num_experts)
        self.top_k = int(hf_config.num_experts_per_tok)
        self.expert_program_config = _expert_program_config(policy)

        self.attention_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.attention_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

        storage_grid = mesh_device.compute_with_storage_grid_size()
        decode_cores = min(batch, storage_grid.x * storage_grid.y)
        rectangle_x = next(
            (
                x
                for x in range(min(decode_cores, storage_grid.x), 0, -1)
                if decode_cores % x == 0 and decode_cores // x <= storage_grid.y
            ),
            None,
        )
        if rectangle_x is not None:
            rectangle_y = decode_cores // rectangle_x
            decode_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(rectangle_x - 1, rectangle_y - 1))}
            )
            self.decode_sub_core_grids = None
        else:
            decode_grid = ttnn.num_cores_to_corerangeset(decode_cores, storage_grid, row_wise=True)
            self.decode_sub_core_grids = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(storage_grid.x - 1, storage_grid.y - 1))}
            )
        self.decode_rope_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=decode_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_concat_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE * math.ceil(self.num_heads / ttnn.TILE_SIZE), self.head_dim),
            core_grid=decode_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if policy.dram_sharded_attention:
            max_cores = storage_grid.x * storage_grid.y
            qkv_input_cores = _largest_tile_divisor(self.hidden_size // ttnn.TILE_SIZE, max_cores)
            qkv_output_width = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
            qkv_output_cores = _largest_tile_divisor(qkv_output_width // ttnn.TILE_SIZE, max_cores)
            attention_width = self.num_heads * self.head_dim
            o_input_cores = _largest_tile_divisor(attention_width // ttnn.TILE_SIZE, max_cores)
            o_output_cores = qkv_input_cores
            self.qkv_decode_input_memory_config = _l1_width_memory_config(
                mesh_device, self.hidden_size, qkv_input_cores
            )
            self.qkv_decode_output_memory_config = _l1_width_memory_config(
                mesh_device, qkv_output_width, qkv_output_cores
            )
            self.qkv_decode_program_config = _dram_decode_matmul_config(
                self.hidden_size, qkv_output_width, qkv_input_cores, qkv_output_cores
            )
            self.o_decode_input_memory_config = _l1_width_memory_config(mesh_device, attention_width, o_input_cores)
            self.o_decode_output_memory_config = _l1_width_memory_config(mesh_device, self.hidden_size, o_output_cores)
            self.o_decode_program_config = _dram_decode_matmul_config(
                attention_width, self.hidden_size, o_input_cores, o_output_cores
            )

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
            raise ValueError(f"unknown optimization candidate {candidate!r}; expected one of {sorted(POLICIES)}")
        policy = POLICIES[candidate]
        # Sparse active-expert execution wins decisively for one user.  At
        # batch 32 its simultaneous expert intermediates exceed L1 and the
        # measured packed dense BFP4 family is faster than both sparse/DRAM and
        # the functional baseline.  Resolve this topology before weight load
        # and trace capture; runtime never switches implementations.
        if candidate == "default" and batch > 1:
            policy = replace(
                policy,
                sparse_experts=False,
                dram_sharded_attention=True,
                packed_decode_config="48_64",
            )
        elif policy.dram_sharded_attention and batch > 1:
            policy = replace(policy, sparse_experts=False)
        if candidate == "default" and hf_config.mlp_layer_types[layer_idx] == "dense":
            policy = replace(POLICIES["bfp8_hifi2"], large_prefill_configs=True)
        if not isinstance(mesh_device, ttnn.MeshDevice) or tuple(mesh_device.shape) != (1, 1):
            raise ValueError("OptimizedDecoder requires a single-device 1x1 MeshDevice")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")
        if batch < 1 or batch > 32:
            raise ValueError(f"optimized decode batch must be in [1, 32], got {batch}")
        if not 1 <= max_cache_len <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_cache_len must be in [1, {hf_config.max_position_embeddings}], got {max_cache_len}")
        if page_size < ttnn.TILE_SIZE or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive multiple of {ttnn.TILE_SIZE}, got {page_size}")

        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        num_kv_heads = int(hf_config.num_key_value_heads)
        head_dim = int(hf_config.head_dim)
        if (hidden_size, num_heads, num_kv_heads, head_dim) != (2048, 32, 4, 128):
            raise ValueError("North-Mini optimized dimensions must be hidden=2048, heads=32, kv_heads=4, head_dim=128")

        q = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        q = q.index_select(0, _rope_output_permutation(num_heads, head_dim))
        k = k.index_select(0, _rope_output_permutation(num_kv_heads, head_dim))
        qkv = torch.cat((q, k, v), dim=0).transpose(-2, -1).to(torch.bfloat16)
        qkv_memory_config = (
            _dram_weight_memory_config(mesh_device, hidden_size, qkv.shape[-1])
            if policy.dram_sharded_attention
            else ttnn.DRAM_MEMORY_CONFIG
        )
        attention_width = num_heads * head_dim
        o_memory_config = (
            _dram_weight_memory_config(mesh_device, attention_width, hidden_size)
            if policy.dram_sharded_attention
            else ttnn.DRAM_MEMORY_CONFIG
        )
        qkv_device_source = qkv.reshape(1, 1, hidden_size, qkv.shape[-1]) if policy.dram_sharded_attention else qkv
        o_device_source = (
            _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").transpose(-2, -1).to(torch.bfloat16)
        )
        if policy.dram_sharded_attention:
            o_device_source = o_device_source.reshape(1, 1, attention_width, hidden_size)
        weights = {
            "qkv": _as_device_tensor(
                qkv_device_source,
                mesh_device=mesh_device,
                dtype=policy.attention_weight_dtype,
                memory_config=qkv_memory_config,
            ),
            "o": _as_device_tensor(
                o_device_source,
                mesh_device=mesh_device,
                dtype=policy.attention_weight_dtype,
                memory_config=o_memory_config,
            ),
            "norm": _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
        }
        if policy.dram_sharded_attention:
            weights["qkv_prefill"] = _as_device_tensor(
                qkv,
                mesh_device=mesh_device,
                dtype=policy.attention_weight_dtype,
            )
            weights["o_prefill"] = _as_device_tensor(
                o_device_source.reshape(attention_width, hidden_size),
                mesh_device=mesh_device,
                dtype=policy.attention_weight_dtype,
            )

        mlp_type = hf_config.mlp_layer_types[layer_idx]
        prefill_sparsity = None
        if mlp_type == "dense":
            gate = _require_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").transpose(-2, -1)
            up = _require_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(-2, -1)
            weights["gate_up"] = _as_device_tensor(
                torch.cat((gate, up), dim=-1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=policy.expert_weight_dtype,
            )
            weights["down"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=policy.expert_weight_dtype,
            )
        elif mlp_type == "sparse":
            gate, up, down = _load_expert_weights(
                state_dict,
                layer_idx,
                int(hf_config.num_experts),
                int(hf_config.intermediate_size),
            )
            weights["router"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.gate.weight")
                .transpose(-2, -1)
                .to(torch.float32 if policy.precise_router else torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.float32 if policy.precise_router else ttnn.bfloat16,
            )
            if policy.sparse_experts:
                if policy.packed_sparse_gate_up:
                    weights["expert_gate_up_sparse"] = _as_device_tensor(
                        torch.cat((gate, up), dim=-1).unsqueeze(0),
                        mesh_device=mesh_device,
                        dtype=policy.expert_weight_dtype,
                    )
                else:
                    weights["expert_gate"] = _as_device_tensor(
                        gate.unsqueeze(0), mesh_device=mesh_device, dtype=policy.expert_weight_dtype
                    )
                    weights["expert_up"] = _as_device_tensor(
                        up.unsqueeze(0), mesh_device=mesh_device, dtype=policy.expert_weight_dtype
                    )
                weights["expert_down_sparse"] = _as_device_tensor(
                    down.unsqueeze(0), mesh_device=mesh_device, dtype=policy.expert_weight_dtype
                )
                prefill_sparsity = _as_device_tensor(
                    torch.ones(1, 1, 1, int(hf_config.num_experts), dtype=torch.bfloat16),
                    mesh_device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            if not policy.sparse_experts or not policy.sparse_prefill:
                # Packed all-expert prefill is materially faster than sparse
                # prefill, but BFP4 falls below the MoE PCC floor for
                # sequence-shaped activations.  Keep the independently
                # validated routed decode weights at BFP4 while loading the
                # phase-specific packed prefill family at BFP8.
                packed_expert_dtype = (
                    ttnn.bfloat8_b if candidate == "default" and policy.sparse_experts else policy.expert_weight_dtype
                )
                weights["expert_gate_up"] = _as_device_tensor(
                    torch.cat((gate, up), dim=-1),
                    mesh_device=mesh_device,
                    dtype=packed_expert_dtype,
                    memory_config=(
                        _dram_batched_weight_memory_config(
                            mesh_device,
                            int(hf_config.num_experts),
                            hidden_size,
                            2 * int(hf_config.intermediate_size),
                        )
                        if policy.dram_sharded_experts in ("gate_up", "both")
                        else ttnn.DRAM_MEMORY_CONFIG
                    ),
                )
                weights["expert_down_packed"] = _as_device_tensor(
                    down,
                    mesh_device=mesh_device,
                    dtype=packed_expert_dtype,
                    memory_config=(
                        _dram_batched_weight_memory_config(
                            mesh_device,
                            int(hf_config.num_experts),
                            int(hf_config.intermediate_size),
                            hidden_size,
                        )
                        if policy.dram_sharded_experts in ("down", "both")
                        else ttnn.DRAM_MEMORY_CONFIG
                    ),
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
            prefill_sparsity=prefill_sparsity,
        )

    @staticmethod
    def build_rope_rows(position_ids, *, hf_config, decode: bool = False):
        import torch

        positions = torch.as_tensor(position_ids, dtype=torch.float32)
        head_dim = int(hf_config.head_dim)
        theta = float(hf_config.rope_parameters["rope_theta"])
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = positions.unsqueeze(-1) * inv_freq
        embedding = torch.cat((freqs, freqs), dim=-1)
        embedding = embedding.reshape(1, -1, 1, head_dim) if decode else embedding.reshape(1, 1, -1, head_dim)
        return embedding.cos().to(torch.bfloat16), embedding.sin().to(torch.bfloat16)

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

    def _validate_hidden(self, hidden_states, *, decode: bool):
        actual = tuple(hidden_states.shape)
        expected_seq = 1 if decode else None
        if len(actual) != 4 or actual[0] != 1 or actual[1] != self.batch or actual[3] != self.hidden_size:
            raise ValueError(
                f"hidden_states must match (1, {self.batch}, {expected_seq}, {self.hidden_size}), got {actual}"
            )
        if decode and actual[2] != 1:
            raise ValueError(f"decode hidden_states sequence must be 1, got {actual[2]}")
        if not decode and not 1 <= actual[2] <= self.max_cache_len:
            raise ValueError(f"prefill sequence must be in [1, {self.max_cache_len}], got {actual[2]}")
        return actual[2]

    def _qkv_prefill(self, normalized, seq_len, position_cos, position_sin):
        prefill_m = self.batch * math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        fused = ttnn.linear(
            normalized,
            self.weights.get("qkv_prefill", self.weights["qkv"]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.attention_compute_kernel_config,
            program_config=(
                _prefill_matmul_config(
                    prefill_m,
                    self.hidden_size,
                    (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                )
                if self.policy.large_prefill_configs
                else None
            ),
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
        cache_key = ttnn.typecast(key, self.policy.cache_dtype)
        cache_value = ttnn.typecast(value, self.policy.cache_dtype)
        for user in range(self.batch):
            key_user = ttnn.slice(cache_key, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            value_user = ttnn.slice(cache_value, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            ttnn.experimental.paged_fill_cache(key_cache, key_user, page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(value_cache, value_user, page_table, batch_idx=user)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                exp_approx_mode=False,
                q_chunk_size=32 if seq_len < 2048 else 256,
                k_chunk_size=32 if seq_len < 2048 else 256,
            ),
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            attended,
            self.weights.get("o_prefill", self.weights["o"]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.attention_compute_kernel_config,
            program_config=(
                _prefill_matmul_config(
                    self.batch * math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE,
                    self.num_heads * self.head_dim,
                    self.hidden_size,
                )
                if self.policy.large_prefill_configs
                else None
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _qkv_decode(self, normalized, position_cos, position_sin):
        if self.policy.dram_sharded_attention:
            normalized = ttnn.reshape(normalized, (1, 1, self.batch, self.hidden_size))
            normalized = ttnn.to_memory_config(normalized, self.qkv_decode_input_memory_config)
        fused = ttnn.linear(
            normalized,
            self.weights["qkv"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.attention_compute_kernel_config,
            program_config=self.qkv_decode_program_config if self.policy.dram_sharded_attention else None,
            memory_config=(
                self.qkv_decode_output_memory_config if self.policy.dram_sharded_attention else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        if self.policy.dram_sharded_attention:
            fused = ttnn.sharded_to_interleaved(fused, ttnn.DRAM_MEMORY_CONFIG)
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
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self.decode_concat_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(
            attended, num_heads=self.num_heads, sub_core_grids=self.decode_sub_core_grids
        )
        if self.policy.dram_sharded_attention:
            attended = ttnn.to_memory_config(attended, self.o_decode_input_memory_config)
        projected = ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.attention_compute_kernel_config,
            program_config=self.o_decode_program_config if self.policy.dram_sharded_attention else None,
            memory_config=(
                self.o_decode_output_memory_config if self.policy.dram_sharded_attention else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        if self.policy.dram_sharded_attention:
            projected = ttnn.sharded_to_interleaved(projected, ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        return ttnn.permute(projected, (0, 2, 1, 3))

    def _dense_mlp(self, normalized):
        seq_len = normalized.shape[2]
        prefill_m = self.batch * math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        use_large_prefill = self.policy.large_prefill_configs and seq_len > 1
        gate_up = ttnn.linear(
            normalized,
            self.weights["gate_up"],
            dtype=ttnn.bfloat16,
            program_config=(
                _prefill_matmul_config(prefill_m, self.hidden_size, 2 * self.intermediate_size)
                if use_large_prefill
                else None
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.slice(
            gate_up,
            (0, 0, 0, 0),
            (1, self.batch, gate_up.shape[2], self.intermediate_size),
        )
        up = ttnn.slice(
            gate_up,
            (0, 0, 0, self.intermediate_size),
            (1, self.batch, gate_up.shape[2], 2 * self.intermediate_size),
        )
        activated = ttnn.multiply(ttnn.silu(gate), up)
        return ttnn.linear(
            activated,
            self.weights["down"],
            dtype=ttnn.bfloat16,
            program_config=(
                _prefill_matmul_config(prefill_m, self.intermediate_size, self.hidden_size)
                if use_large_prefill
                else None
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _routing(self, normalized, token_count):
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))
        logits = ttnn.linear(
            flat,
            self.weights["router"],
            dtype=ttnn.float32 if self.policy.precise_router else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.policy.precise_router:
            logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        return flat, routing

    def _sparse_decode_moe(self, normalized, routing):
        sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing), ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile((32, 32))
        # At logical batch 32 each expert projection output is ~107 MB.  Two
        # simultaneous L1 outputs exceed per-bank capacity; batch 1 remains L1
        # resident while the serving-batch path uses DRAM intermediates.
        expert_memory_config = ttnn.L1_MEMORY_CONFIG if self.batch == 1 else ttnn.DRAM_MEMORY_CONFIG
        if self.policy.packed_sparse_gate_up:
            gate_up = ttnn.sparse_matmul(
                normalized,
                self.weights["expert_gate_up_sparse"],
                sparsity=sparsity,
                nnz=self.policy.static_sparse_nnz,
                memory_config=expert_memory_config,
                output_tile=output_tile,
                program_config=self.expert_program_config.get_decode_gate_up_config(
                    normalized.shape[2], 2 * self.intermediate_size, k=self.hidden_size
                ),
                dtype=self.policy.expert_output_dtype,
            )
            gate_up = ttnn.reshape(gate_up, (self.batch, self.num_experts, 2 * self.intermediate_size))
            gate = ttnn.slice(
                gate_up,
                (0, 0, 0),
                (self.batch, self.num_experts, self.intermediate_size),
            )
            up = ttnn.slice(
                gate_up,
                (0, 0, self.intermediate_size),
                (self.batch, self.num_experts, 2 * self.intermediate_size),
            )
        else:
            gate = ttnn.sparse_matmul(
                normalized,
                self.weights["expert_gate"],
                sparsity=sparsity,
                nnz=self.policy.static_sparse_nnz,
                memory_config=expert_memory_config,
                output_tile=output_tile,
                program_config=self.expert_program_config.get_decode_gate_up_config(
                    normalized.shape[2], self.intermediate_size, k=self.hidden_size
                ),
                dtype=self.policy.expert_output_dtype,
            )
            up = ttnn.sparse_matmul(
                normalized,
                self.weights["expert_up"],
                sparsity=sparsity,
                nnz=self.policy.static_sparse_nnz,
                memory_config=expert_memory_config,
                output_tile=output_tile,
                program_config=self.expert_program_config.get_decode_gate_up_config(
                    normalized.shape[2], self.intermediate_size, k=self.hidden_size
                ),
                dtype=self.policy.expert_output_dtype,
            )
            gate = ttnn.reshape(gate, (self.batch, self.num_experts, self.intermediate_size))
            up = ttnn.reshape(up, (self.batch, self.num_experts, self.intermediate_size))
        down_input = ttnn.multiply(ttnn.silu(gate), up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, self.num_experts, self.batch, self.intermediate_size))
        down_sparsity = sparsity if self.batch == 1 else self.prefill_sparsity
        down_nnz = self.policy.static_sparse_nnz if self.batch == 1 else self.num_experts
        down = ttnn.sparse_matmul(
            down_input,
            self.weights["expert_down_sparse"],
            sparsity=down_sparsity,
            nnz=down_nnz,
            memory_config=expert_memory_config,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=self.expert_program_config.get_decode_down_config(
                self.batch, self.hidden_size, k=self.intermediate_size
            ),
            dtype=self.policy.expert_output_dtype,
        )
        if len(down.shape) == 3:
            down = ttnn.reshape(down, (1, self.num_experts, self.batch, self.hidden_size))
        output = ttnn.permute(down, (0, 2, 1, 3))
        output = ttnn.reshape(output, (self.batch, self.num_experts, self.hidden_size))
        routing = ttnn.permute(routing, (1, 0))
        routing = ttnn.reshape(routing, (self.batch, self.num_experts, 1))
        output = ttnn.multiply(output, routing, output_tensor=output)
        output = ttnn.sum(output, dim=1)
        return ttnn.reshape(output, (1, self.batch, 1, self.hidden_size))

    def _sparse_prefill_moe(self, normalized, routing, seq_len):
        padded_seq_len = math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if padded_seq_len != seq_len:
            normalized = ttnn.pad(
                normalized,
                padding=((0, 0), (0, 0), (0, padded_seq_len - seq_len), (0, 0)),
                value=0.0,
            )
            _, routing = self._routing(normalized, self.batch * padded_seq_len)
        group_count = self.batch * padded_seq_len // ttnn.TILE_SIZE
        hidden_groups = ttnn.reshape(normalized, (1, group_count, ttnn.TILE_SIZE, self.hidden_size))
        sparsity = ttnn.repeat(self.prefill_sparsity, (1, 1, group_count, 1))
        output_tile = ttnn.Tile((32, 32))
        gate = ttnn.sparse_matmul(
            hidden_groups,
            self.weights["expert_gate"],
            sparsity=sparsity,
            nnz=self.num_experts * group_count,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self.expert_program_config.get_prefill_gate_up_config(
                ttnn.TILE_SIZE, self.intermediate_size, k=self.hidden_size
            ),
            dtype=self.policy.expert_output_dtype,
        )
        up = ttnn.sparse_matmul(
            hidden_groups,
            self.weights["expert_up"],
            sparsity=sparsity,
            nnz=self.num_experts * group_count,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self.expert_program_config.get_prefill_gate_up_config(
                ttnn.TILE_SIZE, self.intermediate_size, k=self.hidden_size
            ),
            dtype=self.policy.expert_output_dtype,
        )
        gate = ttnn.transpose(gate, 1, 3)
        up = ttnn.transpose(up, 1, 3)
        gate = ttnn.reshape(gate, (self.batch, self.num_experts, padded_seq_len, self.intermediate_size))
        up = ttnn.reshape(up, (self.batch, self.num_experts, padded_seq_len, self.intermediate_size))
        down_input = ttnn.multiply(ttnn.silu(gate), up)
        down_input = ttnn.reshape(
            down_input, (1, self.num_experts, self.batch * padded_seq_len, self.intermediate_size)
        )
        down = ttnn.sparse_matmul(
            down_input,
            self.weights["expert_down_sparse"],
            sparsity=self.prefill_sparsity,
            nnz=self.num_experts,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=self.expert_program_config.get_prefill_down_config(
                self.batch * padded_seq_len, self.hidden_size, k=self.intermediate_size
            ),
            dtype=self.policy.expert_output_dtype,
        )
        output = ttnn.reshape(down, (self.batch, self.num_experts, padded_seq_len, self.hidden_size))
        routing = ttnn.permute(routing, (1, 0))
        routing = ttnn.reshape(routing, (self.batch, self.num_experts, padded_seq_len, 1))
        output = ttnn.multiply(output, routing, output_tensor=output)
        output = ttnn.experimental.fast_reduce_nc(output, dims=[1])
        output = ttnn.reshape(output, (1, self.batch, padded_seq_len, self.hidden_size))
        if padded_seq_len != seq_len:
            output = ttnn.slice(output, (0, 0, 0, 0), (1, self.batch, seq_len, self.hidden_size))
        return output

    def _dense_expert_moe(self, normalized, routing, seq_len):
        token_count = self.batch * seq_len
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))
        expert_input = ttnn.reshape(flat, (1, token_count, self.hidden_size))
        expert_input = ttnn.repeat(expert_input, ttnn.Shape((self.num_experts, 1, 1)))
        dram_gate_up = seq_len == 1 and self.policy.dram_sharded_experts in ("gate_up", "both")
        if dram_gate_up:
            expert_input = ttnn.to_memory_config(
                expert_input,
                _l1_batched_memory_config(self.mesh_device, self.num_experts, self.batch, self.hidden_size),
                dtype=ttnn.bfloat4_b,
            )
        gate_up = ttnn.matmul(
            expert_input,
            self.weights["expert_gate_up"],
            dtype=ttnn.bfloat8_b if dram_gate_up else ttnn.bfloat16,
            program_config=(
                _packed_dram_program_config(self.hidden_size, 2 * self.intermediate_size)
                if dram_gate_up
                else _packed_decode_matmul_config(self.policy.packed_decode_config, "gate_up")
                if seq_len == 1 and self.policy.packed_decode_config
                else None
            ),
            memory_config=(
                _l1_batched_memory_config(self.mesh_device, self.num_experts, self.batch, 2 * self.intermediate_size)
                if dram_gate_up
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        if dram_gate_up:
            gate_up = ttnn.sharded_to_interleaved(gate_up, ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.slice(
            gate_up,
            (0, 0, 0),
            (self.num_experts, token_count, self.intermediate_size),
        )
        up = ttnn.slice(
            gate_up,
            (0, 0, self.intermediate_size),
            (self.num_experts, token_count, 2 * self.intermediate_size),
        )
        activated = ttnn.multiply(ttnn.silu(gate), up)
        dram_down = seq_len == 1 and self.policy.dram_sharded_experts in ("down", "both")
        if dram_down:
            activated = ttnn.to_memory_config(
                activated,
                _l1_batched_memory_config(self.mesh_device, self.num_experts, self.batch, self.intermediate_size),
                dtype=ttnn.bfloat4_b,
            )
        output = ttnn.matmul(
            activated,
            self.weights["expert_down_packed"],
            dtype=ttnn.bfloat8_b if dram_down else ttnn.bfloat16,
            program_config=(
                _packed_dram_program_config(self.intermediate_size, self.hidden_size)
                if dram_down
                else _packed_decode_matmul_config(self.policy.packed_decode_config, "down")
                if seq_len == 1 and self.policy.packed_decode_config
                else None
            ),
            memory_config=(
                _l1_batched_memory_config(self.mesh_device, self.num_experts, self.batch, self.hidden_size)
                if dram_down
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        if dram_down:
            output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.permute(routing, (1, 0))
        routing = ttnn.reshape(routing, (self.num_experts, token_count, 1))
        output = ttnn.multiply(output, routing)
        output = ttnn.sum(output, dim=0)
        return ttnn.reshape(output, (1, self.batch, seq_len, self.hidden_size))

    def _mlp(self, normalized, seq_len):
        if self.mlp_type == "dense":
            return self._dense_mlp(normalized)
        _, routing = self._routing(normalized, self.batch * seq_len)
        if not self.policy.sparse_experts or (seq_len > 1 and not self.policy.sparse_prefill):
            return self._dense_expert_moe(normalized, routing, seq_len)
        if seq_len == 1:
            return self._sparse_decode_moe(normalized, routing)
        return self._sparse_prefill_moe(normalized, routing, seq_len)

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
        mlp = self._mlp(normalized, seq_len)
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
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape ({self.batch},), got {tuple(current_positions.shape)}")
        if self.use_rope and (position_cos is None or position_sin is None):
            raise ValueError("this layer kind requires position_cos and position_sin")
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
        mlp = self._mlp(normalized, 1)
        return ttnn.add(ttnn.add(hidden_states, attention), mlp)

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
