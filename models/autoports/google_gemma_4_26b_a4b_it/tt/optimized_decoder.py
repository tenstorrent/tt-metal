# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device decoder for ``google/gemma-4-26B-A4B-it``.

The fused decoder is the semantic and topology baseline.  This module owns the
measured decode MoE body and exposes construction-time candidate policies used
by the optimized-decoder evidence harness.  No policy decision is made from a
runtime tensor value, so every path remains trace safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    TILE_SIZE,
    TOP_K_EXPERTS,
    _build_sparse_matmul_config,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder


@dataclass(frozen=True)
class OptimizationPolicy:
    """One reproducible precision/geometry point in the batch-1 search."""

    attention_weight_dtype: Any
    dense_weight_dtype: Any
    expert_weight_dtype: Any
    attention_math_fidelity: Any
    dense_math_fidelity: Any
    expert_math_fidelity: Any
    expert_up_gate_in0_block_w: int
    expert_down_in0_block_w: int
    expert_up_gate_cores: int | None = None
    expert_down_cores: int | None = None
    dense_gate_up_weight_dtype: Any | None = None
    dense_down_weight_dtype: Any | None = None
    packed_dense_gate_up: bool = False

    @property
    def effective_dense_gate_up_dtype(self) -> Any:
        return self.dense_gate_up_weight_dtype or self.dense_weight_dtype

    @property
    def effective_dense_down_dtype(self) -> Any:
        return self.dense_down_weight_dtype or self.dense_weight_dtype


SPARSE_SINGLE_CORE_MCAST_BLOCKER = "kernel_single_core_mcast_blocked"


def sparse_geometry_host_rejection(policy: OptimizationPolicy) -> str | None:
    """Return the checkout-specific reason an explicit sparse geometry is unsafe."""

    if policy.expert_up_gate_cores == 1:
        return (
            f"{SPARSE_SINGLE_CORE_MCAST_BLOCKER}: explicit packed up/gate num_cores=1 is unsafe because "
            "this checkout's sparse 1D-mcast factory lacks the single-core in0 SKIP_MCAST guard"
        )
    return None


def _validate_optimization_policy(candidate: str, policy: OptimizationPolicy) -> None:
    if rejection := sparse_geometry_host_rejection(policy):
        raise ValueError(f"optimization candidate {candidate!r} is host-rejected: {rejection}")


POLICIES = {
    # Exact fused-stage precision with explicit configs.  This is the local
    # control used by candidate tests, not the final default.
    "bf16_hifi4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat16,
        attention_math_fidelity=ttnn.MathFidelity.HiFi4,
        dense_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_up_gate_in0_block_w=1,
        expert_down_in0_block_w=1,
    ),
    "bfp8_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        dense_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
    "bfp4_experts": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        dense_math_fidelity=ttnn.MathFidelity.HiFi2,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
    "bfp4_experts_isolated": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat4_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi4,
        dense_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
    "bfp8_experts_hifi4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi4,
        dense_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
    "bfp8_experts_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat8_b,
        # The inherited fused attention linears use TTNN's BF16 decode
        # default, which the profiler verifies as HiFi2.  Keep the policy
        # metadata aligned with the runtime rather than claiming an unused
        # construction-time HiFi4 config.
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        dense_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
    "bfp4_attention_only": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        dense_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
    "bfp8_attention_only": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        dense_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
    "bfp4_dense_gate_up": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi4,
        dense_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
        dense_gate_up_weight_dtype=ttnn.bfloat4_b,
    ),
    "bfp4_dense_all": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi4,
        dense_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
        dense_gate_up_weight_dtype=ttnn.bfloat4_b,
        dense_down_weight_dtype=ttnn.bfloat4_b,
    ),
    "bfp4_dense_gate_up_packed": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat16,
        dense_weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi4,
        dense_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
        dense_gate_up_weight_dtype=ttnn.bfloat4_b,
        packed_dense_gate_up=True,
    ),
    "bfp8_projection_hifi2": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        dense_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        dense_math_fidelity=ttnn.MathFidelity.HiFi2,
        expert_math_fidelity=ttnn.MathFidelity.HiFi4,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
    "bfp4_all": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        dense_weight_dtype=ttnn.bfloat4_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        dense_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_up_gate_in0_block_w=8,
        expert_down_in0_block_w=11,
    ),
}

# Precision-locked sparse geometry cross-product.  The names intentionally
# encode (up/gate cores, down cores, up/gate K block, down K block).
for _prefix, _base_name in (
    ("bfp4_geo", "bfp4_experts_isolated"),
    ("bfp8_geo", "bfp8_experts_lofi"),
):
    for _uc, _dc, _ub, _db in (
        (1, 2, 1, 1),
        (2, 4, 4, 2),
        (4, 8, 8, 11),
        (2, 4, 11, 11),
        (1, 2, 22, 22),
        (4, 8, 22, 22),
    ):
        POLICIES[f"{_prefix}_u{_uc}_d{_dc}_k{_ub}_{_db}"] = replace(
            POLICIES[_base_name],
            expert_up_gate_in0_block_w=_ub,
            expert_down_in0_block_w=_db,
            expert_up_gate_cores=_uc,
            expert_down_cores=_dc,
        )


def _compute_config(device: Any, fidelity: Any, *, fp32_dest_acc_en: bool) -> Any:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=not fp32_dest_acc_en,
    )


def _sparse_program_config(
    m: int,
    n: int,
    *,
    in0_block_w: int,
    num_cores: int | None,
    projection: str,
) -> Any:
    if projection == "expert_up_gate" and num_cores == 1:
        raise ValueError(
            f"{SPARSE_SINGLE_CORE_MCAST_BLOCKER}: explicit packed up/gate num_cores=1 cannot be dispatched"
        )
    if num_cores is None:
        return _build_sparse_matmul_config(m, n, in0_block_w=in0_block_w)
    n_tiles = (n + TILE_SIZE - 1) // TILE_SIZE
    if n_tiles % num_cores:
        raise ValueError(f"N tiles {n_tiles} must divide sparse core count {num_cores}")
    grid = None
    for grid_y in range(1, 9):
        if num_cores % grid_y == 0 and num_cores // grid_y <= 8:
            grid = ttnn.CoreCoord(num_cores // grid_y, grid_y)
            break
    if grid is None:
        raise ValueError(f"cannot place {num_cores} sparse cores in an 8x8 grid")
    per_core_n = n_tiles // num_cores
    out_subblock_w = min(4, per_core_n)
    while per_core_n % out_subblock_w:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=max(TILE_SIZE, m) // TILE_SIZE,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _large_prefill_dense_program_config(m: int, *, projection: str) -> Any:
    """Shape-derived 2D configs for the material 1,024-token dense MLP."""

    m_tiles = math.ceil(m / TILE_SIZE)
    if projection == "gate_up":
        grid_x, grid_y, k_tiles, n_tiles, in0_block_w = 6, 8, 88, 66, 11
    elif projection == "down":
        grid_x, grid_y, k_tiles, n_tiles, in0_block_w = 8, 8, 66, 88, 6
    else:
        raise ValueError(f"unknown dense projection {projection!r}")
    assert k_tiles % in0_block_w == 0 and n_tiles % grid_x == 0
    per_core_n = n_tiles // grid_x
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=math.ceil(m_tiles / grid_y),
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def _decode_residual_configs(cores: int) -> tuple[Any, Any]:
    if HIDDEN_SIZE % (cores * TILE_SIZE):
        raise ValueError(f"hidden size {HIDDEN_SIZE} must tile-divide residual cores {cores}")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})
    shard_width = HIDDEN_SIZE // cores
    memory = ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, shard_width),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    block_w = shard_width // TILE_SIZE
    subblock_w = min(4, block_w)
    while block_w % subblock_w:
        subblock_w -= 1
    program = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[cores, 1],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    return memory, program


class OptimizedDecoder(FusedDecoder):
    """Batch-1 optimized decoder with an active-expert sparse runtime."""

    DEFAULT_CANDIDATE = "bfp8_experts_lofi"
    USE_LARGE_PREFILL_DENSE_CONFIGS = False
    PREFILL_EXPERT_UP_GATE_CORES = 4
    PREFILL_EXPERT_DOWN_CORES = 8
    PREFILL_EXPERT_UP_GATE_IN0_BLOCK_W = 8
    PREFILL_EXPERT_DOWN_IN0_BLOCK_W = 11

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.candidate = "bf16_hifi4"
        self.policy = POLICIES[self.candidate]
        self.attention_compute_config = _compute_config(
            self.mesh_device, self.policy.attention_math_fidelity, fp32_dest_acc_en=True
        )
        self.dense_compute_config = _compute_config(
            self.mesh_device, self.policy.dense_math_fidelity, fp32_dest_acc_en=True
        )
        self.expert_compute_config = _compute_config(
            self.mesh_device, self.policy.expert_math_fidelity, fp32_dest_acc_en=True
        )
        self.decode_residual_memory_config, self.decode_norm_program_config = _decode_residual_configs(8)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any], *, candidate: str | None = None, **kwargs: Any):
        candidate = candidate or cls.DEFAULT_CANDIDATE
        if candidate not in POLICIES:
            raise ValueError(f"unknown optimization candidate {candidate!r}; expected one of {sorted(POLICIES)}")
        policy = POLICIES[candidate]
        _validate_optimization_policy(candidate, policy)
        # FunctionalDecoder.from_state_dict is a classmethod and therefore
        # constructs ``cls``.  Expert tensors are materialized directly in the
        # requested dtype before FusedDecoder packs them.
        decoder = super().from_state_dict(
            state_dict,
            expert_weight_dtype=policy.expert_weight_dtype,
            **kwargs,
        )
        decoder._apply_policy(candidate, policy)
        return decoder

    def _apply_policy(self, candidate: str, policy: OptimizationPolicy) -> None:
        """Convert non-expert projection groups once, outside the hot path."""

        _validate_optimization_policy(candidate, policy)
        converted = {}
        for name, dtype in {
            "qkv": policy.attention_weight_dtype,
            "o_proj": policy.attention_weight_dtype,
            "mlp_gate": policy.effective_dense_gate_up_dtype,
            "mlp_up": policy.effective_dense_gate_up_dtype,
            "mlp_down": policy.effective_dense_down_dtype,
        }.items():
            old = getattr(self.weights, name)
            if old.dtype == dtype:
                converted[name] = old
            else:
                converted[name] = ttnn.typecast(old, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                old.deallocate(True)
        self.weights = replace(self.weights, **converted)
        if policy.packed_dense_gate_up:
            self.mlp_gate_up = ttnn.concat(
                [self.weights.mlp_gate, self.weights.mlp_up],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.mlp_gate_up = None
        self.candidate = candidate
        self.policy = policy
        self.attention_compute_config = _compute_config(
            self.mesh_device,
            policy.attention_math_fidelity,
            fp32_dest_acc_en=policy.attention_weight_dtype == ttnn.bfloat16,
        )
        self.dense_compute_config = _compute_config(
            self.mesh_device,
            policy.dense_math_fidelity,
            fp32_dest_acc_en=(
                policy.effective_dense_gate_up_dtype == ttnn.bfloat16
                and policy.effective_dense_down_dtype == ttnn.bfloat16
            ),
        )
        self.expert_compute_config = _compute_config(
            self.mesh_device,
            policy.expert_math_fidelity,
            fp32_dest_acc_en=policy.expert_weight_dtype == ttnn.bfloat16,
        )

    def _dense_mlp(self, x: ttnn.Tensor, *, fold_activation: bool) -> ttnn.Tensor:
        """Explicitly configure every dense projection under the chosen policy."""

        del fold_activation
        dense_kwargs = {"compute_kernel_config": self.dense_compute_config}
        if self.USE_LARGE_PREFILL_DENSE_CONFIGS and x.padded_shape[-2] == 1024:
            dense_kwargs["program_config"] = _large_prefill_dense_program_config(
                x.padded_shape[-2], projection="gate_up"
            )
        if self.mlp_gate_up is None:
            gate = ttnn.linear(
                x,
                self.weights.mlp_gate,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **dense_kwargs,
            )
            up = ttnn.linear(
                x,
                self.weights.mlp_up,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **dense_kwargs,
            )
        else:
            physical_m = x.padded_shape[-2]
            gate, up = ttnn.experimental.minimal_matmul_split(
                x,
                self.mlp_gate_up,
                chunks=2,
                dim=-1,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=1 if physical_m == TILE_SIZE else 4,
                    K_block_size=4,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=2,
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                ),
                dtype=self.activation_dtype,
            )
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down_kwargs = {"compute_kernel_config": self.dense_compute_config}
        if self.USE_LARGE_PREFILL_DENSE_CONFIGS and x.padded_shape[-2] == 1024:
            down_kwargs["program_config"] = _large_prefill_dense_program_config(x.padded_shape[-2], projection="down")
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **down_kwargs,
        )

    def _fill_prefill_cache(
        self,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        k_heads: ttnn.Tensor,
        v_heads: ttnn.Tensor,
        page_table: ttnn.Tensor,
        **kwargs: Any,
    ) -> None:
        """Honor reduced cache dtype at fill while keeping decode updates BF16."""

        converted_k = k_heads
        converted_v = v_heads
        if k_heads.dtype != key_cache.dtype:
            converted_k = ttnn.typecast(k_heads, key_cache.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if v_heads.dtype != value_cache.dtype:
            converted_v = ttnn.typecast(v_heads, value_cache.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        try:
            super()._fill_prefill_cache(
                key_cache,
                value_cache,
                converted_k,
                converted_v,
                page_table,
                **kwargs,
            )
        finally:
            if converted_k is not k_heads:
                converted_k.deallocate(True)
            if converted_v is not v_heads:
                converted_v.deallocate(True)

    def _moe_prefill_tile(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Packed BFP8/LoFi prefill with independently tuned sparse K blocks."""

        chunk_len = hidden_states.shape[2]
        group_size = chunk_len // TILE_SIZE
        hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, HIDDEN_SIZE))
        sparsity = ttnn.repeat(self.expert_prefill_sparsity, (1, 1, group_size, 1))
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        up_gate_config = _sparse_program_config(
            TILE_SIZE,
            2 * MOE_INTERMEDIATE_SIZE,
            in0_block_w=self.PREFILL_EXPERT_UP_GATE_IN0_BLOCK_W,
            num_cores=self.PREFILL_EXPERT_UP_GATE_CORES,
            projection="expert_up_gate",
        )
        down_config = _sparse_program_config(
            TILE_SIZE,
            HIDDEN_SIZE,
            in0_block_w=self.PREFILL_EXPERT_DOWN_IN0_BLOCK_W,
            num_cores=self.PREFILL_EXPERT_DOWN_CORES,
            projection="expert_down",
        )

        up_gate = ttnn.sparse_matmul(
            hidden_grouped,
            self.expert_up_gate,
            sparsity=sparsity,
            nnz=NUM_EXPERTS * group_size,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=up_gate_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.expert_compute_config,
        )
        hidden_grouped.deallocate(True)
        packed_intermediate = up_gate.shape[-1]
        up_gate = ttnn.transpose(up_gate, 1, 3)
        up_gate = ttnn.reshape(up_gate, (1, NUM_EXPERTS, chunk_len, packed_intermediate))
        down_input = self._packed_expert_activation(up_gate, use_composite=False)
        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=self.expert_prefill_sparsity,
            nnz=NUM_EXPERTS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
            compute_kernel_config=self.expert_compute_config,
        )
        down_input.deallocate(True)
        next_states = ttnn.reshape(down, (1, NUM_EXPERTS, chunk_len, HIDDEN_SIZE))
        routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))
        next_states = ttnn.mul(next_states, routing_permuted)
        next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
        return ttnn.reshape(next_states, (1, 1, chunk_len, HIDDEN_SIZE))

    def _residual_norm(self, x: ttnn.Tensor, weight: ttnn.Tensor | None) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=weight,
            compute_kernel_config=self.correctness_compute_config,
            memory_config=self.decode_residual_memory_config,
            program_config=self.decode_norm_program_config,
        )

    def _residual_sharded(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.to_memory_config(x, self.decode_residual_memory_config, dtype=x.dtype)

    @staticmethod
    def _residual_interleaved(x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        """Trace-safe batch-1 decode with an 8-core width-sharded residual."""

        if hidden_states.shape[-2] != 1:
            return super().decode_forward(
                hidden_states,
                position_cos=position_cos,
                position_sin=position_sin,
                current_pos=current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                cache_position_modulo=cache_position_modulo,
            )
        residual = self._residual_sharded(hidden_states)
        # Sliding attention sits on the functional PCC edge.  Keep the first
        # normalization's established interleaved reduction order; the
        # residual stream itself remains width-sharded for the layer.
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_decode(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )
        hidden_states = ttnn.add(
            residual,
            self._residual_norm(self._residual_sharded(attn_out), self.weights.post_attn_ln),
            memory_config=self.decode_residual_memory_config,
        )

        residual = hidden_states
        mlp_in = self._residual_interleaved(self._residual_norm(hidden_states, self.weights.pre_ff_ln))
        hidden_1 = self._residual_norm(
            self._residual_sharded(self._dense_mlp(mlp_in, fold_activation=False)),
            self.weights.post_ff_ln_1,
        )
        residual_interleaved = self._residual_interleaved(residual)
        router_weights = self._router_weights(residual_interleaved)
        moe_in = self._residual_interleaved(self._residual_norm(residual, self.weights.pre_ff_ln_2))
        hidden_2 = self._residual_norm(
            self._residual_sharded(self._moe_decode(moe_in, router_weights)),
            self.weights.post_ff_ln_2,
        )
        hidden_states = self._residual_norm(
            ttnn.add(hidden_1, hidden_2, memory_config=self.decode_residual_memory_config),
            self.weights.post_ff_ln,
        )
        hidden_states = ttnn.add(residual, hidden_states, memory_config=self.decode_residual_memory_config)
        return ttnn.mul(hidden_states, self.weights.layer_scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Packed active-expert decode with precision-locked block geometry."""

        batch = hidden_states.shape[2]
        if batch != 1:
            # This stage optimizes and signs off logical batch 1.  Preserve the
            # inherited semantics for other callers without making them a gate.
            return super()._moe_decode_single_user(hidden_states, routing_weights)
        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        up_gate_config = _sparse_program_config(
            batch,
            2 * MOE_INTERMEDIATE_SIZE,
            in0_block_w=self.policy.expert_up_gate_in0_block_w,
            num_cores=self.policy.expert_up_gate_cores,
            projection="expert_up_gate",
        )
        down_config = _sparse_program_config(
            batch,
            HIDDEN_SIZE,
            in0_block_w=self.policy.expert_down_in0_block_w,
            num_cores=self.policy.expert_down_cores,
            projection="expert_down",
        )

        up_gate = ttnn.sparse_matmul(
            hidden_states,
            self.expert_up_gate,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=up_gate_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.expert_compute_config,
        )
        packed_intermediate = up_gate.shape[-1]
        up_gate = ttnn.reshape(up_gate, (batch, NUM_EXPERTS, 1, packed_intermediate))
        up_gate = ttnn.transpose(up_gate, 1, 2)
        up_gate = ttnn.reshape(up_gate, (batch, NUM_EXPERTS, packed_intermediate))
        down_input = self._packed_expert_activation(up_gate, use_composite=False)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, MOE_INTERMEDIATE_SIZE))

        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
            compute_kernel_config=self.expert_compute_config,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(next_states, (1, 1, batch, HIDDEN_SIZE), (1, 1, TILE_SIZE, HIDDEN_SIZE))


__all__ = ["OptimizationPolicy", "OptimizedDecoder", "POLICIES"]
