# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized TTNN decoder layer for ``google/gemma-4-26B-A4B-it``.

The functional decoder remains the semantic reference.  This class owns the
optimized runtime entry points and the optimization policy; tests assert the
concrete class so an accidental functional fallback cannot satisfy this stage.
The initial policy deliberately reproduces the functional path.  Candidate
layout, precision, and program-config changes are added here only after they
pass the real-weight correctness and traced-latency gates documented under
``doc/optimized_decoder``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MLP_INTERMEDIATE_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    NUM_Q_HEADS,
    TILE_SIZE,
    TOP_K_EXPERTS,
    FunctionalDecoder,
    _detect_layer_prefix,
    _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
)
from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config
from models.demos.gemma4.tt.experts.operations import apply_geglu


@dataclass(frozen=True)
class OptimizationPolicy:
    """Explicit policy knobs used by candidate and final optimized runs."""

    name: str = "functional_reproduction"
    decode_layout: str = "dram_interleaved"
    prefill_layout: str = "dram_interleaved"
    attention_weight_dtype: str = "bf16"
    dense_mlp_weight_dtype: str = "bf16"
    expert_weight_dtype: str = "bf16"
    attention_fidelity: str = "functional"
    dense_mlp_fidelity: str = "functional"
    expert_fidelity: str = "functional"
    shard_advisor_seeded: bool = False
    advisor_roles: tuple[str, ...] = ()


ADVISOR_SEED_POLICY = OptimizationPolicy(
    name="shard_advisor_batch1_seed",
    decode_layout="advisor_l1_width_sharded",
    shard_advisor_seeded=True,
    advisor_roles=("qkv", "o_proj", "gate_up", "down"),
)

ADVISOR_SELECTED_POLICY = OptimizationPolicy(
    name="shard_advisor_selected_batch1",
    decode_layout="advisor_selected_l1_width_sharded",
    attention_weight_dtype="bf16",
    dense_mlp_weight_dtype="bf16",
    expert_weight_dtype="bfp8",
    shard_advisor_seeded=True,
    advisor_roles=(
        "qkv_local_w1",
        "persistent_o_proj",
        "packed_dense",
        "dense_down_w3",
        "expert_gate_grid_w11",
        "expert_up_w11",
        "expert_up_b32_w88",
        "expert_up_grid_x11",
        "fused_router_scale",
    ),
)


def _advisor_1d_program_config(*, grid_y: int, in0_block_w: int, out_subblock_w: int) -> Any:
    """Construct the batch-1 program geometry emitted by shard-advise."""

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=out_subblock_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        num_global_cb_receivers=0,
    )


def _blackhole_sparse_program_config(*, m: int, n: int, in0_block_w: int = 1, grid_x: int = 11) -> Any:
    """Use the full 11-column Blackhole grid for Gemma-4 sparse experts.

    The canonical helper is portable across older 8x8 devices and therefore
    selects only two cores for the 22-tile expert intermediate.  Gemma-4's
    Blackhole target can map those tiles exactly to 11x2 cores; the 88-tile
    hidden output similarly maps to 11x8.
    """

    n_tiles = (n + TILE_SIZE - 1) // TILE_SIZE
    if n_tiles not in (22, 88):
        raise ValueError(f"unsupported Gemma-4 sparse output width: {n} ({n_tiles} tiles)")
    grid_y = (n_tiles + grid_x - 1) // grid_x
    if grid_y > 10:
        raise ValueError(f"grid {grid_x}x{grid_y} exceeds the Blackhole worker grid")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=max(TILE_SIZE, m) // TILE_SIZE,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Optimization-owned Gemma-4 decoder.

    Inheriting the proven tensor-contract helpers avoids duplicating paged-cache
    and long-context semantics.  Runtime dispatch is nevertheless owned here:
    both public forward methods are overridden and candidate kernels/configs are
    selected by this class.  This makes a functional-class fallback observable
    to the optimized tests.
    """

    optimization_policy = ADVISOR_SELECTED_POLICY

    def __init__(self, *args: Any, optimization_policy: OptimizationPolicy | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimization_policy = optimization_policy or type(self).optimization_policy
        self.optimized_prefill_invocations = 0
        self.optimized_decode_invocations = 0

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        expert_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        optimization_policy: OptimizationPolicy | None = None,
        **kwargs: Any,
    ) -> "OptimizedDecoder":
        dtype_by_name = {
            "bf16": ttnn.bfloat16,
            "bfp8": ttnn.bfloat8_b,
            "bfp4": ttnn.bfloat4_b,
        }
        selected_policy = optimization_policy or cls.optimization_policy
        policy_weight_dtype = (
            selected_policy.attention_weight_dtype
            if selected_policy.attention_weight_dtype == selected_policy.dense_mlp_weight_dtype
            else "bf16"
        )
        weight_dtype_name = os.getenv("GEMMA4_OPTIMIZED_WEIGHT_DTYPE", policy_weight_dtype)
        expert_dtype_name = os.getenv("GEMMA4_OPTIMIZED_EXPERT_WEIGHT_DTYPE", selected_policy.expert_weight_dtype)
        if weight_dtype_name:
            weight_dtype = dtype_by_name[weight_dtype_name]
        if expert_dtype_name:
            expert_weight_dtype = dtype_by_name[expert_dtype_name]
        decoder = super().from_state_dict(
            state_dict,
            weight_dtype=weight_dtype,
            expert_weight_dtype=expert_weight_dtype,
            activation_dtype=activation_dtype,
            **kwargs,
        )
        decoder.optimization_policy = selected_policy
        if "dram_o_proj" in selected_policy.advisor_roles:
            prefix = _detect_layer_prefix(state_dict, kwargs["layer_idx"])
            source = (
                state_dict[f"{prefix}.self_attn.o_proj.weight"].transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
            )
            dram_size = kwargs["mesh_device"].dram_grid_size()
            dram_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1),
                    )
                }
            )
            padded_n = ((HIDDEN_SIZE + TILE_SIZE * dram_size.x - 1) // (TILE_SIZE * dram_size.x)) * (
                TILE_SIZE * dram_size.x
            )
            memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (source.shape[-2], padded_n // dram_size.x),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            tensor_kwargs = {
                "device": kwargs["mesh_device"],
                "layout": ttnn.TILE_LAYOUT,
                "dtype": weight_dtype,
                "memory_config": memory_config,
            }
            if isinstance(kwargs["mesh_device"], ttnn.MeshDevice):
                tensor_kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(kwargs["mesh_device"])
            dram_o_proj = ttnn.as_tensor(source, **tensor_kwargs)
            decoder.weights = replace(decoder.weights, o_proj=dram_o_proj)
        if "packed_dense" in selected_policy.advisor_roles or any(
            role.startswith(("prefill_packed_dense_w", "b32_packed_dense_w")) for role in selected_policy.advisor_roles
        ):
            import torch

            prefix = _detect_layer_prefix(state_dict, kwargs["layer_idx"])
            gate = state_dict[f"{prefix}.mlp.gate_proj.weight"].transpose(-2, -1)
            up = state_dict[f"{prefix}.mlp.up_proj.weight"].transpose(-2, -1)
            packed_source = torch.cat([gate, up], dim=-1).contiguous().unsqueeze(0).unsqueeze(0)
            tensor_kwargs = {
                "device": kwargs["mesh_device"],
                "layout": ttnn.TILE_LAYOUT,
                "dtype": weight_dtype,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            }
            if isinstance(kwargs["mesh_device"], ttnn.MeshDevice):
                tensor_kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(kwargs["mesh_device"])
            decoder.packed_mlp_gate_up = ttnn.as_tensor(packed_source, **tensor_kwargs)
        if "fused_router_scale" in selected_policy.advisor_roles:
            prefix = _detect_layer_prefix(state_dict, kwargs["layer_idx"])
            fused_scale_source = state_dict[f"{prefix}.router.scale"].reshape(1, 1, 1, HIDDEN_SIZE) * (
                HIDDEN_SIZE**-0.5
            )
            tensor_kwargs = {
                "device": kwargs["mesh_device"],
                "layout": ttnn.TILE_LAYOUT,
                "dtype": ttnn.float32,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            }
            if isinstance(kwargs["mesh_device"], ttnn.MeshDevice):
                tensor_kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(kwargs["mesh_device"])
            decoder.fused_router_scale = ttnn.as_tensor(fused_scale_source, **tensor_kwargs)
        return decoder

    def prefill_forward(self, hidden_states: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        self.optimized_prefill_invocations += 1
        return super().prefill_forward(hidden_states, **kwargs)

    def decode_forward(self, hidden_states: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        self.optimized_decode_invocations += 1
        return super().decode_forward(hidden_states, **kwargs)

    def _advisor_role_enabled(self, role: str, batch: int) -> bool:
        return (
            self.optimization_policy.shard_advisor_seeded
            and batch == 1
            and role in self.optimization_policy.advisor_roles
        )

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        """Decode attention with an explicit, compiler-visible layout contract."""

        kind = self.layer_kind
        batch = x.shape[-2]
        qkv_sweep_role = next(
            (role for role in self.optimization_policy.advisor_roles if role.startswith("qkv_local_w")),
            None,
        )
        advisor_qkv = self._advisor_role_enabled("qkv", batch) or (
            batch == 1 and qkv_sweep_role and kind.name == "sliding_attention"
        )
        qkv_in0_block_w = int(qkv_sweep_role.removeprefix("qkv_local_w")) if qkv_sweep_role else 2
        advisor_o_proj = self._advisor_role_enabled("o_proj", batch)
        dram_o_proj = self._advisor_role_enabled("dram_o_proj", batch)
        persistent_o_proj = self._advisor_role_enabled("persistent_o_proj", batch)
        advisor_o_proj = advisor_o_proj or persistent_o_proj
        if advisor_qkv:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
        xqkv = ttnn.linear(
            x,
            self.weights.qkv,
            dtype=self.activation_dtype,
            program_config=(
                _advisor_1d_program_config(grid_y=8, in0_block_w=qkv_in0_block_w, out_subblock_w=3)
                if advisor_qkv
                else None
            ),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if advisor_qkv else ttnn.DRAM_MEMORY_CONFIG,
        )
        if advisor_qkv:
            # final_ir.mlir %5 -> %6: head-split consumes L1 interleaved,
            # not the QKV matmul's 86-core width-sharded output.
            xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
        head_mem_config = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            memory_config=head_mem_config,
        )
        q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
        v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)

        if kind.name == "full_attention":
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.to_memory_config(q_heads, head_mem_config, dtype=self.activation_dtype)
            k_heads = ttnn.to_memory_config(k_heads, head_mem_config, dtype=self.activation_dtype)
            v_heads = ttnn.to_memory_config(v_heads, head_mem_config, dtype=self.activation_dtype)
        else:
            q_heads = ttnn.to_memory_config(q_heads, head_mem_config, dtype=self.activation_dtype)
            k_heads = ttnn.to_memory_config(k_heads, head_mem_config, dtype=self.activation_dtype)
            v_heads = ttnn.to_memory_config(v_heads, head_mem_config, dtype=self.activation_dtype)
            rope_mem_config = _make_decode_rope_memory_config(self.mesh_device, batch, kind.head_dim)
            position_cos = ttnn.interleaved_to_sharded(position_cos, rope_mem_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, rope_mem_config)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=True)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=True)

        key_cache, value_cache = kv_cache
        update_kwargs = self._cache_view_kwargs(prefill=False)
        if cache_position_modulo is not None:
            update_kwargs["cache_position_modulo"] = cache_position_modulo
        ttnn.experimental.paged_update_cache(
            key_cache,
            k_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            **update_kwargs,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            v_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            **update_kwargs,
        )
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=1.0,
            sliding_window_size=kind.sliding_window,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self._cache_view_kwargs(prefill=False),
        )
        attn_out = ttnn.to_memory_config(attn_out, head_mem_config, dtype=self.activation_dtype)
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=NUM_Q_HEADS)
        attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
        if advisor_o_proj or dram_o_proj:
            input_mem_config = (
                ttnn.create_sharded_memory_config(
                    shape=(TILE_SIZE, 4096),
                    core_grid=ttnn.CoreGrid(x=8, y=1),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                )
                if dram_o_proj
                else ttnn.L1_MEMORY_CONFIG
            )
            attn_out = ttnn.to_memory_config(attn_out, input_mem_config, dtype=self.activation_dtype)
        attn_out = ttnn.linear(
            attn_out,
            self.weights.o_proj,
            dtype=self.activation_dtype,
            program_config=(
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=8,
                    per_core_M=1,
                    per_core_N=8,
                    fused_activation=None,
                )
                if dram_o_proj
                else (_advisor_1d_program_config(grid_y=8, in0_block_w=8, out_subblock_w=1) if advisor_o_proj else None)
            ),
            memory_config=(
                ttnn.create_sharded_memory_config(
                    shape=(TILE_SIZE, HIDDEN_SIZE),
                    core_grid=ttnn.CoreGrid(x=11, y=1),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                )
                if dram_o_proj
                else (ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if advisor_o_proj else ttnn.DRAM_MEMORY_CONFIG)
            ),
        )
        if attn_out.shape[-2] != batch:
            slice_memory_config = ttnn.L1_MEMORY_CONFIG if persistent_o_proj else ttnn.DRAM_MEMORY_CONFIG
            attn_out = ttnn.slice(
                attn_out,
                starts=[0, 0, 0, 0],
                ends=[1, 1, batch, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=slice_memory_config,
            )
        elif (advisor_o_proj or dram_o_proj) and not persistent_o_proj:
            # Preserve the surrounding decoder's current DRAM residual
            # contract for the isolated advisor seed. A coherent sharded
            # residual-chain candidate is measured separately.
            attn_out = ttnn.to_memory_config(attn_out, ttnn.DRAM_MEMORY_CONFIG, dtype=self.activation_dtype)
        return attn_out

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Dense MLP with the advisor's batch-1 layout/program seed."""

        batch = x.shape[-2]
        b32_packed_role = next(
            (role for role in self.optimization_policy.advisor_roles if role.startswith("b32_packed_dense_w")),
            None,
        )
        if batch == 32 and b32_packed_role:
            in0_block_w = int(b32_packed_role.removeprefix("b32_packed_dense_w"))
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
            packed = ttnn.linear(
                x,
                self.packed_mlp_gate_up,
                dtype=self.activation_dtype,
                program_config=_advisor_1d_program_config(grid_y=6, in0_block_w=in0_block_w, out_subblock_w=2),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            gate = ttnn.slice(
                packed,
                [0, 0, 0, 0],
                [1, 1, batch, MLP_INTERMEDIATE_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                packed,
                [0, 0, 0, MLP_INTERMEDIATE_SIZE],
                [1, 1, batch, 2 * MLP_INTERMEDIATE_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            hidden = ttnn.mul(
                ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                up,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            b32_down_role = next(
                (role for role in self.optimization_policy.advisor_roles if role.startswith("b32_dense_down_w")),
                None,
            )
            if b32_down_role:
                hidden = ttnn.to_memory_config(hidden, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
                down = ttnn.linear(
                    hidden,
                    self.weights.mlp_down,
                    dtype=self.activation_dtype,
                    program_config=_advisor_1d_program_config(
                        grid_y=8,
                        in0_block_w=int(b32_down_role.removeprefix("b32_dense_down_w")),
                        out_subblock_w=1,
                    ),
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                return ttnn.to_memory_config(down, ttnn.DRAM_MEMORY_CONFIG, dtype=self.activation_dtype)
            return ttnn.linear(
                hidden,
                self.weights.mlp_down,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        prefill_packed_role = next(
            (role for role in self.optimization_policy.advisor_roles if role.startswith("prefill_packed_dense_w")),
            None,
        )
        if batch >= TILE_SIZE and prefill_packed_role:
            in0_block_w = int(prefill_packed_role.removeprefix("prefill_packed_dense_w"))
            packed = ttnn.linear(
                x,
                self.packed_mlp_gate_up,
                dtype=self.activation_dtype,
                program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(11, 8),
                    in0_block_w=in0_block_w,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=max(1, (batch // TILE_SIZE + 7) // 8),
                    per_core_N=12,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.slice(
                packed,
                [0, 0, 0, 0],
                [1, 1, batch, MLP_INTERMEDIATE_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                packed,
                [0, 0, 0, MLP_INTERMEDIATE_SIZE],
                [1, 1, batch, 2 * MLP_INTERMEDIATE_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.linear(
                hidden,
                self.weights.mlp_down,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self._advisor_role_enabled("packed_dense", batch):
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
            packed = ttnn.linear(
                x,
                self.packed_mlp_gate_up,
                dtype=self.activation_dtype,
                program_config=_advisor_1d_program_config(grid_y=6, in0_block_w=8, out_subblock_w=2),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            gate = ttnn.slice(
                packed,
                [0, 0, 0, 0],
                [1, 1, batch, MLP_INTERMEDIATE_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                packed,
                [0, 0, 0, MLP_INTERMEDIATE_SIZE],
                [1, 1, batch, 2 * MLP_INTERMEDIATE_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            dense_down_role = next(
                (role for role in self.optimization_policy.advisor_roles if role.startswith("dense_down_w")),
                None,
            )
            if self._advisor_role_enabled("down", batch) or (batch == 1 and dense_down_role):
                dense_down_in0_block_w = int(dense_down_role.removeprefix("dense_down_w")) if dense_down_role else 2
                hidden = ttnn.to_memory_config(hidden, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
                down = ttnn.linear(
                    hidden,
                    self.weights.mlp_down,
                    dtype=self.activation_dtype,
                    program_config=_advisor_1d_program_config(
                        grid_y=8, in0_block_w=dense_down_in0_block_w, out_subblock_w=1
                    ),
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                return ttnn.to_memory_config(down, ttnn.DRAM_MEMORY_CONFIG, dtype=self.activation_dtype)
            return ttnn.linear(
                hidden,
                self.weights.mlp_down,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        advisor_gate_up = self._advisor_role_enabled("gate_up", batch)
        advisor_down = self._advisor_role_enabled("down", batch)
        if not (advisor_gate_up or advisor_down):
            return super()._dense_mlp(x)
        if advisor_gate_up:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
            gate_up_config = _advisor_1d_program_config(grid_y=6, in0_block_w=8, out_subblock_w=1)
            gate = ttnn.linear(
                x,
                self.weights.mlp_gate,
                dtype=self.activation_dtype,
                program_config=gate_up_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            up = ttnn.linear(
                x,
                self.weights.mlp_up,
                dtype=self.activation_dtype,
                program_config=gate_up_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            gate = ttnn.gelu(
                gate,
                fast_and_approximate_mode=True,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            hidden = ttnn.mul(gate, up, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        else:
            gate = ttnn.linear(
                x, self.weights.mlp_gate, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            up = ttnn.linear(x, self.weights.mlp_up, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if advisor_down:
            hidden = ttnn.to_memory_config(hidden, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
            down = ttnn.linear(
                hidden,
                self.weights.mlp_down,
                dtype=self.activation_dtype,
                program_config=_advisor_1d_program_config(grid_y=8, in0_block_w=2, out_subblock_w=1),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            return ttnn.to_memory_config(down, ttnn.DRAM_MEMORY_CONFIG, dtype=self.activation_dtype)
        hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG, dtype=self.activation_dtype)
        return ttnn.linear(
            hidden, self.weights.mlp_down, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Use Blackhole-wide sparse projection grids when selected."""

        batch = hidden_states.shape[2]
        sparse_roles = set(self.optimization_policy.advisor_roles)
        gate_grid_roles = {
            "experts",
            "expert_gate_grid_w1",
            "expert_gate_grid_w2",
            "expert_gate_grid_w4",
            "expert_gate_grid_w8",
            "expert_gate_grid_w11",
        }
        sparse_sweep_roles = {
            role
            for role in sparse_roles
            if role.startswith("expert_up_w")
            or role.startswith("expert_up_b32_w")
            or role.startswith("expert_down_w")
            or role.startswith("expert_up_grid_x")
            or role in {"expert_up_dram", "expert_down_dram"}
        }
        if not (
            self.optimization_policy.shard_advisor_seeded
            and batch in (1, 32)
            and (sparse_roles.intersection(gate_grid_roles) or sparse_sweep_roles)
        ):
            return super()._moe_decode_single_user(hidden_states, routing_weights)

        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_in0_block_w = 1
        if "expert_gate_grid_w11" in sparse_roles:
            gate_in0_block_w = 11
        elif "expert_gate_grid_w8" in sparse_roles:
            gate_in0_block_w = 8
        elif "expert_gate_grid_w2" in sparse_roles:
            gate_in0_block_w = 2
        elif "expert_gate_grid_w4" in sparse_roles:
            gate_in0_block_w = 4
        gate_config = (
            _blackhole_sparse_program_config(m=batch, n=MOE_INTERMEDIATE_SIZE, in0_block_w=gate_in0_block_w)
            if sparse_roles.intersection(gate_grid_roles)
            else _build_sparse_matmul_config(batch, MOE_INTERMEDIATE_SIZE)
        )
        use_all_wide_grids = "experts" in sparse_roles
        up_in0_block_w = next(
            (
                int(role.removeprefix("expert_up_b32_w"))
                for role in sparse_roles
                if batch == 32 and role.startswith("expert_up_b32_w")
            ),
            next(
                (int(role.removeprefix("expert_up_w")) for role in sparse_roles if role.startswith("expert_up_w")),
                1,
            ),
        )
        up_grid_x = next(
            (
                int(role.removeprefix("expert_up_grid_x"))
                for role in sparse_roles
                if role.startswith("expert_up_grid_x")
            ),
            None,
        )
        down_in0_block_w = next(
            (int(role.removeprefix("expert_down_w")) for role in sparse_roles if role.startswith("expert_down_w")),
            1,
        )
        up_config = (
            _blackhole_sparse_program_config(
                m=batch,
                n=MOE_INTERMEDIATE_SIZE,
                in0_block_w=up_in0_block_w,
                grid_x=up_grid_x or 11,
            )
            if use_all_wide_grids or up_grid_x is not None
            else _build_sparse_matmul_config(batch, MOE_INTERMEDIATE_SIZE, up_in0_block_w)
        )
        down_config = (
            _blackhole_sparse_program_config(m=batch, n=HIDDEN_SIZE)
            if use_all_wide_grids
            else _build_sparse_matmul_config(batch, HIDDEN_SIZE, down_in0_block_w)
        )
        up_memory_config = ttnn.DRAM_MEMORY_CONFIG if "expert_up_dram" in sparse_roles else ttnn.L1_MEMORY_CONFIG
        down_memory_config = ttnn.DRAM_MEMORY_CONFIG if "expert_down_dram" in sparse_roles else ttnn.L1_MEMORY_CONFIG
        gate = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_gate,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.correctness_compute_config,
        )
        sparse_intermediate = gate.shape[-1]
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, sparse_intermediate))
        up = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_up,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=up_memory_config,
            output_tile=output_tile,
            program_config=up_config,
            dtype=self.activation_dtype,
        )
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, sparse_intermediate))
        down_input = apply_geglu(gate, up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, sparse_intermediate))
        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=down_memory_config,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
        )

    def _router_weights(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        """Fuse the two static router input scales at load time."""

        tokens = residual.shape[-2]
        if "fused_router_scale" not in self.optimization_policy.advisor_roles:
            return super()._router_weights(residual)
        router_in = self._rms_norm(residual, None)
        router_in = ttnn.mul(router_in, self.fused_router_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_in = ttnn.reshape(router_in, [tokens, HIDDEN_SIZE])
        router_in = ttnn.typecast(router_in, ttnn.float32)
        logits = ttnn.linear(
            router_in,
            self.weights.router_proj,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=TOP_K_EXPERTS, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        routing = ttnn.mul(
            routing,
            self.weights.router_per_expert_scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.typecast(routing, ttnn.bfloat16)
        return ttnn.reshape(routing, [1, 1, tokens, NUM_EXPERTS])


__all__ = [
    "ADVISOR_SEED_POLICY",
    "ADVISOR_SELECTED_POLICY",
    "OptimizationPolicy",
    "OptimizedDecoder",
]
