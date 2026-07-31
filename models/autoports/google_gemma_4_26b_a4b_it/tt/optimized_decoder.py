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
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder
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
        "sliding_dram_qkv_w1",
        "full_dram_qkv_w2",
        "persistent_o_proj",
        "packed_dense",
        "dense_down_w3",
        "expert_gate_grid_w11",
        "expert_up_w11",
        "expert_up_grid_x11",
        "fused_router_scale",
        "prefill_expert_packed_gate_up_grid_11x4",
        "prefill_expert_packed_gate_up_w11",
        "prefill_expert_packed_gate_up_l1",
        "prefill_expert_down_grid_11x8",
        "prefill_expert_down_w11",
        "prefill_expert_down_l1",
        "b32_dram_packed_dense_w4",
        "b32_sliding_dram_qkv_w2",
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


_DRAM_DECODE_ROLE_WIDTHS = {
    "qkv": (1, 2, 4),
    "packed_dense": (1, 2, 4, 8),
    "dense_down": (1, 2, 3, 6),
}


def _has_dram_decode_role(roles: tuple[str, ...], projection: str) -> bool:
    """Return whether either decode batch owns a DRAM-sharded projection role."""

    role_prefixes = (
        f"dram_{projection}_w",
        f"b32_dram_{projection}_w",
        f"sliding_dram_{projection}_w",
        f"full_dram_{projection}_w",
        f"b32_sliding_dram_{projection}_w",
        f"b32_full_dram_{projection}_w",
    )
    return any(role.startswith(role_prefixes) for role in roles)


def _dram_decode_role_width(
    policy: OptimizationPolicy,
    projection: str,
    *,
    batch: int,
    decode_active: bool,
    layer_type: str | None = None,
) -> int | None:
    """Resolve one decode-only DRAM role and validate its adaptable K block width."""

    if not (policy.shard_advisor_seeded and decode_active and batch in (1, 32)):
        return None
    prefixes = [f"b32_dram_{projection}_w" if batch == 32 else f"dram_{projection}_w"]
    if layer_type in ("sliding_attention", "full_attention"):
        batch_prefix = "b32_" if batch == 32 else ""
        prefixes.insert(
            0,
            f"{batch_prefix}{layer_type.removesuffix('_attention')}_dram_{projection}_w",
        )
    matches = [role for role in policy.advisor_roles if any(role.startswith(prefix) for prefix in prefixes)]
    if len(matches) > 1:
        raise ValueError(f"select at most one of {prefixes!r} per layer kind")
    if not matches:
        return None
    prefix = next(prefix for prefix in prefixes if matches[0].startswith(prefix))
    suffix = matches[0].removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError(f"invalid DRAM decode role {matches[0]!r}: expected {prefix}<integer>")
    width = int(suffix)
    allowed_widths = _DRAM_DECODE_ROLE_WIDTHS[projection]
    if width not in allowed_widths:
        raise ValueError(
            f"unsupported {projection} DRAM K block width {width}; expected one of {allowed_widths}"
        )
    return width


def _dram_width_sharded_weight_memory_config(mesh_device: Any, *, k: int, n: int) -> Any:
    """Shard a logical ``[K, N]`` weight across every DRAM bank, padding only physical N."""

    dram_size = mesh_device.dram_grid_size()
    if dram_size.y != 1:
        raise ValueError(f"DRAM-sharded decode expects a one-row DRAM grid, got {dram_size.x}x{dram_size.y}")
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_size.x - 1, 0),
            )
        }
    )
    padded_n = ((n + TILE_SIZE * dram_size.x - 1) // (TILE_SIZE * dram_size.x)) * (
        TILE_SIZE * dram_size.x
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            dram_grid,
            (k, padded_n // dram_size.x),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _decode_l1_width_sharded_memory_config(*, num_cores: int, shard_width: int) -> Any:
    """Build an explicit one-tile-high L1 width shard for the decode DRAM kernel."""

    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_cores - 1, 0),
            )
        }
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_grid,
            (TILE_SIZE, shard_width),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _dram_decode_activation_memory_config(*, k: int) -> Any:
    """Use 11 exact K shards: 8 tiles for 2816 and 6 tiles for 2112."""

    num_cores = 11
    if k % (TILE_SIZE * num_cores):
        raise ValueError(f"decode DRAM activation width {k} cannot shard evenly over {num_cores} cores")
    return _decode_l1_width_sharded_memory_config(num_cores=num_cores, shard_width=k // num_cores)


def _dram_decode_output_geometry(mesh_device: Any, *, n: int) -> tuple[Any, int]:
    """Match output storage width to the physical N width of one DRAM bank."""

    dram_size = mesh_device.dram_grid_size()
    if dram_size.y != 1:
        raise ValueError(f"DRAM-sharded decode expects a one-row DRAM grid, got {dram_size.x}x{dram_size.y}")
    padded_n = ((n + TILE_SIZE * dram_size.x - 1) // (TILE_SIZE * dram_size.x)) * (
        TILE_SIZE * dram_size.x
    )
    shard_width = padded_n // dram_size.x
    memory_config = _decode_l1_width_sharded_memory_config(
        num_cores=dram_size.x,
        shard_width=shard_width,
    )
    return memory_config, shard_width // TILE_SIZE


def _dram_decode_program_config(*, in0_block_w: int, per_core_n: int) -> Any:
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fused_activation=None,
    )


def _blackhole_sparse_program_config(*, m: int, n: int, in0_block_w: int = 1, grid_x: int = 11) -> Any:
    """Use the full 11-column Blackhole grid for Gemma-4 sparse experts.

    The canonical helper is portable across older 8x8 devices and therefore
    selects only two cores for the 22-tile expert intermediate.  Gemma-4's
    Blackhole target can map those tiles exactly to 11x2 cores; packed gate/up
    maps 44 output tiles to 11x4 and the 88-tile hidden output maps to 11x8.
    """

    n_tiles = (n + TILE_SIZE - 1) // TILE_SIZE
    if n_tiles not in (22, 44, 88):
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


_PREFILL_SPARSE_PROJECTIONS = ("gate", "up", "down", "packed_gate_up")
_PREFILL_SPARSE_K_BLOCK_WIDTHS = (1, 2, 4, 8, 11)
_PREFILL_SPARSE_CHUNK_SIZES = (64, 128)
_PREFILL_SPARSE_EXACT_GRIDS = {
    "gate": "11x2",
    "up": "11x2",
    "down": "11x8",
    "packed_gate_up": "11x4",
}


def _prefill_sparse_candidate(
    roles: set[str],
    projection: str,
    *,
    m: int,
    k: int,
    n: int,
) -> tuple[Any, Any]:
    """Resolve one independently selectable sparse-prefill projection candidate."""

    role_prefix = f"prefill_expert_{projection}"
    exact_grid_role = f"{role_prefix}_grid_{_PREFILL_SPARSE_EXACT_GRIDS[projection]}"
    width_roles = {f"{role_prefix}_w{width}": width for width in _PREFILL_SPARSE_K_BLOCK_WIDTHS}
    selected_widths = [width for role, width in width_roles.items() if role in roles]
    if len(selected_widths) > 1:
        raise ValueError(f"select at most one K block width for prefill expert {projection}")
    in0_block_w = selected_widths[0] if selected_widths else 1
    k_tiles = (k + TILE_SIZE - 1) // TILE_SIZE
    if k_tiles % in0_block_w:
        raise ValueError(
            f"prefill expert {projection} K width {in0_block_w} does not divide "
            f"the {k_tiles}-tile contracted dimension"
        )

    l1_role = f"{role_prefix}_l1"
    dram_role = f"{role_prefix}_dram"
    if l1_role in roles and dram_role in roles:
        raise ValueError(f"select only one output memory for prefill expert {projection}")
    memory_config = ttnn.L1_MEMORY_CONFIG if l1_role in roles else ttnn.DRAM_MEMORY_CONFIG

    if exact_grid_role in roles:
        program_config = _blackhole_sparse_program_config(m=m, n=n, in0_block_w=in0_block_w)
    else:
        program_config = _build_sparse_matmul_config(m, n, in0_block_w)
    return program_config, memory_config


class OptimizedDecoder(FusedDecoder):
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
        mesh_device = kwargs["mesh_device"]

        def upload_weight(source: Any, *, dtype: Any, memory_config: Any) -> ttnn.Tensor:
            tensor_kwargs = {
                "device": mesh_device,
                "layout": ttnn.TILE_LAYOUT,
                "dtype": dtype,
                "memory_config": memory_config,
            }
            if isinstance(mesh_device, ttnn.MeshDevice):
                tensor_kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(mesh_device)
            return ttnn.as_tensor(source, **tensor_kwargs)

        packed_expert_role = "prefill_expert_packed_gate_up"
        needs_packed_expert_gate_up = any(
            role == packed_expert_role or role.startswith(f"{packed_expert_role}_")
            for role in selected_policy.advisor_roles
        )
        if needs_packed_expert_gate_up:
            prefix = _detect_layer_prefix(state_dict, kwargs["layer_idx"])
            packed_expert_source = (
                state_dict[f"{prefix}.experts.gate_up_proj"].transpose(-2, -1).contiguous().unsqueeze(0)
            )
            decoder.packed_expert_gate_up = upload_weight(
                packed_expert_source,
                dtype=expert_weight_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        attention_dtype_name = os.getenv("GEMMA4_OPTIMIZED_ATTENTION_DTYPE")
        attention_weight_dtype = weight_dtype
        if attention_dtype_name is not None:
            import torch

            attention_dtype = dtype_by_name[attention_dtype_name]
            attention_weight_dtype = attention_dtype
            prefix = _detect_layer_prefix(state_dict, kwargs["layer_idx"])
            q = state_dict[f"{prefix}.self_attn.q_proj.weight"].transpose(-2, -1).contiguous()
            k = state_dict[f"{prefix}.self_attn.k_proj.weight"].transpose(-2, -1).contiguous()
            v = (
                k
                if decoder.layer_kind.uses_k_as_v
                else state_dict[f"{prefix}.self_attn.v_proj.weight"].transpose(-2, -1).contiguous()
            )

            def attention_weight(source: Any) -> ttnn.Tensor:
                tensor_kwargs = {
                    "device": kwargs["mesh_device"],
                    "layout": ttnn.TILE_LAYOUT,
                    "dtype": attention_dtype,
                    "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                }
                if isinstance(kwargs["mesh_device"], ttnn.MeshDevice):
                    tensor_kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(kwargs["mesh_device"])
                return ttnn.as_tensor(source.unsqueeze(0).unsqueeze(0), **tensor_kwargs)

            decoder.weights = replace(
                decoder.weights,
                qkv=attention_weight(torch.cat([q, k, v], dim=-1)),
                o_proj=attention_weight(
                    state_dict[f"{prefix}.self_attn.o_proj.weight"].transpose(-2, -1).contiguous()
                ),
            )
        if _has_dram_decode_role(selected_policy.advisor_roles, "qkv"):
            import torch

            prefix = _detect_layer_prefix(state_dict, kwargs["layer_idx"])
            q = state_dict[f"{prefix}.self_attn.q_proj.weight"].transpose(-2, -1).contiguous()
            k = state_dict[f"{prefix}.self_attn.k_proj.weight"].transpose(-2, -1).contiguous()
            v = (
                k
                if decoder.layer_kind.uses_k_as_v
                else state_dict[f"{prefix}.self_attn.v_proj.weight"].transpose(-2, -1).contiguous()
            )
            qkv_source = torch.cat([q, k, v], dim=-1).contiguous().unsqueeze(0).unsqueeze(0)
            decoder.dram_qkv = upload_weight(
                qkv_source,
                dtype=attention_weight_dtype,
                memory_config=_dram_width_sharded_weight_memory_config(
                    mesh_device,
                    k=HIDDEN_SIZE,
                    n=decoder.layer_kind.qkv_width,
                ),
            )
        attention_fidelity_name = os.getenv("GEMMA4_OPTIMIZED_ATTENTION_FIDELITY")
        decoder.attention_compute_kernel_config = (
            ttnn.init_device_compute_kernel_config(
                kwargs["mesh_device"].arch(),
                math_fidelity=getattr(ttnn.MathFidelity, attention_fidelity_name),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            if attention_fidelity_name is not None
            else None
        )
        expert_gate_fidelity_name = os.getenv("GEMMA4_OPTIMIZED_EXPERT_GATE_FIDELITY")
        decoder.expert_gate_compute_kernel_config = (
            ttnn.init_device_compute_kernel_config(
                kwargs["mesh_device"].arch(),
                math_fidelity=getattr(ttnn.MathFidelity, expert_gate_fidelity_name),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            if expert_gate_fidelity_name is not None
            else decoder.correctness_compute_config
        )
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
        dram_packed_dense = _has_dram_decode_role(selected_policy.advisor_roles, "packed_dense")
        needs_interleaved_packed_dense = "packed_dense" in selected_policy.advisor_roles or any(
            role.startswith(("prefill_packed_dense_w", "b32_packed_dense_w"))
            for role in selected_policy.advisor_roles
        )
        if needs_interleaved_packed_dense or dram_packed_dense:
            import torch

            prefix = _detect_layer_prefix(state_dict, kwargs["layer_idx"])
            gate = state_dict[f"{prefix}.mlp.gate_proj.weight"].transpose(-2, -1)
            up = state_dict[f"{prefix}.mlp.up_proj.weight"].transpose(-2, -1)
            packed_source = torch.cat([gate, up], dim=-1).contiguous().unsqueeze(0).unsqueeze(0)
            if needs_interleaved_packed_dense:
                decoder.packed_mlp_gate_up = upload_weight(
                    packed_source,
                    dtype=weight_dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if dram_packed_dense:
                decoder.dram_packed_mlp_gate_up = upload_weight(
                    packed_source,
                    dtype=weight_dtype,
                    memory_config=_dram_width_sharded_weight_memory_config(
                        mesh_device,
                        k=HIDDEN_SIZE,
                        n=2 * MLP_INTERMEDIATE_SIZE,
                    ),
                )
        if _has_dram_decode_role(selected_policy.advisor_roles, "dense_down"):
            prefix = _detect_layer_prefix(state_dict, kwargs["layer_idx"])
            down_source = (
                state_dict[f"{prefix}.mlp.down_proj.weight"]
                .transpose(-2, -1)
                .contiguous()
                .unsqueeze(0)
                .unsqueeze(0)
            )
            decoder.dram_mlp_down = upload_weight(
                down_source,
                dtype=weight_dtype,
                memory_config=_dram_width_sharded_weight_memory_config(
                    mesh_device,
                    k=MLP_INTERMEDIATE_SIZE,
                    n=HIDDEN_SIZE,
                ),
            )
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
        self._optimized_decode_active = True
        try:
            return super().decode_forward(hidden_states, **kwargs)
        finally:
            self._optimized_decode_active = False

    def _advisor_role_enabled(self, role: str, batch: int) -> bool:
        return (
            self.optimization_policy.shard_advisor_seeded
            and getattr(self, "_optimized_decode_active", False)
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
        dram_qkv_in0_block_w = _dram_decode_role_width(
            self.optimization_policy,
            "qkv",
            batch=batch,
            decode_active=getattr(self, "_optimized_decode_active", False),
            layer_type=kind.name,
        )
        advisor_qkv = dram_qkv_in0_block_w is None and (
            self._advisor_role_enabled("qkv", batch)
            or (batch == 1 and qkv_sweep_role and kind.name == "sliding_attention")
        )
        qkv_in0_block_w = int(qkv_sweep_role.removeprefix("qkv_local_w")) if qkv_sweep_role else 2
        advisor_o_proj = self._advisor_role_enabled("o_proj", batch)
        dram_o_proj = self._advisor_role_enabled("dram_o_proj", batch)
        persistent_o_proj = self._advisor_role_enabled("persistent_o_proj", batch)
        advisor_o_proj = advisor_o_proj or persistent_o_proj
        if dram_qkv_in0_block_w is not None:
            x = ttnn.to_memory_config(
                x,
                _dram_decode_activation_memory_config(k=HIDDEN_SIZE),
                dtype=self.activation_dtype,
            )
            qkv_output_memory_config, qkv_per_core_n = _dram_decode_output_geometry(
                self.mesh_device,
                n=kind.qkv_width,
            )
        elif advisor_qkv:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=self.activation_dtype)
        xqkv = ttnn.linear(
            x,
            self.dram_qkv if dram_qkv_in0_block_w is not None else self.weights.qkv,
            dtype=self.activation_dtype,
            program_config=(
                _dram_decode_program_config(
                    in0_block_w=dram_qkv_in0_block_w,
                    per_core_n=qkv_per_core_n,
                )
                if dram_qkv_in0_block_w is not None
                else (
                    _advisor_1d_program_config(grid_y=8, in0_block_w=qkv_in0_block_w, out_subblock_w=3)
                    if advisor_qkv
                    else None
                )
            ),
            memory_config=(
                qkv_output_memory_config
                if dram_qkv_in0_block_w is not None
                else (ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if advisor_qkv else ttnn.DRAM_MEMORY_CONFIG)
            ),
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        if dram_qkv_in0_block_w is not None or advisor_qkv:
            # final_ir.mlir %5 -> %6: head-split consumes L1 interleaved,
            # not either QKV matmul's width-sharded output.
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
        native_geometry = (
            key_cache.shape[1] == kind.num_kv_heads
            and key_cache.shape[2] == kind.block_size
            and key_cache.shape[3] == kind.head_dim
        )
        if (
            native_geometry
            and cache_position_modulo is None
            and not (kind.name == "full_attention" and batch == 32)
        ):
            v_update_mem_config = self._disjoint_cache_update_memory_config(
                k_heads.memory_config(), batch, kind.head_dim
            )
            v_heads = ttnn.to_memory_config(v_heads, v_update_mem_config, dtype=v_heads.dtype)
            ttnn.experimental.paged_fused_update_cache(
                key_cache,
                k_heads,
                value_cache,
                v_heads,
                update_idxs_tensor=current_pos,
                page_table=page_table,
            )
        else:
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
            compute_kernel_config=self.attention_compute_kernel_config,
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
        decode_active = getattr(self, "_optimized_decode_active", False)
        dram_packed_in0_block_w = _dram_decode_role_width(
            self.optimization_policy,
            "packed_dense",
            batch=batch,
            decode_active=decode_active,
        )
        dram_down_in0_block_w = _dram_decode_role_width(
            self.optimization_policy,
            "dense_down",
            batch=batch,
            decode_active=decode_active,
        )
        if dram_packed_in0_block_w is not None or dram_down_in0_block_w is not None:
            if dram_packed_in0_block_w is not None:
                x = ttnn.to_memory_config(
                    x,
                    _dram_decode_activation_memory_config(k=HIDDEN_SIZE),
                    dtype=self.activation_dtype,
                )
                packed_output_memory_config, packed_per_core_n = _dram_decode_output_geometry(
                    self.mesh_device,
                    n=2 * MLP_INTERMEDIATE_SIZE,
                )
                packed = ttnn.linear(
                    x,
                    self.dram_packed_mlp_gate_up,
                    dtype=self.activation_dtype,
                    program_config=_dram_decode_program_config(
                        in0_block_w=dram_packed_in0_block_w,
                        per_core_n=packed_per_core_n,
                    ),
                    memory_config=packed_output_memory_config,
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
            else:
                gate = ttnn.linear(
                    x,
                    self.weights.mlp_gate,
                    dtype=self.activation_dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                up = ttnn.linear(
                    x,
                    self.weights.mlp_up,
                    dtype=self.activation_dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if dram_down_in0_block_w is None:
                return ttnn.linear(
                    hidden,
                    self.weights.mlp_down,
                    dtype=self.activation_dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            hidden = ttnn.to_memory_config(
                hidden,
                _dram_decode_activation_memory_config(k=MLP_INTERMEDIATE_SIZE),
                dtype=self.activation_dtype,
            )
            down_output_memory_config, down_per_core_n = _dram_decode_output_geometry(
                self.mesh_device,
                n=HIDDEN_SIZE,
            )
            down = ttnn.linear(
                hidden,
                self.dram_mlp_down,
                dtype=self.activation_dtype,
                program_config=_dram_decode_program_config(
                    in0_block_w=dram_down_in0_block_w,
                    per_core_n=down_per_core_n,
                ),
                memory_config=down_output_memory_config,
            )
            return ttnn.to_memory_config(down, ttnn.DRAM_MEMORY_CONFIG, dtype=self.activation_dtype)

        b32_packed_role = next(
            (role for role in self.optimization_policy.advisor_roles if role.startswith("b32_packed_dense_w")),
            None,
        )
        if decode_active and batch == 32 and b32_packed_role:
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

    def _moe_prefill_chunk(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Run isolated sparse-prefill candidates without changing public chunking semantics."""

        sparse_roles = set(self.optimization_policy.advisor_roles)
        selected_roles = {role for role in sparse_roles if role.startswith("prefill_expert_")}
        if not (self.optimization_policy.shard_advisor_seeded and selected_roles):
            return super()._moe_prefill_chunk(hidden_states, routing_weights)

        known_roles = set()
        for projection in _PREFILL_SPARSE_PROJECTIONS:
            role_prefix = f"prefill_expert_{projection}"
            known_roles.add(f"{role_prefix}_grid_{_PREFILL_SPARSE_EXACT_GRIDS[projection]}")
            known_roles.update(f"{role_prefix}_w{width}" for width in _PREFILL_SPARSE_K_BLOCK_WIDTHS)
            known_roles.update((f"{role_prefix}_l1", f"{role_prefix}_dram"))
        known_roles.add("prefill_expert_packed_gate_up")
        known_roles.update(f"prefill_expert_chunk_{chunk_size}" for chunk_size in _PREFILL_SPARSE_CHUNK_SIZES)
        unknown_roles = selected_roles - known_roles
        if unknown_roles:
            raise ValueError(f"unsupported sparse-prefill advisor roles: {sorted(unknown_roles)}")

        selected_chunk_sizes = [
            chunk_size
            for chunk_size in _PREFILL_SPARSE_CHUNK_SIZES
            if f"prefill_expert_chunk_{chunk_size}" in selected_roles
        ]
        if len(selected_chunk_sizes) > 1:
            raise ValueError("select at most one sparse-prefill chunk size")
        chunk_size = selected_chunk_sizes[0] if selected_chunk_sizes else TILE_SIZE

        seq_len = hidden_states.shape[2]
        assert seq_len % TILE_SIZE == 0, f"Prefill seq_len must be multiple of {TILE_SIZE}, got {seq_len}"
        if seq_len > chunk_size:
            hidden_chunks = ttnn.split(hidden_states, chunk_size, dim=2)
            routing_chunks = ttnn.split(routing_weights, chunk_size, dim=2)
        else:
            hidden_chunks = [hidden_states]
            routing_chunks = [routing_weights]

        result_acc = None
        for hidden_chunk, routing_chunk in zip(hidden_chunks, routing_chunks):
            chunk_result = self._moe_prefill_tile_chunk(hidden_chunk, routing_chunk, sparse_roles)
            if result_acc is None:
                result_acc = chunk_result
            else:
                result_concat = ttnn.concat([result_acc, chunk_result], dim=2)
                result_acc.deallocate(True)
                chunk_result.deallocate(True)
                result_acc = result_concat
        return result_acc

    def _moe_prefill_tile_chunk(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
        sparse_roles: set[str],
    ) -> ttnn.Tensor:
        """Preserve the canonical all-expert 32-token math with selectable configs."""

        chunk_len = hidden_states.shape[2]
        num_experts = self.expert_config.num_experts
        hidden_size = self.expert_config.hidden_size
        intermediate_size = self.expert_weights.intermediate_size_per_device
        group_size = chunk_len // TILE_SIZE
        hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, hidden_size))
        sparsity = ttnn.repeat(self.expert_prefill_sparsity, (1, 1, group_size, 1))
        nnz = num_experts * group_size
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])

        down_config, down_memory_config = _prefill_sparse_candidate(
            sparse_roles,
            "down",
            m=chunk_len,
            k=intermediate_size,
            n=hidden_size,
        )

        packed_expert_role = "prefill_expert_packed_gate_up"
        packed_gate_up = any(
            role == packed_expert_role or role.startswith(f"{packed_expert_role}_") for role in sparse_roles
        )
        if packed_gate_up:
            packed_config, packed_memory_config = _prefill_sparse_candidate(
                sparse_roles,
                "packed_gate_up",
                m=chunk_len,
                k=hidden_size,
                n=2 * intermediate_size,
            )
            packed = ttnn.sparse_matmul(
                hidden_grouped,
                self.packed_expert_gate_up,
                sparsity=sparsity,
                nnz=nnz,
                memory_config=packed_memory_config,
                output_tile=output_tile,
                program_config=packed_config,
                dtype=ttnn.bfloat16,
            )
            hidden_grouped.deallocate(True)
            packed = ttnn.transpose(packed, 1, 3)
            packed = ttnn.reshape(packed, (1, num_experts, chunk_len, 2 * intermediate_size))
            gate = ttnn.slice(
                packed,
                [0, 0, 0, 0],
                [1, num_experts, chunk_len, intermediate_size],
                memory_config=packed_memory_config,
            )
            up = ttnn.slice(
                packed,
                [0, 0, 0, intermediate_size],
                [1, num_experts, chunk_len, 2 * intermediate_size],
                memory_config=packed_memory_config,
            )
            packed.deallocate(True)
            sparse_intermediate = intermediate_size
        else:
            gate_config, gate_memory_config = _prefill_sparse_candidate(
                sparse_roles,
                "gate",
                m=chunk_len,
                k=hidden_size,
                n=intermediate_size,
            )
            up_config, up_memory_config = _prefill_sparse_candidate(
                sparse_roles,
                "up",
                m=chunk_len,
                k=hidden_size,
                n=intermediate_size,
            )
            gate = ttnn.sparse_matmul(
                hidden_grouped,
                self.expert_weights.gate_proj,
                sparsity=sparsity,
                nnz=nnz,
                memory_config=gate_memory_config,
                output_tile=output_tile,
                program_config=gate_config,
                dtype=ttnn.bfloat16,
            )
            sparse_intermediate = gate.shape[-1]
            gate = ttnn.transpose(gate, 1, 3)
            gate = ttnn.reshape(gate, (1, num_experts, chunk_len, sparse_intermediate))

            up = ttnn.sparse_matmul(
                hidden_grouped,
                self.expert_weights.up_proj,
                sparsity=sparsity,
                nnz=nnz,
                memory_config=up_memory_config,
                output_tile=output_tile,
                program_config=up_config,
                dtype=ttnn.bfloat16,
            )
            hidden_grouped.deallocate(True)
            up = ttnn.transpose(up, 1, 3)
            up = ttnn.reshape(up, (1, num_experts, chunk_len, sparse_intermediate))

        down_input = apply_geglu(gate, up)
        down_input = ttnn.reshape(down_input, (1, num_experts, chunk_len, sparse_intermediate))
        down = ttnn.sparse_matmul(
            down_input,
            self.expert_weights.down_proj,
            sparsity=self.expert_prefill_sparsity,
            nnz=num_experts,
            memory_config=down_memory_config,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
        )
        down_input.deallocate(True)

        next_states = ttnn.reshape(down, (1, num_experts, chunk_len, hidden_size))
        routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))
        next_states = ttnn.mul(next_states, routing_permuted)
        next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
        return ttnn.reshape(next_states, (1, 1, chunk_len, hidden_size))

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
            (int(role.removeprefix("expert_up_w")) for role in sparse_roles if role.startswith("expert_up_w")),
            1,
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
            compute_kernel_config=self.expert_gate_compute_kernel_config,
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

    def _moe_decode(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Serialize independent routing rows while dynamically dispatching optimized sparse kernels."""

        return FunctionalDecoder._moe_decode(self, hidden_states, routing_weights)

    def _rms_norm(self, x: ttnn.Tensor, weight: ttnn.Tensor | None) -> ttnn.Tensor:
        """Screen an exact-width sharded decode norm while preserving the public DRAM contract."""

        role_by_weight = {
            id(self.weights.input_ln): "sharded_norm_input_ln",
            id(self.weights.post_attn_ln): "sharded_norm_post_attn_ln",
            id(self.weights.pre_ff_ln): "sharded_norm_pre_ff_ln",
            id(self.weights.post_ff_ln_1): "sharded_norm_post_ff_ln_1",
            id(self.weights.pre_ff_ln_2): "sharded_norm_pre_ff_ln_2",
            id(self.weights.post_ff_ln_2): "sharded_norm_post_ff_ln_2",
            id(self.weights.post_ff_ln): "sharded_norm_post_ff_ln",
        }
        selected_roles = self.optimization_policy.advisor_roles
        if (
            "sharded_rms_norm" not in selected_roles
            and role_by_weight.get(id(weight)) not in selected_roles
        ) or (
            not getattr(self, "_optimized_decode_active", False)
            or x.shape[-1] != HIDDEN_SIZE
        ):
            return super()._rms_norm(x, weight)
        width_tiles = x.shape[-1] // TILE_SIZE
        grid_x = min(11, width_tiles)
        while width_tiles % grid_x:
            grid_x -= 1
        grid_y = min(8, width_tiles // grid_x)
        while width_tiles % (grid_x * grid_y):
            grid_y -= 1
        num_cores = grid_x * grid_y
        input_memory_config = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, x.shape[-1] // num_cores),
            core_grid=ttnn.CoreGrid(x=grid_x, y=grid_y),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        sharded = ttnn.to_memory_config(x, input_memory_config, dtype=x.dtype)
        normalized = ttnn.rms_norm(
            sharded,
            epsilon=self.eps,
            weight=weight,
            program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[grid_x, grid_y],
                subblock_w=1,
                block_h=1,
                block_w=width_tiles // num_cores,
                inplace=False,
            ),
            compute_kernel_config=self.correctness_compute_config,
        )
        return ttnn.sharded_to_interleaved(normalized, ttnn.DRAM_MEMORY_CONFIG)

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
