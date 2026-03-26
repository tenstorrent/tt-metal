# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MoE routing and expert dispatch for GLM-4.7-REAP-218B.

96 routed experts across EP=32 (3 per device) + 1 shared expert (TP=8).
Follows the DSv3 all_to_all dispatch/combine pattern on Galaxy TG Mesh(8,4).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

import ttnn
from models.demos.glm4_moe.tt.ccl import glm4_moe_ccl_num_links_for_axis, glm4_moe_ccl_topology_for_collectives
from models.demos.glm4_moe.tt.config import Glm4MoeHParams
from models.demos.glm4_moe.tt.layer_weights import MoELayerTTWeights

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCATTER_ZERO_CACHE: dict[tuple[int, int, int], ttnn.Tensor] = {}


def _get_scatter_zero_tensor(*, device: Any, tokens_per_device: int, num_experts: int) -> ttnn.Tensor:
    """Return a cached zero base tensor for `ttnn.scatter` in reduce dispatch mode."""
    key = (id(device), int(tokens_per_device), int(num_experts))
    cached = _SCATTER_ZERO_CACHE.get(key)
    if cached is not None:
        return cached
    if device.__class__.__name__ == "MeshDevice":
        host = torch.zeros((1, 1, int(tokens_per_device), int(num_experts)), dtype=torch.bfloat16, device="cpu")
        out = ttnn.as_tensor(
            host,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=None,
        )
    else:
        out = ttnn.zeros(
            ttnn.Shape((1, 1, int(tokens_per_device), int(num_experts))),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    _SCATTER_ZERO_CACHE[key] = out
    return out


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _parse_math_fidelity(value: str, *, default: ttnn.MathFidelity) -> ttnn.MathFidelity:
    raw = value.strip().lower()
    if not raw:
        return default
    table = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi3": ttnn.MathFidelity.HiFi3,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }
    return table.get(raw, default)


def _get_mesh_shape(device: Any) -> tuple[int, int]:
    if device.__class__.__name__ != "MeshDevice":
        return (1, 1)
    return (int(device.shape[0]), int(device.shape[1]))


def _get_num_devices(device: Any) -> int:
    if device.__class__.__name__ != "MeshDevice":
        return 1
    return int(device.get_num_devices())


def _make_sparse_matmul_program_config(
    *,
    device: Any,
    out_features: int,
    in0_block_w: int,
    out_subblock_h: int = 1,
    out_subblock_w: int = 1,
    per_core_M: int = 1,
    worker_grid: tuple[int, int] | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    if worker_grid is not None:
        core_x, core_y = worker_grid
    else:
        grid = device.compute_with_storage_grid_size()
        core_x = int(getattr(grid, "x"))
        core_y = int(getattr(grid, "y"))
    n_tiles = (int(out_features) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    num_cores = max(1, core_x * core_y)
    per_core_N = max(1, int(math.ceil(n_tiles / num_cores)))
    # sparse_matmul requires a rectangular core grid (no holes in the bounding box).
    # Increase per_core_N until ceil(n_tiles / per_core_N) is a multiple of core_x
    # (full rows) or fits in a single row (<= core_x).
    num_blocks = math.ceil(n_tiles / per_core_N)
    while num_blocks > core_x and num_blocks % core_x != 0:
        per_core_N += 1
        num_blocks = math.ceil(n_tiles / per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=int(in0_block_w),
        out_subblock_h=int(out_subblock_h),
        out_subblock_w=int(out_subblock_w),
        out_block_h=1,
        out_block_w=1,
        per_core_M=int(per_core_M),
        per_core_N=int(per_core_N),
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


# ---------------------------------------------------------------------------
# MoE Runtime Constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Glm4MoeMoERuntime:
    """Runtime constants and helper tensors shared across all MoE layers."""

    # Routing helpers
    expert_mapping_tensors: ttnn.Tensor  # [1,1,num_experts,num_devices] row-major uint16
    remap_topk_mask: ttnn.Tensor  # [1,num_dispatch_devices,1,num_experts] row-major bf16
    expert_start_offset: ttnn.Tensor
    expert_end_offset: ttnn.Tensor

    # all_to_all config
    dispatch_cluster_axis: int
    reduce_cluster_axis: int
    num_links: int
    topology: ttnn.Topology
    output_concat_dim: int
    output_shard_dim: int

    # Expert compute
    sparsity_block_size: int
    gate_up_program_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
    down_program_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig

    # Dimensions
    num_experts: int
    num_experts_per_device: int
    num_experts_per_tok: int
    hidden_size: int
    moe_intermediate_size: int

    # Memory config for decode-path expert intermediates
    decode_memory_config: ttnn.MemoryConfig

    # Fused gate+up program config (when GLM4_MOE_FUSE_EXPERTS_GATE_UP=1)
    gate_up_fused_program_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None


def create_moe_runtime(*, device: Any, hparams: Glm4MoeHParams) -> Glm4MoeMoERuntime:
    """Create per-device MoE runtime constants for GLM-4.7-REAP.

    EP=32: 96 experts / 32 devices = 3 experts per device.
    Dispatch on cluster_axis=0 (rows), reduce on cluster_axis=1 (columns).
    """
    num_devices = _get_num_devices(device)
    num_experts = int(hparams.n_routed_experts)
    if num_experts % max(1, num_devices) != 0:
        raise ValueError(f"n_routed_experts={num_experts} must be divisible by num_devices={num_devices}")
    num_experts_per_device = num_experts // max(1, num_devices)

    mesh_rows, mesh_cols = _get_mesh_shape(device)
    # Galaxy TG: Mesh(8,4). Dispatch along rows (axis=0), reduce along columns (axis=1).
    if mesh_rows > 1:
        dispatch_cluster_axis = 0
    elif mesh_cols > 1:
        dispatch_cluster_axis = 1
    else:
        dispatch_cluster_axis = 0
    reduce_cluster_axis = 1 - dispatch_cluster_axis
    num_dispatch_devices = int((mesh_rows, mesh_cols)[dispatch_cluster_axis])

    # Expert mapping: torch.eye(32).repeat_interleave(3, dim=0) -> 96x32
    mapping = (
        torch.eye(num_devices, dtype=torch.int32)
        .repeat_interleave(num_experts_per_device, dim=0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    is_mesh = device.__class__.__name__ == "MeshDevice"
    expert_mapping_tensors = ttnn.from_torch(
        mapping,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    remap_topk_mask = ttnn.from_torch(
        torch.ones((1, num_dispatch_devices, 1, num_experts), dtype=torch.bfloat16),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Per-device expert id range [start, end).
    k = int(hparams.num_experts_per_tok)
    k_pad = max(2, ((k + 1) // 2) * 2)
    expert_starts_torch = (torch.arange(num_devices, dtype=torch.int32) * num_experts_per_device).view(
        1, num_devices, 1, 1
    )
    expert_ends_torch = expert_starts_torch + num_experts_per_device
    expert_starts_torch = expert_starts_torch.repeat(1, 1, 1, k_pad)
    expert_ends_torch = expert_ends_torch.repeat(1, 1, 1, k_pad)
    shard0 = ttnn.ShardTensorToMesh(device, dim=1) if is_mesh else None
    expert_start_offset = ttnn.from_torch(
        expert_starts_torch,
        device=device,
        mesh_mapper=shard0,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    expert_end_offset = ttnn.from_torch(
        expert_ends_torch,
        device=device,
        mesh_mapper=shard0,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    sparsity_block_size = 32
    per_core_M = 1
    gate_up_program_config = _make_sparse_matmul_program_config(
        device=device,
        out_features=int(hparams.moe_intermediate_size),
        in0_block_w=8,
        per_core_M=per_core_M,
    )
    down_program_config = _make_sparse_matmul_program_config(
        device=device,
        out_features=int(hparams.hidden_size),
        in0_block_w=8,
        per_core_M=per_core_M,
    )

    ep_l1 = _env_bool("GLM4_MOE_EP_L1", default=False)
    decode_memory_config = ttnn.L1_MEMORY_CONFIG if ep_l1 else ttnn.DRAM_MEMORY_CONFIG

    fuse_gate_up = _env_bool("GLM4_MOE_FUSE_EXPERTS_GATE_UP", default=False)
    gate_up_fused_program_config = None
    if fuse_gate_up:
        gate_up_fused_program_config = _make_sparse_matmul_program_config(
            device=device,
            out_features=int(hparams.moe_intermediate_size) * 2,
            in0_block_w=8,
            per_core_M=per_core_M,
        )

    _nl = max(glm4_moe_ccl_num_links_for_axis(0), glm4_moe_ccl_num_links_for_axis(1))
    _topo = glm4_moe_ccl_topology_for_collectives()

    return Glm4MoeMoERuntime(
        expert_mapping_tensors=expert_mapping_tensors,
        remap_topk_mask=remap_topk_mask,
        expert_start_offset=expert_start_offset,
        expert_end_offset=expert_end_offset,
        dispatch_cluster_axis=dispatch_cluster_axis,
        reduce_cluster_axis=reduce_cluster_axis,
        num_links=_nl,
        topology=_topo,
        output_concat_dim=2,
        output_shard_dim=2,
        sparsity_block_size=sparsity_block_size,
        gate_up_program_config=gate_up_program_config,
        down_program_config=down_program_config,
        num_experts=num_experts,
        num_experts_per_device=num_experts_per_device,
        num_experts_per_tok=int(hparams.num_experts_per_tok),
        hidden_size=int(hparams.hidden_size),
        moe_intermediate_size=int(hparams.moe_intermediate_size),
        decode_memory_config=decode_memory_config,
        gate_up_fused_program_config=gate_up_fused_program_config,
    )


# ---------------------------------------------------------------------------
# Router: Top-K Gating
# ---------------------------------------------------------------------------


def moe_topk_tt(
    *,
    x: ttnn.Tensor,  # [1,1,T,H] TILE
    moe_w: MoELayerTTWeights,
    hparams: Glm4MoeHParams,
    compute_kernel_config: Any | None = None,
    router_program_config: Any | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return (topk_weights, topk_indices) for routed experts.

    GLM-4.7-REAP routing: sigmoid + e_score_correction_bias + top-8 + normalize + scale(2.5).

    Shapes:
    - topk_weights: [1,1,T,K] TILE bf16
    - topk_indices: [1,1,T,K] TILE uint16
    """
    k = int(hparams.num_experts_per_tok)
    routed_scaling_factor = float(getattr(hparams, "routed_scaling_factor", 2.5))
    norm_topk_prob = bool(getattr(hparams, "norm_topk_prob", True))

    # Use L1 memory for router ops in decode mode (small token count).
    use_l1 = int(x.shape[2]) <= 64  # decode mode only (MoE sees full batch, not DP-split)
    mc = ttnn.L1_MEMORY_CONFIG if use_l1 else None

    linear_kwargs: dict[str, Any] = {}
    if compute_kernel_config is not None:
        linear_kwargs["compute_kernel_config"] = compute_kernel_config
    if mc is not None:
        linear_kwargs["memory_config"] = mc
    if router_program_config is not None:
        linear_kwargs["program_config"] = router_program_config

    logits = ttnn.linear(x, moe_w.w_gate, **linear_kwargs)  # [1,1,T,96]
    scores = ttnn.sigmoid(logits, memory_config=mc) if mc else ttnn.sigmoid(logits)
    ttnn.deallocate(logits, force=False)

    # scores_for_choice = scores + e_score_correction_bias
    if int(scores.shape[2]) == 1 and moe_w.e_score_correction_bias_tile is not None:
        bias = moe_w.e_score_correction_bias_tile
        bias_owned = False
    else:
        bias_rm = moe_w.e_score_correction_bias
        bias_rm_owned = False
        if int(scores.shape[2]) != 1:
            bias_rm = ttnn.repeat(bias_rm, ttnn.Shape((1, 1, scores.shape[2], 1)))
            bias_rm_owned = True
        bias = ttnn.to_layout(bias_rm, ttnn.TILE_LAYOUT)
        if bias_rm_owned:
            ttnn.deallocate(bias_rm, force=False)
        bias_owned = True

    add_kwargs = {"dtype": ttnn.bfloat16}
    if mc is not None:
        add_kwargs["memory_config"] = mc
    scores_with_bias = ttnn.add(scores, bias, **add_kwargs)
    if bias_owned:
        ttnn.deallocate(bias, force=False)

    topk_values, topk_indices = ttnn.topk(scores_with_bias, k=k, dim=-1, largest=True, sorted=False)
    ttnn.deallocate(topk_values, force=False)
    ttnn.deallocate(scores_with_bias, force=False)

    # Gather weights from the *unbiased* sigmoid scores.
    topk_weights = ttnn.gather(scores, dim=3, index=topk_indices)
    ttnn.deallocate(scores, force=False)

    if norm_topk_prob:
        sum_kwargs = {"dim": 3, "keepdim": True}
        if mc is not None:
            sum_kwargs["memory_config"] = mc
        denom = ttnn.sum(topk_weights, **sum_kwargs)
        denom = ttnn.add(denom, 1e-20, output_tensor=denom)
        div_kwargs = {}
        if mc is not None:
            div_kwargs["memory_config"] = mc
        topk_weights = ttnn.div(topk_weights, denom, **div_kwargs)
        ttnn.deallocate(denom, force=False)

    if routed_scaling_factor != 1.0:
        mul_kwargs = {}
        if mc is not None:
            mul_kwargs["memory_config"] = mc
        topk_weights = ttnn.mul(topk_weights, routed_scaling_factor, **mul_kwargs)

    return topk_weights, topk_indices


# ---------------------------------------------------------------------------
# Sparse Expert Forward (replicated-token + all-reduce path)
# ---------------------------------------------------------------------------


def _moe_sparse_tokens_multiple(*, device: Any, moe_runtime: Glm4MoeMoERuntime) -> int:
    """Return the minimum per-device token multiple required by sparse MoE."""
    block = max(1, int(moe_runtime.sparsity_block_size))
    dispatch_axis = int(moe_runtime.dispatch_cluster_axis)
    mesh_rows, mesh_cols = _get_mesh_shape(device)
    dispatch_devices = int((mesh_rows, mesh_cols)[dispatch_axis])
    dispatch_devices = max(1, dispatch_devices)
    return max(1, block // math.gcd(block, dispatch_devices))


def moe_sparse_experts_forward_tt(
    *,
    device: Any,
    hidden_states: ttnn.Tensor,  # [1,1,T,H] TILE
    topk_expert_indices: ttnn.Tensor,  # [1,1,T,K] TILE uint16
    topk_expert_weights: ttnn.Tensor,  # [1,1,T,K] TILE bf16
    moe_w: MoELayerTTWeights,
    rt: Glm4MoeMoERuntime,
    memory_config: ttnn.MemoryConfig,
    skip_final_reduce: bool = False,
) -> ttnn.Tensor:
    """Run routed experts using replicated-token + all-reduce strategy.

    Default path for GLM-4.7-REAP on Galaxy: each device holds 3 experts,
    tokens are replicated, local experts compute contributions, then all-reduce.
    """
    packer_l1_acc = _env_bool("GLM4_MOE_PACKER_L1_ACC", default=False)
    sparse_fidelity = _parse_math_fidelity(
        os.environ.get("GLM4_MOE_MOE_SPARSE_FIDELITY", ""),
        default=ttnn.MathFidelity.HiFi2,
    )
    sparse_fp32_acc = os.environ.get("GLM4_MOE_MOE_SPARSE_FP32_ACC", "").strip() == "1"
    sparse_approx = os.environ.get("GLM4_MOE_MOE_SPARSE_APPROX", "1").strip() != "0"
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=sparse_fidelity,
        math_approx_mode=sparse_approx,
        fp32_dest_acc_en=sparse_fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )

    input_shape = hidden_states.shape
    tokens_per_device = int(input_shape[0]) * int(input_shape[2])
    hidden_size = int(rt.hidden_size)
    num_devices = _get_num_devices(device)

    hidden_states = ttnn.reshape(hidden_states, (1, 1, tokens_per_device, hidden_size))
    topk_expert_indices = ttnn.reshape(topk_expert_indices, (1, 1, tokens_per_device, int(rt.num_experts_per_tok)))
    topk_expert_weights = ttnn.reshape(topk_expert_weights, (1, 1, tokens_per_device, int(rt.num_experts_per_tok)))

    total_tokens = tokens_per_device

    # Pad to sparsity block alignment.
    block = int(rt.sparsity_block_size)
    pad_tokens = (-total_tokens) % block
    if pad_tokens:
        hidden_states = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
        topk_expert_indices = ttnn.pad(topk_expert_indices, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0)
        topk_expert_weights = ttnn.pad(topk_expert_weights, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
        total_tokens += pad_tokens
        tokens_per_device += pad_tokens

    # Build local expert routing weights and sparsity via scatter + moe_expert_token_remap.
    topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(topk_expert_indices, force=False)
    ttnn.deallocate(topk_expert_weights, force=False)

    weights_zero = _get_scatter_zero_tensor(
        device=device, tokens_per_device=tokens_per_device, num_experts=int(rt.num_experts)
    )
    topk_weights_dense = ttnn.scatter(
        weights_zero,
        3,
        topk_indices_rm,
        topk_weights_rm,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(topk_weights_rm, force=False)
    local_weights, sparsity = ttnn.moe_expert_token_remap(
        topk_weights_dense,
        rt.expert_mapping_tensors,
        topk_indices_rm,
        reduction_size=int(rt.sparsity_block_size),
    )
    ttnn.deallocate(topk_weights_dense, force=False)
    ttnn.deallocate(topk_indices_rm, force=False)

    # Expert compute: sparse matmul.
    num_blocks = total_tokens // block
    expert_input = ttnn.reshape(hidden_states, shape=(1, num_blocks, block, hidden_size))

    if num_blocks > 1:
        _gate_up_pc = _make_sparse_matmul_program_config(
            device=device,
            out_features=int(rt.moe_intermediate_size),
            in0_block_w=8,
            per_core_M=num_blocks,
        )
        _down_pc = _make_sparse_matmul_program_config(
            device=device,
            out_features=int(rt.hidden_size),
            in0_block_w=8,
            per_core_M=num_blocks,
        )
        _gate_up_fused_pc = (
            _make_sparse_matmul_program_config(
                device=device,
                out_features=int(rt.moe_intermediate_size) * 2,
                in0_block_w=8,
                per_core_M=num_blocks,
            )
            if rt.gate_up_fused_program_config is not None
            else None
        )
    else:
        _gate_up_pc = rt.gate_up_program_config
        _down_pc = rt.down_program_config
        _gate_up_fused_pc = rt.gate_up_fused_program_config

    sparse_mc = rt.decode_memory_config if total_tokens <= block else memory_config
    if sparse_mc is not ttnn.DRAM_MEMORY_CONFIG:
        expert_input = ttnn.to_memory_config(expert_input, sparse_mc)

    # STEP 4: Expert compute (gate + up + SiLU + down).
    if getattr(moe_w, "w1w3_experts", None) is not None and _gate_up_fused_pc is not None:
        # Fused gate+up projection: single sparse_matmul -> split -> SiLU-gated multiply.
        w1w3_out = ttnn.sparse_matmul(
            expert_input,
            moe_w.w1w3_experts,
            sparsity=sparsity,
            memory_config=sparse_mc,
            program_config=_gate_up_fused_pc,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
        )
        ttnn.deallocate(expert_input, force=False)

        # Split fused output [.., 2*moe_intermediate] into gate (w1) and up (w3) halves.
        moe_inter = int(rt.moe_intermediate_size)
        ndim = len(w1w3_out.shape)
        begin_gate = [0] * ndim
        end_gate = [int(w1w3_out.shape[i]) for i in range(ndim)]
        end_gate[-1] = moe_inter
        begin_up = [0] * ndim
        begin_up[-1] = moe_inter
        end_up = [int(w1w3_out.shape[i]) for i in range(ndim)]

        gate_view = ttnn.slice(w1w3_out, begin_gate, end_gate)
        up_view = ttnn.slice(w1w3_out, begin_up, end_up)
        w1_out = ttnn.typecast(gate_view, dtype=gate_view.dtype, memory_config=sparse_mc)
        w3_out = ttnn.typecast(up_view, dtype=up_view.dtype, memory_config=sparse_mc)
        ttnn.deallocate(w1w3_out, force=False)
    else:
        # Separate gate (w1) and up (w3) projections.
        w1_out = ttnn.sparse_matmul(
            expert_input,
            moe_w.w1_experts,
            sparsity=sparsity,
            memory_config=sparse_mc,
            program_config=_gate_up_pc,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
        )
        w3_out = ttnn.sparse_matmul(
            expert_input,
            moe_w.w3_experts,
            sparsity=sparsity,
            memory_config=sparse_mc,
            program_config=_gate_up_pc,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
        )
        ttnn.deallocate(expert_input, force=False)

    # SiLU(gate) * up.
    x_ff = ttnn.mul(w1_out, w3_out, memory_config=sparse_mc, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(w1_out, force=False)
    ttnn.deallocate(w3_out, force=False)

    # Reshape for down projection.
    _x_ff_target = (num_blocks, int(rt.num_experts_per_device), block, int(rt.moe_intermediate_size))
    if tuple(x_ff.shape) != _x_ff_target:
        x_ff = ttnn.reshape(x_ff, _x_ff_target)

    # Down projection (w2).
    expert_output_sparse = ttnn.sparse_matmul(
        x_ff,
        moe_w.w2_experts,
        sparsity=sparsity,
        memory_config=sparse_mc,
        program_config=_down_pc,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
    )
    ttnn.deallocate(x_ff, force=False)
    ttnn.deallocate(sparsity, force=False)

    # Prepare expert output for aggregation.
    _eos_target = (num_blocks, int(rt.num_experts_per_device), block, hidden_size)
    if tuple(expert_output_sparse.shape) != _eos_target:
        expert_output_sparse = ttnn.reshape(expert_output_sparse, _eos_target)

    expert_output = ttnn.permute(expert_output_sparse, (1, 0, 2, 3))
    expert_output = ttnn.reshape(
        expert_output,
        shape=(int(rt.num_experts_per_device), 1, total_tokens, hidden_size),
    )

    # Apply local routing weights via broadcast mul (no repeat needed).
    # local_weights: [1,1,T,E_local] -> permute -> [E_local,1,T,1] broadcast with [E_local,1,T,H]
    local_weights_rm = ttnn.to_layout(local_weights, ttnn.ROW_MAJOR_LAYOUT)
    local_weights_rm = ttnn.permute(local_weights_rm, (3, 1, 2, 0))  # [E_local,1,T,1]
    local_weights_tiled = ttnn.to_layout(local_weights_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(local_weights_rm, force=False)

    weighted = ttnn.mul(expert_output, local_weights_tiled, memory_config=memory_config)
    ttnn.deallocate(expert_output, force=False)
    ttnn.deallocate(local_weights_tiled, force=False)

    # Sum across local experts.
    routed_out = ttnn.sum(weighted, dim=0, keepdim=True)
    ttnn.deallocate(weighted, force=False)

    # All-reduce across all EP devices (experts sharded across EP=32).
    # Use env var GLM4_MOE_EP_REDUCE to control implementation:
    #   "full_ar" (default) — single all_reduce(no cluster_axis) across all 32 devices
    #   "2step"  — 2-step: all_reduce(axis=0) then all_reduce(axis=1)
    #   "host"   — host-side CPU fallback (2-step per axis)
    if not skip_final_reduce and num_devices > 1:
        ep_reduce = os.environ.get("GLM4_MOE_EP_REDUCE", "full_ar").strip().lower()
        if ep_reduce == "full_ar":
            # Single all_reduce across all 32 devices (no cluster_axis).
            result = ttnn.all_reduce(routed_out, memory_config=memory_config)
            ttnn.deallocate(routed_out, force=False)
            routed_out = result
        elif ep_reduce == "2step":
            from models.demos.glm4_moe.tt.attention_tt import _simple_all_reduce

            routed_out = _simple_all_reduce(routed_out, device, cluster_axis=0, memory_config=memory_config)
            routed_out = _simple_all_reduce(routed_out, device, cluster_axis=1, memory_config=memory_config)
        else:
            # host fallback
            from models.demos.glm4_moe.tt.attention_tt import _simple_all_reduce_host

            routed_out = _simple_all_reduce_host(routed_out, device, cluster_axis=0, memory_config=memory_config)
            routed_out = _simple_all_reduce_host(routed_out, device, cluster_axis=1, memory_config=memory_config)

    # Slice back to unpadded token count.
    unpadded = int(input_shape[0]) * int(input_shape[2])
    if int(routed_out.shape[2]) != unpadded:
        routed_out = ttnn.slice(routed_out, [0, 0, 0, 0], [1, 1, unpadded, hidden_size])

    return routed_out


# ---------------------------------------------------------------------------
# All-to-All Expert Forward (DSv3-style dispatch/combine)
# ---------------------------------------------------------------------------


def moe_a2a_experts_forward_tt(
    *,
    device: Any,
    hidden_states: ttnn.Tensor,  # [1,1,T,H] TILE
    topk_expert_indices: ttnn.Tensor,  # [1,1,T,K] TILE uint16
    topk_expert_weights: ttnn.Tensor,  # [1,1,T,K] TILE bf16
    moe_w: MoELayerTTWeights,
    rt: Glm4MoeMoERuntime,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """Run routed experts using DSv3-style all_to_all dispatch/combine on cluster_axis=0.

    Used when GLM4_MOE_DISPATCH_IMPL=a2a. Suitable for DP-sharded inputs.
    """
    packer_l1_acc = _env_bool("GLM4_MOE_PACKER_L1_ACC", default=False)
    sparse_fidelity = _parse_math_fidelity(
        os.environ.get("GLM4_MOE_MOE_SPARSE_FIDELITY", ""),
        default=ttnn.MathFidelity.HiFi2,
    )
    sparse_fp32_acc = os.environ.get("GLM4_MOE_MOE_SPARSE_FP32_ACC", "").strip() == "1"
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=sparse_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=sparse_fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )

    input_shape = hidden_states.shape
    tokens_per_device = int(input_shape[0]) * int(input_shape[2])
    hidden_size = int(rt.hidden_size)
    mesh_rows, mesh_cols = _get_mesh_shape(device)
    num_dispatch_devices = int((mesh_rows, mesh_cols)[rt.dispatch_cluster_axis])
    total_tokens = tokens_per_device * num_dispatch_devices
    block = int(rt.sparsity_block_size)

    hidden_states = ttnn.reshape(hidden_states, (1, 1, tokens_per_device, hidden_size))
    topk_expert_indices = ttnn.reshape(topk_expert_indices, (1, 1, tokens_per_device, int(rt.num_experts_per_tok)))

    # STEP 1: all_to_all_dispatch (ROW_MAJOR inputs).
    hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(hidden_states, force=False)
    topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(topk_expert_indices, force=False)

    dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
        hidden_rm,
        topk_indices_rm,
        rt.expert_mapping_tensors,
        cluster_axis=rt.dispatch_cluster_axis,
        num_links=rt.num_links,
        topology=rt.topology,
        memory_config=memory_config,
        output_concat_dim=rt.output_concat_dim,
    )
    ttnn.deallocate(hidden_rm, force=False)
    ttnn.deallocate(topk_indices_rm, force=False)

    # STEP 2: Token->expert remap for sparsity.
    remap_mask = ttnn.repeat(rt.remap_topk_mask, ttnn.Shape((1, 1, tokens_per_device, 1)))
    remap_mask = ttnn.reshape(remap_mask, (1, 1, total_tokens, int(rt.num_experts)))
    _, sparsity = ttnn.moe_expert_token_remap(
        remap_mask,
        rt.expert_mapping_tensors,
        dispatch_metadata,
        reduction_size=block,
    )
    ttnn.deallocate(remap_mask, force=False)

    # STEP 3: Prepare for expert compute.
    post_dispatch = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens, hidden_size))
    post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)

    num_blocks = total_tokens // block
    expert_input = ttnn.reshape(post_dispatch, shape=(1, num_blocks, block, hidden_size))

    if num_blocks > 1:
        _gate_up_pc = _make_sparse_matmul_program_config(
            device=device,
            out_features=int(rt.moe_intermediate_size),
            in0_block_w=8,
            per_core_M=num_blocks,
        )
        _down_pc = _make_sparse_matmul_program_config(
            device=device,
            out_features=int(rt.hidden_size),
            in0_block_w=8,
            per_core_M=num_blocks,
        )
        _gate_up_fused_pc = (
            _make_sparse_matmul_program_config(
                device=device,
                out_features=int(rt.moe_intermediate_size) * 2,
                in0_block_w=8,
                per_core_M=num_blocks,
            )
            if rt.gate_up_fused_program_config is not None
            else None
        )
    else:
        _gate_up_pc = rt.gate_up_program_config
        _down_pc = rt.down_program_config
        _gate_up_fused_pc = rt.gate_up_fused_program_config

    sparse_mc = rt.decode_memory_config if total_tokens <= block else memory_config

    # STEP 4: Expert compute (gate + up + SiLU + down).
    if getattr(moe_w, "w1w3_experts", None) is not None and _gate_up_fused_pc is not None:
        # Fused gate+up projection: single sparse_matmul -> split -> SiLU-gated multiply.
        w1w3_out = ttnn.sparse_matmul(
            expert_input,
            moe_w.w1w3_experts,
            sparsity=sparsity,
            memory_config=sparse_mc,
            program_config=_gate_up_fused_pc,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
        )
        ttnn.deallocate(expert_input, force=False)

        # Split fused output [.., 2*moe_intermediate] into gate (w1) and up (w3) halves.
        moe_inter = int(rt.moe_intermediate_size)
        ndim = len(w1w3_out.shape)
        begin_gate = [0] * ndim
        end_gate = [int(w1w3_out.shape[i]) for i in range(ndim)]
        end_gate[-1] = moe_inter
        begin_up = [0] * ndim
        begin_up[-1] = moe_inter
        end_up = [int(w1w3_out.shape[i]) for i in range(ndim)]

        gate_view = ttnn.slice(w1w3_out, begin_gate, end_gate)
        up_view = ttnn.slice(w1w3_out, begin_up, end_up)
        w1_out = ttnn.typecast(gate_view, dtype=gate_view.dtype, memory_config=sparse_mc)
        w3_out = ttnn.typecast(up_view, dtype=up_view.dtype, memory_config=sparse_mc)
        ttnn.deallocate(w1w3_out, force=False)
    else:
        # Separate gate (w1) and up (w3) projections.
        w1_out = ttnn.sparse_matmul(
            expert_input,
            moe_w.w1_experts,
            sparsity=sparsity,
            memory_config=sparse_mc,
            program_config=_gate_up_pc,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
        )
        w3_out = ttnn.sparse_matmul(
            expert_input,
            moe_w.w3_experts,
            sparsity=sparsity,
            memory_config=sparse_mc,
            program_config=_gate_up_pc,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
        )
        ttnn.deallocate(expert_input, force=False)

    x_ff = ttnn.mul(w1_out, w3_out, memory_config=sparse_mc, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(w1_out, force=False)
    ttnn.deallocate(w3_out, force=False)

    _x_ff_target = (num_blocks, int(rt.num_experts_per_device), block, int(rt.moe_intermediate_size))
    if tuple(x_ff.shape) != _x_ff_target:
        x_ff = ttnn.reshape(x_ff, _x_ff_target)

    expert_output_sparse = ttnn.sparse_matmul(
        x_ff,
        moe_w.w2_experts,
        sparsity=sparsity,
        memory_config=sparse_mc,
        program_config=_down_pc,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
    )
    ttnn.deallocate(x_ff, force=False)
    ttnn.deallocate(sparsity, force=False)

    _eos_target = (num_blocks, int(rt.num_experts_per_device), block, hidden_size)
    if tuple(expert_output_sparse.shape) != _eos_target:
        expert_output_sparse = ttnn.reshape(expert_output_sparse, _eos_target)

    expert_output = ttnn.permute(expert_output_sparse, (1, 0, 2, 3))
    expert_output = ttnn.reshape(
        expert_output,
        shape=(int(rt.num_experts_per_device), 1, total_tokens, hidden_size),
    )

    # STEP 5: all_to_all_combine.
    expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)
    dispatch_metadata = ttnn.reshape(
        dispatch_metadata,
        shape=(1, total_tokens, 1, int(rt.num_experts_per_tok)),
    )
    combine_output = ttnn.all_to_all_combine(
        expert_output,
        dispatch_metadata,
        rt.expert_mapping_tensors,
        cluster_axis=rt.dispatch_cluster_axis,
        num_links=rt.num_links,
        topology=rt.topology,
        memory_config=memory_config,
        output_shard_dim=rt.output_shard_dim,
    )
    ttnn.deallocate(expert_output, force=False)
    ttnn.deallocate(dispatch_metadata, force=False)

    # STEP 6: Apply routing weights.
    combine_output = ttnn.reshape(
        combine_output,
        shape=(int(rt.num_experts_per_tok), 1, tokens_per_device, hidden_size),
    )
    combine_output = ttnn.to_layout(combine_output, ttnn.TILE_LAYOUT)

    topk_weights = ttnn.reshape(topk_expert_weights, (1, 1, tokens_per_device, int(rt.num_experts_per_tok)))
    topk_weights_rm = ttnn.to_layout(topk_weights, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(topk_expert_weights, force=False)
    topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 2, 0))  # [K,1,T,1]
    topk_weights_tiled = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(topk_weights_rm, force=False)

    weighted = ttnn.mul(combine_output, topk_weights_tiled, memory_config=memory_config)
    ttnn.deallocate(combine_output, force=False)
    ttnn.deallocate(topk_weights_tiled, force=False)

    routed_out = ttnn.sum(weighted, dim=0, keepdim=True)
    ttnn.deallocate(weighted, force=False)

    # STEP 7: reduce_scatter on cluster_axis=1 (TP dimension).
    routed_out = ttnn.experimental.reduce_scatter_minimal_async(
        routed_out,
        cluster_axis=rt.reduce_cluster_axis,
        dim=3,
        num_links=rt.num_links,
        topology=rt.topology,
        memory_config=memory_config,
    )

    return routed_out


# ---------------------------------------------------------------------------
# Shared Expert MLP (TP=8)
# ---------------------------------------------------------------------------


def shared_expert_forward_tt(
    *,
    x: ttnn.Tensor,  # [1,1,T,H] TILE (H is full hidden or TP-sharded)
    w_gate: ttnn.Tensor,  # column-parallel: [1,1,H,inter_tp]
    w_up: ttnn.Tensor,  # column-parallel: [1,1,H,inter_tp]
    w_down: ttnn.Tensor,  # row-parallel: [1,1,inter_tp,H]
    w_gate_up: ttnn.Tensor | None = None,  # fused: [1,1,H,2*inter_tp]
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    compute_kernel_config: Any | None = None,
    gate_up_program_config: Any | None = None,
    down_program_config: Any | None = None,
) -> ttnn.Tensor:
    """Standard TP-sharded gate-up-SiLU-down MLP for the shared expert."""
    kwargs: dict[str, Any] = {"memory_config": memory_config}
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    gate_up_kwargs = {**kwargs}
    if gate_up_program_config is not None:
        gate_up_kwargs["program_config"] = gate_up_program_config

    if w_gate_up is not None:
        # Fused gate+up: single matmul then slice.
        gate_up = ttnn.linear(x, w_gate_up, **gate_up_kwargs)  # [1,1,T,2*inter_tp]
        inter_tp = gate_up.shape[-1] // 2
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], inter_tp])
        up = ttnn.slice(
            gate_up, [0, 0, 0, inter_tp], [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], gate_up.shape[-1]]
        )
        ttnn.deallocate(gate_up, force=False)
    else:
        gate = ttnn.linear(x, w_gate, **gate_up_kwargs)
        up = ttnn.linear(x, w_up, **gate_up_kwargs)

    x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(gate, force=False)
    ttnn.deallocate(up, force=False)
    down_kwargs = {**kwargs}
    if down_program_config is not None:
        down_kwargs["program_config"] = down_program_config
    out = ttnn.linear(x_ff, w_down, **down_kwargs)
    ttnn.deallocate(x_ff, force=False)
    return out
