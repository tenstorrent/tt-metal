# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import torch

import ttnn
import math

from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.layer_weights import MoELayerTTWeights


_SCATTER_ZERO_CACHE: dict[tuple[int, int, int], ttnn.Tensor] = {}


def _get_scatter_zero_tensor(*, device: Any, tokens_per_device: int, num_experts: int) -> ttnn.Tensor:
    """Return a cached zero base tensor for `ttnn.scatter` in reduce dispatch mode.

    Important for tracing:
    - Allocating/initializing device buffers can trigger host writes which are forbidden
      during trace capture.
    - We therefore allocate the zero tensor once (typically during the untraced warmup
      run that happens before `begin_trace_capture`) and reuse it for subsequent steps.

    The tensor is treated as read-only input to `ttnn.scatter` (scatter returns a new
    output tensor), so it is safe to cache without per-step re-zeroing.
    """
    key = (id(device), int(tokens_per_device), int(num_experts))
    cached = _SCATTER_ZERO_CACHE.get(key)
    if cached is not None:
        return cached
    # IMPORTANT: `ttnn.zeros(..., device=MeshDevice)` can create tensors with
    # distributed host buffers that are shape-coupled to the mesh, and we've seen
    # it hang or produce unexpected shapes during bring-up. For the scatter base
    # tensor we want a replicated, plain logical shape [1,1,T,E] across the mesh.
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


def _tt_to_torch_device0(t: ttnn.Tensor) -> torch.Tensor:
    """Best-effort TT -> torch conversion that works for 1x1 mesh bring-up.

    In a true sharded multi-device deployment, this must be replaced with a
    topology-aware gather/compose.
    """
    device_tensors = ttnn.get_device_tensors(t)
    if device_tensors:
        return ttnn.to_torch(device_tensors[0])
    return ttnn.to_torch(t)


def _get_mesh_shape(device: Any) -> tuple[int, int]:
    if device.__class__.__name__ != "MeshDevice":
        return (1, 1)
    # vLLM uses ttnn.MeshShape which is indexable like a 2-tuple.
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
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    grid = device.compute_with_storage_grid_size()
    core_x = int(getattr(grid, "x"))
    core_y = int(getattr(grid, "y"))
    n_tiles = (int(out_features) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    # The sparse matmul 1D program assigns 2D blocks across a 2D core grid and requires the
    # number of blocks not to exceed the number of available cores. Use a conservative
    # per_core_N based on ceil-div to keep num_blocks_x small when out_features > num_cores.
    num_cores = max(1, core_x * core_y)
    per_core_N = max(1, int(math.ceil(n_tiles / num_cores)))
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


@dataclass(frozen=True)
class Glm4MoeLiteMoERuntime:
    """Runtime constants and helper tensors shared across all MoE layers."""

    # Routing helpers
    expert_mapping_tensors: ttnn.Tensor  # [1,1,num_experts,num_devices] row-major uint16
    remap_topk_mask: ttnn.Tensor  # [1,num_dispatch_devices,1,num_experts] row-major bf16
    expert_start_offset: ttnn.Tensor  # per-device scalar (sharded), row-major uint16
    expert_end_offset: ttnn.Tensor  # per-device scalar (sharded), row-major uint16

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


def create_moe_runtime(*, device: Any, hparams: Glm4MoeLiteHParams) -> Glm4MoeLiteMoERuntime:
    num_devices = _get_num_devices(device)
    num_experts = int(hparams.n_routed_experts)
    if num_experts % max(1, num_devices) != 0:
        raise ValueError(f"n_routed_experts={num_experts} must be divisible by num_devices={num_devices}")
    num_experts_per_device = num_experts // max(1, num_devices)

    mesh_rows, mesh_cols = _get_mesh_shape(device)
    # Dispatch across an axis that actually has multiple devices.
    # - On N300 (typical): mesh shape is (1, 8) so we dispatch along cluster_axis=1.
    # - On a 1D row mesh (8, 1): dispatch along cluster_axis=0.
    if mesh_rows > 1:
        dispatch_cluster_axis = 0
    elif mesh_cols > 1:
        dispatch_cluster_axis = 1
    else:
        dispatch_cluster_axis = 0
    reduce_cluster_axis = 1 - dispatch_cluster_axis
    num_dispatch_devices = int((mesh_rows, mesh_cols)[dispatch_cluster_axis])

    mapping = (
        torch.eye(num_devices, dtype=torch.int32)
        .repeat_interleave(num_experts_per_device, dim=0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    expert_mapping_tensors = ttnn.from_torch(
        mapping,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    remap_topk_mask = ttnn.from_torch(
        torch.ones((1, num_dispatch_devices, 1, num_experts), dtype=torch.bfloat16),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Per-device global expert id range [start, end) for the local expert shard.
    # Experts are assigned contiguously per device (see `mapping` above).
    #
    # We shard the scalar offsets along dim0 so each device gets its own start/end
    # for local routing weight construction in replicated-token mode.
    k = int(hparams.num_experts_per_tok)
    # ROW_MAJOR uint16 requires last dim multiple of 2; pad K for scalar broadcast.
    k_pad = max(2, ((k + 1) // 2) * 2)
    # Shape [1, D, 1, K] so we can shard along the device dimension (dim=1) and keep
    # a per-device scalar range for local expert ids.
    expert_starts_torch = (torch.arange(num_devices, dtype=torch.int32) * num_experts_per_device).view(1, num_devices, 1, 1)
    expert_ends_torch = expert_starts_torch + num_experts_per_device
    expert_starts_torch = expert_starts_torch.repeat(1, 1, 1, k_pad)
    expert_ends_torch = expert_ends_torch.repeat(1, 1, 1, k_pad)
    shard0 = ttnn.ShardTensorToMesh(device, dim=1) if device.__class__.__name__ == "MeshDevice" else None
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
    # Sparse matmul program config:
    # - `per_core_M` is tied to the token-tile height (M dimension), not the expert dimension.
    # - We previously (incorrectly) scaled `per_core_M` with experts-per-device, which can yield
    #   invalid configs for `M=32` (1 tile) and produce garbage outputs for inactive experts.
    per_core_M = 1
    gate_up_program_config = _make_sparse_matmul_program_config(
        device=device,
        out_features=int(hparams.moe_intermediate_size),
        in0_block_w=1,
        per_core_M=per_core_M,
    )
    down_program_config = _make_sparse_matmul_program_config(
        device=device,
        out_features=int(hparams.hidden_size),
        in0_block_w=1,
        per_core_M=per_core_M,
    )

    return Glm4MoeLiteMoERuntime(
        expert_mapping_tensors=expert_mapping_tensors,
        remap_topk_mask=remap_topk_mask,
        expert_start_offset=expert_start_offset,
        expert_end_offset=expert_end_offset,
        dispatch_cluster_axis=dispatch_cluster_axis,
        reduce_cluster_axis=reduce_cluster_axis,
        num_links=1,
        topology=ttnn.Topology.Ring,
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
    )


def moe_topk_tt(
    *,
    x: ttnn.Tensor,  # [1,1,T,H] TILE
    moe_w: MoELayerTTWeights,
    hparams: Glm4MoeLiteHParams,
    compute_kernel_config: Any | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return (topk_weights, topk_indices) for routed experts.

    Shapes:
    - topk_weights: [1,1,T,K] TILE bf16
    - topk_indices: [1,1,T,K] TILE uint16
    """
    k = int(hparams.num_experts_per_tok)
    routed_scaling_factor = float(getattr(hparams, "routed_scaling_factor", 1.8))
    norm_topk_prob = bool(getattr(hparams, "norm_topk_prob", True))

    if compute_kernel_config is None:
        logits = ttnn.linear(x, moe_w.w_gate)  # [1,1,T,E]
    else:
        logits = ttnn.linear(x, moe_w.w_gate, compute_kernel_config=compute_kernel_config)  # [1,1,T,E]
    scores = ttnn.sigmoid(logits)
    ttnn.deallocate(logits)

    # scores_for_choice = scores + e_score_correction_bias (broadcast over tokens)
    bias_rm_owned = False
    bias_rm = moe_w.e_score_correction_bias  # [1,1,1,E] ROW_MAJOR (weight tensor; must not deallocate)
    if int(scores.shape[2]) != 1:
        bias_rm = ttnn.repeat(bias_rm, ttnn.Shape((1, 1, scores.shape[2], 1)))
        bias_rm_owned = True
    bias = ttnn.to_layout(bias_rm, ttnn.TILE_LAYOUT)
    if bias_rm_owned:
        ttnn.deallocate(bias_rm)

    scores_with_bias = ttnn.add(scores, bias, dtype=ttnn.bfloat16)
    ttnn.deallocate(bias)

    topk_values, topk_indices = ttnn.topk(scores_with_bias, k=k, dim=-1, largest=True, sorted=False)
    ttnn.deallocate(topk_values)
    ttnn.deallocate(scores_with_bias)

    # Gather weights from the *unbiased* sigmoid scores.
    topk_weights = ttnn.gather(scores, dim=3, index=topk_indices)
    ttnn.deallocate(scores)

    if norm_topk_prob:
        denom = ttnn.sum(topk_weights, dim=3, keepdim=True)
        denom = ttnn.add(denom, 1e-20, output_tensor=denom)
        topk_weights = ttnn.div(topk_weights, denom)
        ttnn.deallocate(denom)

    if routed_scaling_factor != 1.0:
        topk_weights = ttnn.mul(topk_weights, routed_scaling_factor)

    return topk_weights, topk_indices


def moe_topk_cpu_reference(
    *,
    device: Any,
    x: ttnn.Tensor,  # [1,1,T,H] TILE
    moe_w: MoELayerTTWeights,
    hparams: Glm4MoeLiteHParams,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Debug-only CPU reference for MoE routing (top-k indices + weights).

    This matches HF semantics (float32 routing math):
    - router_logits computed in fp32 from fp32(hidden_states)
    - scores = sigmoid(router_logits)
    - indices chosen by topk(scores + correction_bias)
    - weights gathered from unbiased scores (then optional renorm + scaling)

    Returns TT tensors:
    - topk_weights: [1,1,T,K] TILE bf16
    - topk_indices: [1,1,T,K] TILE uint16
    """
    if len(x.shape) != 4:
        raise ValueError(f"expected x rank 4 [1,1,T,H], got shape={tuple(x.shape)}")

    k = int(hparams.num_experts_per_tok)
    routed_scaling_factor = float(getattr(hparams, "routed_scaling_factor", 1.8))
    norm_topk_prob = bool(getattr(hparams, "norm_topk_prob", True))

    # Read inputs/weights back to host.
    x_torch = _tt_to_torch_device0(x).to(dtype=torch.float32)
    # [1,1,T,H] -> [T,H]
    x_2d = x_torch.reshape(-1, int(x_torch.shape[-1]))

    w_gate = _tt_to_torch_device0(moe_w.w_gate).to(dtype=torch.float32)  # [1,1,H,E]
    w_gate_2d = w_gate.reshape(int(w_gate.shape[-2]), int(w_gate.shape[-1]))  # [H,E]

    bias = _tt_to_torch_device0(moe_w.e_score_correction_bias).to(dtype=torch.float32).reshape(-1)  # [E]

    router_logits = x_2d @ w_gate_2d  # [T,E]
    scores = torch.sigmoid(router_logits)
    scores_for_choice = scores + bias.view(1, -1)

    topk = torch.topk(scores_for_choice, k=k, dim=-1, largest=True, sorted=False)
    topk_indices = topk.indices  # [T,K] int64
    topk_weights = torch.gather(scores, 1, topk_indices)  # [T,K] fp32

    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    # Convert back to TT tensors.
    tokens = int(topk_indices.shape[0])
    idx_host = topk_indices.to(dtype=torch.int16).view(1, 1, tokens, k).contiguous()
    w_host = topk_weights.to(dtype=torch.bfloat16).view(1, 1, tokens, k).contiguous()

    is_mesh = device.__class__.__name__ == "MeshDevice"
    mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

    idx_rm = ttnn.from_torch(
        idx_host,
        device=device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    w_rm = ttnn.from_torch(
        w_host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    idx = ttnn.to_layout(idx_rm, ttnn.TILE_LAYOUT)
    w = ttnn.to_layout(w_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(idx_rm)
    ttnn.deallocate(w_rm)
    return w, idx


def moe_dense_experts_forward_decode_tt(
    *,
    device: Any,
    hidden_states: ttnn.Tensor,  # [1,1,1,H] TILE (consumed)
    topk_expert_indices: ttnn.Tensor,  # [1,1,1,K] TILE uint16 (consumed)
    topk_expert_weights: ttnn.Tensor,  # [1,1,1,K] TILE bf16 (consumed)
    moe_w: MoELayerTTWeights,
    hparams: Glm4MoeLiteHParams,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config: Any | None = None,
) -> ttnn.Tensor:
    """Correctness-first dense expert execution for decode (seq_len=1).

    This is a debug bring-up helper used when sparse expert kernels introduce enough
    numerical error to flip greedy tokens.

    Notes:
    - This path reads top-k indices/weights back to host to drive per-expert slicing.
    - It is not intended to be performant.
    """
    if len(hidden_states.shape) != 4 or int(hidden_states.shape[2]) != 1:
        raise ValueError(f"expected hidden_states [1,1,1,H], got shape={tuple(hidden_states.shape)}")
    if len(topk_expert_indices.shape) != 4 or int(topk_expert_indices.shape[2]) != 1:
        raise ValueError(f"expected topk_expert_indices [1,1,1,K], got shape={tuple(topk_expert_indices.shape)}")
    if len(topk_expert_weights.shape) != 4 or int(topk_expert_weights.shape[2]) != 1:
        raise ValueError(f"expected topk_expert_weights [1,1,1,K], got shape={tuple(topk_expert_weights.shape)}")

    hidden = int(hparams.hidden_size)
    inter = int(hparams.moe_intermediate_size)
    k = int(hparams.num_experts_per_tok)

    # Pull indices/weights to host.
    idx_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    w_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    # Consume router tensors.
    ttnn.deallocate(topk_expert_indices)
    ttnn.deallocate(topk_expert_weights)

    idx_host = _tt_to_torch_device0(idx_rm).reshape(-1).to(dtype=torch.int64).cpu().tolist()
    w_host = _tt_to_torch_device0(w_rm).reshape(-1).to(dtype=torch.float32).cpu().tolist()
    ttnn.deallocate(idx_rm)
    ttnn.deallocate(w_rm)
    if len(idx_host) != k or len(w_host) != k:
        raise RuntimeError(f"topk host shapes mismatch: got {len(idx_host)} indices and {len(w_host)} weights, expected {k}")

    out_sum: ttnn.Tensor | None = None
    for expert_id, weight in zip(idx_host, w_host):
        expert_id = int(expert_id)
        if expert_id < 0 or expert_id >= int(hparams.n_routed_experts):
            raise ValueError(f"expert_id out of range: {expert_id}")

        # Slice stacked weights for this expert.
        w1 = ttnn.slice(moe_w.w1_experts, [0, expert_id, 0, 0], [1, expert_id + 1, hidden, inter])  # [1,1,H,I]
        w3 = ttnn.slice(moe_w.w3_experts, [0, expert_id, 0, 0], [1, expert_id + 1, hidden, inter])  # [1,1,H,I]
        w2 = ttnn.slice(moe_w.w2_experts, [0, expert_id, 0, 0], [1, expert_id + 1, inter, hidden])  # [1,1,I,H]

        if compute_kernel_config is None:
            gate = ttnn.linear(hidden_states, w1, memory_config=memory_config)
            up = ttnn.linear(hidden_states, w3, memory_config=memory_config)
        else:
            gate = ttnn.linear(hidden_states, w1, memory_config=memory_config, compute_kernel_config=compute_kernel_config)
            up = ttnn.linear(hidden_states, w3, memory_config=memory_config, compute_kernel_config=compute_kernel_config)
        ttnn.deallocate(w1)
        ttnn.deallocate(w3)

        gate = ttnn.silu(gate)
        x_ff = gate * up
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        if compute_kernel_config is None:
            out = ttnn.linear(x_ff, w2, memory_config=memory_config)  # [1,1,1,H]
        else:
            out = ttnn.linear(x_ff, w2, memory_config=memory_config, compute_kernel_config=compute_kernel_config)  # [1,1,1,H]
        ttnn.deallocate(x_ff)
        ttnn.deallocate(w2)

        if weight != 1.0:
            out = ttnn.mul(out, float(weight), memory_config=memory_config)

        if out_sum is None:
            out_sum = out
        else:
            out_next = ttnn.add(out_sum, out, memory_config=memory_config)
            ttnn.deallocate(out_sum)
            ttnn.deallocate(out)
            out_sum = out_next

    # Consume hidden_states now that experts are computed.
    ttnn.deallocate(hidden_states)

    if out_sum is None:
        raise RuntimeError("dense decode experts produced no output (empty top-k?)")
    return out_sum


def moe_sparse_experts_forward_tt(
    *,
    device: Any,
    hidden_states: ttnn.Tensor,  # [1,1,T,H] TILE
    topk_expert_indices: ttnn.Tensor,  # [1,1,T,K] TILE uint16
    topk_expert_weights: ttnn.Tensor,  # [1,1,T,K] TILE bf16
    moe_w: MoELayerTTWeights,
    rt: Glm4MoeLiteMoERuntime,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """Run routed experts and return routed output [1,1,T,H] TILE.

    Performance note:
    - `ttnn.all_to_all_dispatch` expects input tokens to be sharded across the
      dispatch axis. In our vLLM bring-up (replicated activations on a mesh),
      using all-to-all can inflate the effective token count and crater decode
      throughput.
    - Default path is therefore a replicated-token strategy:
      compute local expert contributions and sum across the mesh via all-reduce.
    - The older all-to-all dispatch/combine path remains available behind
      `GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL=a2a` for future DP-sharded inputs.
    """
    # Default to BF16-speed settings for sparse MoE. Users can opt back into
    # higher-precision accumulation via env overrides if needed for debugging.
    sparse_fidelity = _parse_math_fidelity(
        os.environ.get("GLM4_MOE_LITE_MOE_SPARSE_FIDELITY", ""),
        default=ttnn.MathFidelity.HiFi2,
    )
    sparse_fp32_acc = os.environ.get("GLM4_MOE_LITE_MOE_SPARSE_FP32_ACC", "").strip() == "1"
    sparse_approx = os.environ.get("GLM4_MOE_LITE_MOE_SPARSE_APPROX", "1").strip() != "0"
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=sparse_fidelity,
        math_approx_mode=sparse_approx,
        fp32_dest_acc_en=sparse_fp32_acc,
        packer_l1_acc=False,
    )
    # STEP 0: Put all tokens on dim -2: [1,1,tokens,H]
    input_shape = hidden_states.shape
    tokens_per_device = int(input_shape[0]) * int(input_shape[2])
    hidden_size = int(rt.hidden_size)

    hidden_states = ttnn.reshape(hidden_states, (1, 1, tokens_per_device, hidden_size))
    topk_expert_indices = ttnn.reshape(topk_expert_indices, (1, 1, tokens_per_device, int(rt.num_experts_per_tok)))
    topk_expert_weights = ttnn.reshape(topk_expert_weights, (1, 1, tokens_per_device, int(rt.num_experts_per_tok)))

    mesh_rows, mesh_cols = _get_mesh_shape(device)
    num_devices = _get_num_devices(device)
    num_dispatch_devices = int((mesh_rows, mesh_cols)[rt.dispatch_cluster_axis])
    num_dispatch_devices = max(1, num_dispatch_devices)

    dispatch_impl = os.environ.get("GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL", "").strip().lower()
    if not dispatch_impl:
        dispatch_impl = "reduce"
    if dispatch_impl not in {"reduce", "a2a", "all_to_all"}:
        raise ValueError(
            f"Invalid GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL={dispatch_impl!r}; "
            "expected one of ['reduce','a2a']"
        )
    use_all_to_all = dispatch_impl in {"a2a", "all_to_all"} and num_dispatch_devices > 1

    # If we are not using all-to-all, do not scale token count by dispatch width.
    total_tokens = tokens_per_device * (num_dispatch_devices if use_all_to_all else 1)
    unpadded_total_tokens = total_tokens
    # In replicated-token mode, sparse matmul requires total_tokens to be divisible by the block size.
    # all_to_all mode achieves this by padding to the minimum legal multiple for the dispatch width.
    if not use_all_to_all:
        block = int(rt.sparsity_block_size)
        pad_tokens = (-total_tokens) % block
        if pad_tokens:
            # IMPORTANT: `ttnn.pad` can return a view that aliases the input buffer.
            # Materialize with `ttnn.clone` before deallocating the original tensor.
            hs_padded_view = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
            hs_padded = ttnn.clone(hs_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hidden_states)
            hidden_states = hs_padded

            idx_padded_view = ttnn.pad(topk_expert_indices, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0)
            idx_padded = ttnn.clone(idx_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(topk_expert_indices)
            topk_expert_indices = idx_padded

            w_padded_view = ttnn.pad(topk_expert_weights, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
            w_padded = ttnn.clone(w_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(topk_expert_weights)
            topk_expert_weights = w_padded

            total_tokens += pad_tokens
            tokens_per_device += pad_tokens

    debug = os.environ.get("GLM4_MOE_LITE_MOE_SPARSE_DEBUG", "").strip() == "1"
    if debug:
        device_tensors = ttnn.get_device_tensors(hidden_states)
        mesh_info = "no-mesh"
        if device_tensors:
            mesh_info = f"mesh_tensors={len(device_tensors)} local_shape={tuple(device_tensors[0].shape)}"
        print(
            "[glm4_moe_lite][moe_sparse] "
            f"{mesh_info} dispatch_impl={dispatch_impl} use_all_to_all={use_all_to_all} "
            f"input_shape={tuple(input_shape)} tokens_per_device={tokens_per_device} num_dispatch_devices={num_dispatch_devices} "
            f"total_tokens={total_tokens} hidden={hidden_size} k={int(rt.num_experts_per_tok)}",
            flush=True,
        )

    # The sparse experts implementation materializes an intermediate tensor that
    # scales with `total_tokens * num_experts * moe_intermediate_size`, which
    # can easily exceed device DRAM for long prefills (e.g. agentic UIs that
    # inject large tool schemas/instructions).
    #
    # Chunking is safe because MoE is token-wise (no cross-token dependencies).
    # We chunk based on *global* token count (local tokens * dispatch devices).
    chunk_total_tokens = int(os.environ.get("GLM4_MOE_LITE_MOE_SPARSE_CHUNK_TOKENS", "4096").strip() or "0")
    if chunk_total_tokens > 0 and total_tokens > chunk_total_tokens:
        if debug:
            print(
                "[glm4_moe_lite][moe_sparse] "
                f"chunking enabled: chunk_total_tokens={chunk_total_tokens} total_tokens={total_tokens}",
                flush=True,
            )
        block = int(rt.sparsity_block_size)
        # Convert global chunk budget into per-device chunk size.
        per_device_chunk = max(block, chunk_total_tokens // max(1, num_dispatch_devices))
        per_device_chunk = (per_device_chunk // block) * block
        per_device_chunk = max(block, per_device_chunk)

        out_chunks: list[ttnn.Tensor] = []
        for start in range(0, tokens_per_device, per_device_chunk):
            end = min(start + per_device_chunk, tokens_per_device)
            hs_chunk = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, end, hidden_size])
            idx_chunk = ttnn.slice(
                topk_expert_indices,
                [0, 0, start, 0],
                [1, 1, end, int(rt.num_experts_per_tok)],
            )
            w_chunk = ttnn.slice(
                topk_expert_weights,
                [0, 0, start, 0],
                [1, 1, end, int(rt.num_experts_per_tok)],
            )
            out_chunks.append(
                moe_sparse_experts_forward_tt(
                    device=device,
                    hidden_states=hs_chunk,
                    topk_expert_indices=idx_chunk,
                    topk_expert_weights=w_chunk,
                    moe_w=moe_w,
                    rt=rt,
                    memory_config=memory_config,
                )
            )

        # The chunked calls consume their own slice inputs; we still own the base tensors.
        ttnn.deallocate(hidden_states)
        ttnn.deallocate(topk_expert_indices)
        ttnn.deallocate(topk_expert_weights)

        if len(out_chunks) == 1:
            return out_chunks[0]
        out = ttnn.concat(out_chunks, dim=2)
        for c in out_chunks:
            ttnn.deallocate(c)
        return out

    if use_all_to_all:
        # STEP 1: all_to_all_dispatch expects ROW_MAJOR.
        hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(hidden_states)

        topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(topk_expert_indices)

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
        ttnn.deallocate(hidden_rm)
        ttnn.deallocate(topk_indices_rm)

        if debug:
            do_dt = ttnn.get_device_tensors(dispatch_output)
            dm_dt = ttnn.get_device_tensors(dispatch_metadata)
            print(
                "[glm4_moe_lite][moe_sparse] "
                f"a2a dispatch_output.shape={tuple(dispatch_output.shape)} device_tensors={len(do_dt)} "
                f"dispatch_metadata.shape={tuple(dispatch_metadata.shape)} device_tensors={len(dm_dt)}",
                flush=True,
            )

        # STEP 2: token->expert remap for sparse matmul sparsity.
        remap_mask = ttnn.repeat(rt.remap_topk_mask, ttnn.Shape((1, 1, tokens_per_device, 1)))
        remap_mask = ttnn.reshape(remap_mask, (1, 1, total_tokens, int(rt.num_experts)))
        _, sparsity = ttnn.moe_expert_token_remap(
            remap_mask,
            rt.expert_mapping_tensors,
            dispatch_metadata,
            reduction_size=int(rt.sparsity_block_size),
        )
        # `ttnn.moe_expert_token_remap` returns a UINT16 (0/1) sparsity tensor.
        # `ttnn.sparse_matmul` accepts this UINT16 sparsity directly, and keeping it as
        # UINT16 avoids a device-side typecast on small Row-Major tensors (e.g. last dim
        # = experts_per_device = 8) which currently requires padding to a multiple of 32.
        ttnn.deallocate(remap_mask)

        if debug:
            sp_dt = ttnn.get_device_tensors(sparsity)
            local_shape = tuple(sp_dt[0].shape) if sp_dt else None
            print(
                "[glm4_moe_lite][moe_sparse] "
                f"a2a sparsity.shape={tuple(sparsity.shape)} device_tensors={len(sp_dt)} local_shape={local_shape} dtype={sparsity.dtype} layout={sparsity.layout}",
                flush=True,
            )

        # STEP 3: Prepare dispatch output for expert computation.
        post_dispatch = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens, hidden_size))
        post_dispatch_rm = post_dispatch
        post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)
        ttnn.deallocate(post_dispatch_rm)  # view deallocates dispatch_output
    else:
        # Replicated-token + all-reduce path:
        # - Tokens are replicated across devices, and experts are sharded.
        # - Each device computes its local expert contributions and we sum across the mesh.
        topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(topk_expert_indices)
        ttnn.deallocate(topk_expert_weights)

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
        ttnn.deallocate(topk_weights_rm)
        local_weights, sparsity = ttnn.moe_expert_token_remap(
            topk_weights_dense,
            rt.expert_mapping_tensors,
            topk_indices_rm,
            reduction_size=int(rt.sparsity_block_size),
        )
        # `ttnn.moe_expert_token_remap` returns a UINT16 (0/1) sparsity tensor.
        # `ttnn.sparse_matmul` accepts this UINT16 sparsity directly, and keeping it as
        # UINT16 avoids a device-side typecast on small Row-Major tensors (e.g. last dim
        # = experts_per_device = 8) which currently requires padding to a multiple of 32.
        ttnn.deallocate(topk_weights_dense)
        ttnn.deallocate(topk_indices_rm)

        if debug:
            lw_dt = ttnn.get_device_tensors(local_weights)
            sp_dt = ttnn.get_device_tensors(sparsity)
            lw_local_shape = tuple(lw_dt[0].shape) if lw_dt else None
            sp_local_shape = tuple(sp_dt[0].shape) if sp_dt else None
            print(
                "[glm4_moe_lite][moe_sparse] "
                f"reduce local_weights.shape={tuple(local_weights.shape)} device_tensors={len(lw_dt)} local_shape={lw_local_shape} "
                f"sparsity.shape={tuple(sparsity.shape)} device_tensors={len(sp_dt)} local_shape={sp_local_shape} dtype={sparsity.dtype} layout={sparsity.layout}",
                flush=True,
            )

        post_dispatch = hidden_states
        # hidden_states is consumed by subsequent reshape/sparse matmul; do not deallocate here.

    block = int(rt.sparsity_block_size)
    if total_tokens % block != 0:
        raise ValueError(f"total_tokens={total_tokens} must be divisible by sparsity_block_size={block}")
    num_blocks = total_tokens // block
    expert_input = ttnn.reshape(post_dispatch, shape=(1, num_blocks, block, hidden_size))

    if debug:
        print(
            "[glm4_moe_lite][moe_sparse] "
            f"expert_input.shape={tuple(expert_input.shape)} w1.shape={tuple(moe_w.w1_experts.shape)} w3.shape={tuple(moe_w.w3_experts.shape)} w2.shape={tuple(moe_w.w2_experts.shape)} sparsity.shape={tuple(sparsity.shape)}",
            flush=True,
        )

    # STEP 4: Expert compute (sparse).
    w1_out = ttnn.sparse_matmul(
        expert_input,
        moe_w.w1_experts,
        sparsity=sparsity,
        memory_config=memory_config,
        program_config=rt.gate_up_program_config,
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
        memory_config=memory_config,
        program_config=rt.gate_up_program_config,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
    )
    ttnn.deallocate(expert_input)

    gate = ttnn.silu(w1_out)
    ttnn.deallocate(w1_out)
    x_ff = ttnn.mul(gate, w3_out, memory_config=memory_config)
    ttnn.deallocate(gate)
    ttnn.deallocate(w3_out)

    # Collapse sparse_matmul rank-6 output into [num_blocks, E, block, moe_intermediate]
    x_ff = ttnn.squeeze(x_ff, 0)
    x_ff = ttnn.squeeze(x_ff, 1)

    expert_output_sparse = ttnn.sparse_matmul(
        x_ff,
        moe_w.w2_experts,
        sparsity=sparsity,
        memory_config=memory_config,
        program_config=rt.down_program_config,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        output_tile=ttnn.Tile([block, ttnn.TILE_SIZE]),
    )
    ttnn.deallocate(x_ff)
    ttnn.deallocate(sparsity)

    # STEP 5: Prepare expert output for aggregation.
    while len(expert_output_sparse.shape) > 4:
        expert_output_sparse = ttnn.squeeze(expert_output_sparse, 0)

    expert_output = ttnn.permute(expert_output_sparse, (1, 0, 2, 3))  # [E, num_blocks, block, H]
    ttnn.deallocate(expert_output_sparse)
    expert_output = ttnn.reshape(
        expert_output,
        shape=(int(rt.num_experts_per_device), 1, total_tokens, hidden_size),
    )

    if use_all_to_all:
        # Convert expert output to row-major for all_to_all_combine.
        expert_output_tiled = expert_output
        expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(expert_output_tiled)

        # STEP 6: all_to_all_combine.
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
        ttnn.deallocate(expert_output)
        ttnn.deallocate(dispatch_metadata)

        # STEP 7: Apply routing weights and reduce across K.
        post_combine = ttnn.to_layout(combine_output, ttnn.TILE_LAYOUT)
        ttnn.deallocate(combine_output)

        topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(topk_expert_weights)
        topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 2, 0))  # [K,1,tokens,1]
        topk_weights = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_weights_rm)

        weighted_output = ttnn.mul(post_combine, topk_weights, memory_config=memory_config)
        ttnn.deallocate(post_combine)
        ttnn.deallocate(topk_weights)
        output = ttnn.sum(weighted_output, dim=0, keepdim=True)
        ttnn.deallocate(weighted_output)

        # STEP 8: Aggregate across the *other* mesh axis if present.
        mesh_shape = _get_mesh_shape(device)
        if int(mesh_shape[int(rt.reduce_cluster_axis)]) > 1:
            output_all_reduced = ttnn.all_reduce(
                output,
                num_links=rt.num_links,
                topology=rt.topology,
                cluster_axis=rt.reduce_cluster_axis,
                memory_config=memory_config,
            )
            ttnn.deallocate(output)
            return output_all_reduced

        return output

    # Replicated-token mode:
    # local_weights is [1,1,total_tokens,experts_per_device] ROW_MAJOR.
    # Expand to [experts_per_device,1,total_tokens,hidden] and reduce across experts.
    local_weights_rm = local_weights
    if local_weights_rm.layout != ttnn.ROW_MAJOR_LAYOUT:
        local_weights_rm = ttnn.to_layout(local_weights_rm, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(local_weights)
    local_weights_rm = ttnn.repeat(local_weights_rm, ttnn.Shape((hidden_size, 1, 1, 1)))  # [H,1,T,E]
    local_weights_rm = ttnn.permute(local_weights_rm, (3, 1, 2, 0))  # [E,1,T,H]
    local_weights_tiled = ttnn.to_layout(local_weights_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(local_weights_rm)

    weighted = ttnn.mul(expert_output, local_weights_tiled, memory_config=memory_config)
    ttnn.deallocate(expert_output)
    ttnn.deallocate(local_weights_tiled)
    output = ttnn.sum(weighted, dim=0, keepdim=True)
    ttnn.deallocate(weighted)

    # Sum contributions across devices (experts are sharded across the mesh).
    if num_devices > 1:
        output_all_reduced = ttnn.all_reduce(
            output,
            num_links=rt.num_links,
            topology=rt.topology,
            memory_config=memory_config,
        )
        ttnn.deallocate(output)
        output = output_all_reduced

    # If we padded tokens for block-aligned sparse kernels, slice back to the original size.
    if unpadded_total_tokens != total_tokens:
        output_sliced = ttnn.slice(output, [0, 0, 0, 0], [1, 1, int(unpadded_total_tokens), hidden_size])
        ttnn.deallocate(output)
        output = output_sliced

    return output


__all__ = [
    "Glm4MoeLiteMoERuntime",
    "create_moe_runtime",
    "moe_topk_tt",
    "moe_topk_cpu_reference",
    "moe_dense_experts_forward_decode_tt",
    "moe_sparse_experts_forward_tt",
]
