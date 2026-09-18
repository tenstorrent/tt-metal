# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generic expert weight loading and management."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig, Mode
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .config import SORTED_MOE_MIN_EXPERTS, ExpertConfig


@dataclass(frozen=True)  # ✅ Make immutable to prevent accidental modification
class ExpertWeights:
    """Container for expert weight tensors - immutable after creation.

    Gate and up projections are stored FUSED along the output dimension so the two projections run
    as one sparse_matmul in decode (the per-expert cost of that op is a fixed overhead, not bandwidth,
    so one call with N = 2 * intermediate is ~half the price of two calls) and as one batched dense
    matmul over all experts in single-row (EP=1, SP=1) prefill. Per device the fused output is
    laid out as [gate (intermediate_padded_per_device) | up (intermediate_padded_per_device)], each
    half zero-padded from intermediate_size_per_device up to a tile multiple so that the halves can
    be split at a tile boundary and fed to SwiGLU / the down projection without any re-layout.

    down_proj's K (rows) is zero-padded once at load from intermediate_size_per_device to
    intermediate_padded_per_device, the width the SwiGLU output carries: dense matmuls check the logical
    K, and the padded activation columns are exactly zero, so one copy serves the sparse (decode, and
    prefill on multi-row meshes: SP>1 or EP>1) and the dense (single-row EP=1, SP=1 prefill) consumers
    alike. No other weight copies persist: the dense prefill path slices the experts it needs per call and
    frees them. The one exception is tiny: for models with fewer than SORTED_MOE_MIN_EXPERTS experts (whose
    long prefill splits run one matmul per expert) the fused gate/up bias is also kept per expert so that
    ttnn.linear can fuse the bias add (E x [1, 1, 1, 2Ip] bf16 tiles, ~1.5 MB per layer for gpt-oss-20b).
    """

    gate_up_proj: ttnn.Tensor  # [1, E, hidden, 2 * intermediate_padded_per_device] per device
    down_proj: ttnn.Tensor  # [1, E, intermediate_padded_per_device, hidden] per device (K zero-padded, see above)
    gate_up_proj_bias: ttnn.Tensor  # [1, E, 2 * intermediate_padded_per_device]
    down_proj_bias: ttnn.Tensor  # [1, E, hidden]; only TP rank 0 holds non-zeros
    intermediate_size_per_device: int
    intermediate_padded_per_device: int
    # Expert-major copy of the fused gate/up bias, [E, 1, 2 * intermediate_padded_per_device]: broadcast
    # directly onto [1, E, tokens, N] activations (prefill and multi-user decode) without a per-call
    # ttnn.transpose. Stored bfloat8_b (the activations it is added to are bfloat8_b).
    gate_up_proj_bias_t: ttnn.Tensor = None
    # Per-expert [1, 1, 1, 2 * intermediate_padded_per_device] bf16 copies of the fused gate/up bias for the dense
    # per-expert prefill loop's fused-bias ttnn.linear; built at load only when num_experts < SORTED_MOE_MIN_EXPERTS.
    gate_up_proj_bias_per_expert: list = None


def _fuse_gate_up_per_device(gate, up, tp, local, padded):
    """Interleave per-device gate and up column blocks: [..., tp * 2 * padded] laid out as
    [gate_dev0 | up_dev0 | gate_dev1 | up_dev1 | ...] with each block zero-padded from `local` to `padded`
    columns, so that column-parallel sharding across `tp` devices gives every device [gate | up]."""
    out_shape = gate.shape[:-1] + (tp * 2 * padded,)
    fused = gate.new_zeros(out_shape)
    for d in range(tp):
        base = d * 2 * padded
        fused[..., base : base + local] = gate[..., d * local : (d + 1) * local]
        fused[..., base + padded : base + padded + local] = up[..., d * local : (d + 1) * local]
    return fused


def load_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat4_b,
    tensor_cache_path=None,
) -> ExpertWeights:
    """
    Load and shard expert weights.

    Args:
        mesh_device: TTNN mesh device
        config: Expert configuration
        state_dict: Dictionary with expert weights
        mesh_config: Mesh parallelization configuration
        weight_dtype: Data type for weights
        tensor_cache_path: Optional path for weight caching

    Returns:
        ExpertWeights with loaded and sharded tensors
    """
    tp = mesh_config.decode.tp
    intermediate_size_per_device = mesh_config.shard_size(config.intermediate_size, mode=Mode.DECODE)
    intermediate_padded_per_device = (
        (intermediate_size_per_device + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    ) * ttnn.TILE_SIZE
    fused_suffix = f"_fused_tp{tp}"

    if state_dict:
        # HF stores gate/up interleaved along the last dim: even columns gate, odd columns up.
        gate = state_dict["gate_up_proj"][..., ::2]  # [E, hidden, intermediate]
        up = state_dict["gate_up_proj"][..., 1::2]
        gate_bias = state_dict["gate_up_proj_bias"][..., ::2]  # [E, intermediate]
        up_bias = state_dict["gate_up_proj_bias"][..., 1::2]
        gate_up_proj = _fuse_gate_up_per_device(
            gate, up, tp, intermediate_size_per_device, intermediate_padded_per_device
        ).reshape(1, config.num_experts, config.hidden_size, tp * 2 * intermediate_padded_per_device)
        gate_up_proj_bias = _fuse_gate_up_per_device(
            gate_bias, up_bias, tp, intermediate_size_per_device, intermediate_padded_per_device
        ).reshape(1, config.num_experts, tp * 2 * intermediate_padded_per_device)
        gate_up_proj_bias_t = gate_up_proj_bias.permute(1, 0, 2).contiguous()  # [E, 1, tp * 2 * padded]
    else:
        gate_up_proj = None
        gate_up_proj_bias = None
        gate_up_proj_bias_t = None
    # Get mesh mappers
    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    gate_up_proj_tt = ttnn.as_tensor(
        gate_up_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_up_proj{fused_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias_dtype = ttnn.bfloat16
    gate_up_proj_bias_tt = ttnn.as_tensor(
        gate_up_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_up_proj_bias{fused_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gate_up_proj_bias_t_tt = ttnn.as_tensor(
        gate_up_proj_bias_t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_up_proj_bias_t_bfp8{fused_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load down projection
    if state_dict:
        down_proj = state_dict["down_proj"].reshape(1, config.num_experts, config.intermediate_size, config.hidden_size)
        down_proj_bias = state_dict["down_proj_bias"].reshape(1, config.num_experts, config.hidden_size)
        # Handle row-parallel bias (must not be replicated across TP devices)
        if mesh_config.decode.tp > 1:
            down_proj_bias = torch.cat(
                [down_proj_bias] + [torch.zeros_like(down_proj_bias)] * (mesh_config.decode.tp - 1), dim=-1
            )
    else:
        down_proj = None
        down_proj_bias = None

    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pad_k = intermediate_padded_per_device - intermediate_size_per_device
    if pad_k > 0:
        # One-time device-side K padding (see the class docstring); the cached file keeps the HF shape. For the
        # block-float weight dtypes ttnn.pad round-trips the layer's shard through bf16 to zero the padding rows
        # (a load-time transient of ~4x the shard, freed before the next layer) and returns a fresh buffer; for
        # bf16/fp32 it pads in place and returns a view of the same buffer, which must then not be freed.
        down_proj_padded = ttnn.pad(down_proj_tt, padding=[(0, 0), (0, 0), (0, pad_k), (0, 0)], value=0.0)
        if weight_dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b):
            down_proj_tt.deallocate(True)
        down_proj_tt = down_proj_padded

    down_proj_bias_tt = ttnn.as_tensor(
        down_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gate_up_proj_bias_per_expert = None
    if config.num_experts < SORTED_MOE_MIN_EXPERTS:
        n = gate_up_proj_bias_t_tt.shape[-1]
        gate_up_proj_bias_per_expert = [
            ttnn.typecast(
                ttnn.reshape(ttnn.slice(gate_up_proj_bias_t_tt, [e, 0, 0], [e + 1, 1, n]), (1, 1, 1, n)), bias_dtype
            )
            for e in range(config.num_experts)
        ]

    return ExpertWeights(
        gate_up_proj=gate_up_proj_tt,
        down_proj=down_proj_tt,
        gate_up_proj_bias=gate_up_proj_bias_tt,
        down_proj_bias=down_proj_bias_tt,
        intermediate_size_per_device=intermediate_size_per_device,
        intermediate_padded_per_device=intermediate_padded_per_device,
        gate_up_proj_bias_t=gate_up_proj_bias_t_tt,
        gate_up_proj_bias_per_expert=gate_up_proj_bias_per_expert,
    )
