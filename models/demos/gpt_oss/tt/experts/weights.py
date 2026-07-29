# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generic expert weight loading and management."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig, Mode
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .config import ExpertConfig


@dataclass(frozen=True)  # ✅ Make immutable to prevent accidental modification
class ExpertWeights:
    """Container for expert weight tensors - immutable after creation"""

    gate_proj: ttnn.Tensor
    up_proj: ttnn.Tensor
    down_proj: ttnn.Tensor
    gate_proj_bias: ttnn.Tensor
    up_proj_bias: ttnn.Tensor
    down_proj_bias: ttnn.Tensor
    intermediate_size_per_device: int
    # Fused gate+up weight (concat along N) built ON-DEVICE from the cached gate/up
    # tensors, so one sparse_matmul produces [gate|up]; output splits [..,:I]/[..,I:].
    gate_up_proj: ttnn.Tensor = None
    gateup_bias: ttnn.Tensor = None  # concat[gate_bias|up_bias] [1,E,1,2I] for fused SwiGLU


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
    # Calculate sharded dimensions
    intermediate_size_per_device = mesh_config.shard_size(config.intermediate_size, mode=Mode.DECODE)

    if state_dict:
        # Extract gate and up projections from fused weight
        gate_proj = state_dict["gate_up_proj"][..., ::2].reshape(
            1, config.num_experts, config.hidden_size, config.intermediate_size
        )
        up_proj = state_dict["gate_up_proj"][..., 1::2].reshape(
            1, config.num_experts, config.hidden_size, config.intermediate_size
        )
        gate_proj_bias = state_dict["gate_up_proj_bias"][..., ::2].reshape(
            1, config.num_experts, config.intermediate_size
        )
        up_proj_bias = state_dict["gate_up_proj_bias"][..., 1::2].reshape(
            1, config.num_experts, config.intermediate_size
        )
    else:
        gate_proj = None
        up_proj = None
        gate_proj_bias = None
        up_proj_bias = None
    # Get mesh mappers
    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # Load gate projection
    gate_proj_tt = ttnn.as_tensor(
        gate_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load up projection
    up_proj_tt = ttnn.as_tensor(
        up_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias_dtype = ttnn.bfloat16
    # Load gate bias
    gate_proj_bias_tt = ttnn.as_tensor(
        gate_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load up bias
    up_proj_bias_tt = ttnn.as_tensor(
        up_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj_bias"),
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

    down_proj_bias_tt = ttnn.as_tensor(
        down_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # SwiGLU alpha-fold (build-time): gate*sigmoid(a*gate)*(up+1) = silu(a*gate)/a*(up+1).
    # Pre-scale gate weights+bias by alpha and down weights by 1/alpha ON-DEVICE so the
    # per-token SwiGLU can use a single fused ttnn.silu (no separate mul-by-alpha op),
    # with ZERO runtime correction (the 1/alpha is absorbed into down_proj). Block-float
    # weights are per-block scale-invariant, so folding a scalar is quantization-neutral.
    # Done on-device (not in torch) to stay cache-safe (warm cache = device tensors).
    # The SwiGLU clamp becomes alpha*swiglu_limit (see operations.py / prefill.py).
    _alpha = config.alpha
    gate_proj_tt = ttnn.mul(gate_proj_tt, _alpha, dtype=weight_dtype)
    gate_proj_bias_tt = ttnn.mul(gate_proj_bias_tt, _alpha, dtype=bias_dtype)
    down_proj_tt = ttnn.mul(down_proj_tt, 1.0 / _alpha, dtype=weight_dtype)

    # Build the fused gate+up weight on-device by concatenating the (already cached /
    # loaded) gate and up tensors along the output dim. Works with warm cache
    # (state_dict=None) since gate_proj_tt/up_proj_tt exist as device tensors. One-time
    # at model build. Halves the gate/up sparse_matmul launches at inference.
    gate_up_proj_tt = ttnn.concat([gate_proj_tt, up_proj_tt], dim=3)

    # Prebuild concat[gate_bias|up_bias] as [1,E,1,2I] for the fused SwiGLU (one-time
    # at model build, OUTSIDE the traced decode path). gate/up bias are [1,E,I].
    _E = config.num_experts
    _I = intermediate_size_per_device
    gateup_bias_tt = ttnn.concat(
        [ttnn.reshape(gate_proj_bias_tt, (1, _E, 1, _I)), ttnn.reshape(up_proj_bias_tt, (1, _E, 1, _I))], dim=3
    )

    return ExpertWeights(
        gateup_bias=gateup_bias_tt,
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        gate_proj_bias=gate_proj_bias_tt,
        up_proj_bias=up_proj_bias_tt,
        down_proj_bias=down_proj_bias_tt,
        intermediate_size_per_device=intermediate_size_per_device,
        gate_up_proj=gate_up_proj_tt,
    )
