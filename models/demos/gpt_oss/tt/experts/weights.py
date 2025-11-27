# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
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

    # Extract gate and up projections from fused weight
    gate_proj = state_dict["gate_up_proj"][..., ::2].reshape(
        1, config.num_experts, config.hidden_size, config.intermediate_size
    )
    up_proj = state_dict["gate_up_proj"][..., 1::2].reshape(
        1, config.num_experts, config.hidden_size, config.intermediate_size
    )
    gate_proj_bias = state_dict["gate_up_proj_bias"][..., ::2].reshape(1, config.num_experts, config.intermediate_size)
    up_proj_bias = state_dict["gate_up_proj_bias"][..., 1::2].reshape(1, config.num_experts, config.intermediate_size)

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
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj_bias_{gate_proj_bias.shape}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load up bias
    up_proj_bias_tt = ttnn.as_tensor(
        up_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj_bias_{up_proj_bias.shape}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load down projection and pad hidden_size dimension for row-sharding
    down_proj = state_dict["down_proj"].reshape(1, config.num_experts, config.intermediate_size, config.hidden_size)
    down_proj_bias = state_dict["down_proj_bias"].reshape(1, config.num_experts, config.hidden_size)

    # Calculate padded hidden size for row-sharding (must be divisible by num_rows * TILE_SIZE)
    # Note: mesh_config may have EP on rows, need to check the actual sharding dimension
    num_rows = mesh_config.mesh_shape[0]
    shard_chunk_size = num_rows * ttnn.TILE_SIZE
    hidden_size = config.hidden_size
    if hidden_size % shard_chunk_size != 0:
        padded_hidden_size = ((hidden_size + shard_chunk_size - 1) // shard_chunk_size) * shard_chunk_size
        padding_size = 192
        assert padding_size % num_rows == 0, "Expert down_proj weight padding size must be divisible by number of rows"
        local_padding_size = padding_size // num_rows  # (padded_hidden_size - hidden_size) / num_rows

        assert hidden_size % num_rows == 0, "Hidden size must be divisible by number of rows"
        local_hidden_size = hidden_size // num_rows
        # Pad last dimension: [1, num_experts, intermediate_size, hidden_size] -> [1, num_experts, intermediate_size, padded_hidden_size]
        down_proj_sliced = [down_proj[:, :, :, i*local_hidden_size:(i+1)*local_hidden_size] for i in range(num_rows)]
        down_proj = torch.cat([torch.nn.functional.pad(d, (0, local_padding_size), value=0.0) for d in down_proj_sliced], dim=-1)
        # Pad last dimension: [1, num_experts, hidden_size] -> [1, num_experts, padded_hidden_size]
        down_proj_bias_sliced = [down_proj_bias[:, :, i*local_hidden_size:(i+1)*local_hidden_size] for i in range(num_rows)]
        down_proj_bias = torch.cat([torch.nn.functional.pad(d, (0, local_padding_size), value=0.0) for d in down_proj_bias_sliced], dim=-1)

    print("down proj", down_proj.shape, down_proj_bias.shape)
    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Handle row-parallel bias (must not be replicated across TP devices)
    if mesh_config.decode.tp > 1:
        down_proj_bias = torch.cat(
            [down_proj_bias] + [torch.zeros_like(down_proj_bias)] * (mesh_config.decode.tp - 1), dim=-1
        )

    down_proj_bias_tt = ttnn.as_tensor(
        down_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj_bias_{down_proj_bias.shape}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return ExpertWeights(
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        gate_proj_bias=gate_proj_bias_tt,
        up_proj_bias=up_proj_bias_tt,
        down_proj_bias=down_proj_bias_tt,
        intermediate_size_per_device=intermediate_size_per_device,
    )
