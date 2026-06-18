# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generic expert weight loading and management."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.minimax_m3.config import MeshConfig, Mode
from models.demos.minimax_m3.utils.general_utils import get_cache_file_name

from .config import ExpertConfig


@dataclass(frozen=True)  # ✅ Make immutable to prevent accidental modification
class ExpertWeights:
    """Container for expert weight tensors - immutable after creation.

    MiniMax-M2 experts (MiniMaxM2MLP) are separate w1 (gate), w3 (up), w2 (down)
    Linears with NO bias and plain SiLU SwiGLU.
    """

    gate_proj: ttnn.Tensor  # w1, [1, num_experts, hidden, intermediate]
    up_proj: ttnn.Tensor  # w3, [1, num_experts, hidden, intermediate]
    down_proj: ttnn.Tensor  # w2, [1, num_experts, intermediate, hidden]
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

    if state_dict:
        # MiniMax-M2 experts are a ModuleList of MiniMaxM2MLP; per-expert keys are
        # "{e}.w1.weight" (gate), "{e}.w3.weight" (up), "{e}.w2.weight" (down).
        # nn.Linear weight is [out, in]; transpose to [in, out] for x @ W, then stack
        # into [1, num_experts, in, out].
        E, H, I = config.num_experts, config.hidden_size, config.intermediate_size
        gate_proj = torch.stack([state_dict[f"{e}.w1.weight"].t() for e in range(E)], dim=0).reshape(1, E, H, I)
        up_proj = torch.stack([state_dict[f"{e}.w3.weight"].t() for e in range(E)], dim=0).reshape(1, E, H, I)
    else:
        gate_proj = None
        up_proj = None
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
    # Load down projection (w2): [out=hidden, in=intermediate] -> transpose to
    # [intermediate, hidden], stack -> [1, num_experts, intermediate, hidden].
    if state_dict:
        E, H, I = config.num_experts, config.hidden_size, config.intermediate_size
        down_proj = torch.stack([state_dict[f"{e}.w2.weight"].t() for e in range(E)], dim=0).reshape(1, E, I, H)
    else:
        down_proj = None

    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return ExpertWeights(
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        intermediate_size_per_device=intermediate_size_per_device,
    )
