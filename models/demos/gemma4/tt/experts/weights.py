# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading for Gemma4 routed experts.

Unfuses HF gate_up_proj [num_experts, 2*intermediate, hidden] into separate
gate and up projections for sparse_matmul.

No bias in Gemma4 experts (unlike GPT-OSS).
"""

from dataclasses import dataclass

import ttnn
from models.demos.gemma4.utils.general_utils import get_cache_file_name


@dataclass(frozen=True)
class ExpertWeights:
    """Container for expert weight tensors — immutable after creation."""

    gate_proj: ttnn.Tensor  # [1, num_experts, hidden_size, intermediate_size]
    up_proj: ttnn.Tensor  # [1, num_experts, hidden_size, intermediate_size]
    down_proj: ttnn.Tensor  # [1, num_experts, intermediate_size, hidden_size]
    intermediate_size_per_device: int


def load_expert_weights(
    mesh_device,
    config,
    state_dict,
    mesh_config=None,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> ExpertWeights:
    """
    Load expert weights to device for sparse_matmul.

    Unfuses HF gate_up_proj [E, 2*I, H] into gate [1, E, H, I] and up [1, E, H, I].
    Transposes down_proj [E, H, I] into [1, E, I, H].
    """
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    intermediate_size = config.moe_intermediate_size

    if state_dict and "gate_up_proj" in state_dict:
        fused = state_dict["gate_up_proj"]  # [E, 2*I, H]

        # Gemma4: first half is gate, second half is up (contiguous, NOT interleaved)
        gate_proj = fused[:, :intermediate_size, :]  # [E, I, H]
        up_proj = fused[:, intermediate_size:, :]  # [E, I, H]

        # Transpose for matmul: [E, I, H] -> [1, E, H, I]
        gate_proj = gate_proj.transpose(-2, -1).unsqueeze(0)
        up_proj = up_proj.transpose(-2, -1).unsqueeze(0)

        # Down: [E, H, I] -> transpose -> [1, E, I, H]
        down_proj = state_dict["down_proj"].transpose(-2, -1).unsqueeze(0)
    else:
        gate_proj = None
        up_proj = None
        down_proj = None

    # For now, no TP/EP sharding on expert weights (replicate to all devices)
    is_mesh = hasattr(mesh_device, "shape")
    replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    gate_proj_tt = ttnn.as_tensor(
        gate_proj,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    up_proj_tt = ttnn.as_tensor(
        up_proj,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return ExpertWeights(
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        intermediate_size_per_device=intermediate_size,
    )
