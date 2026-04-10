# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading for Gemma4 routed experts.

Unfuses HF gate_up_proj [num_experts, 2*intermediate, hidden] into separate
gate and up projections for sparse_matmul.

No bias in Gemma4 experts (unlike GPT-OSS).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4.utils.general_utils import get_cache_file_name

TILE_SIZE = 32


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

    TP sharding (matching SharedMLP pattern):
    - gate/up: column-parallel (shard output/intermediate dim across TP devices)
    - down: row-parallel (shard input/intermediate dim across TP devices)
    - All-reduce after down_proj recombines partial results
    """
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    intermediate_size = config.moe_intermediate_size
    tp = mesh_config.tp if mesh_config else 1

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

        # Pad intermediate dim to tile-aligned per-device size for sparse_matmul.
        # E.g. 704/8=88 is not tile-aligned; pad to 768 so each device gets 96 (=3*32).
        if tp > 1:
            per_device = intermediate_size // tp
            padded_per_device = ((per_device + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
            pad_amount = padded_per_device * tp - intermediate_size
            if pad_amount > 0:
                # gate/up: [1, E, H, I] → pad last dim
                gate_proj = torch.nn.functional.pad(gate_proj, (0, pad_amount))
                up_proj = torch.nn.functional.pad(up_proj, (0, pad_amount))
                # down: [1, E, I, H] → pad dim -2
                down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_amount))
    else:
        gate_proj = None
        up_proj = None
        down_proj = None

    # Compute per-device intermediate size (tile-aligned when TP > 1)
    if tp > 1:
        per_device_intermediate = ((intermediate_size // tp + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    else:
        per_device_intermediate = intermediate_size

    # TP sharding: column-parallel for gate/up, row-parallel for down
    is_mesh = hasattr(mesh_device, "shape")
    tp_suffix = f"_tp{tp}" if tp > 1 else ""

    if tp > 1 and mesh_config is not None:
        col_mapper = mesh_config.column_parallel(mesh_device)
        row_mapper = mesh_config.row_parallel(mesh_device)
    else:
        col_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        row_mapper = col_mapper

    gate_proj_tt = ttnn.as_tensor(
        gate_proj,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj{tp_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    up_proj_tt = ttnn.as_tensor(
        up_proj,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj{tp_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=row_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj{tp_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return ExpertWeights(
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        intermediate_size_per_device=per_device_intermediate,
    )
