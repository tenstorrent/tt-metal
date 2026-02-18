# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading and management for throughput-optimized MoE experts.

This module handles loading and sharding expert weights across devices for
the all_to_all-based throughput experts implementation.
"""

import math
from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .config import ThroughputExpertConfig


@dataclass(frozen=True)
class ThroughputExpertWeights:
    """Container for throughput expert weight tensors.

    Weights are sharded across devices such that each device holds
    num_experts_per_device complete expert weight sets.

    When use_fused_gate_up=True, only w2, w2_bias, w1_w3_fused, and w1_w3_bias_fused are loaded.
    When use_fused_gate_up=False, only w1, w2, w3, w1_bias, w2_bias, w3_bias are loaded.

    Attributes:
        w2: Down projection weights [num_experts_per_device, intermediate_size, hidden_size]
        w2_bias: Down projection bias [num_experts_per_device, 1, hidden_size]
        w1: Optional gate projection weights [num_experts_per_device, hidden_size, intermediate_size]
        w3: Optional up projection weights [num_experts_per_device, hidden_size, intermediate_size]
        w1_bias: Optional gate projection bias [num_experts_per_device, 1, intermediate_size]
        w3_bias: Optional up projection bias [num_experts_per_device, 1, intermediate_size]
        w1_w3_fused: Optional fused gate+up weights [num_experts_per_device, hidden_size, 2*intermediate_size]
        w1_w3_bias_fused: Optional fused gate+up bias [num_experts_per_device, 1, 2*intermediate_size]
    """

    # Required weights (always loaded)
    w2: ttnn.Tensor  # Down projection
    w2_bias: ttnn.Tensor  # Down projection bias

    # Unfused weights (loaded only when use_fused_gate_up=False)
    w1: ttnn.Tensor = None  # Gate projection
    w3: ttnn.Tensor = None  # Up projection
    w1_bias: ttnn.Tensor = None  # Gate projection bias
    w3_bias: ttnn.Tensor = None  # Up projection bias

    # Fused weights (loaded only when use_fused_gate_up=True)
    w1_w3_fused: ttnn.Tensor = None  # Fused gate+up projection
    w1_w3_bias_fused: ttnn.Tensor = None  # Fused gate+up bias


def _shard_experts_by_device(
    weights: torch.Tensor,
    num_devices: int,
    mesh_device,
    dtype: ttnn.DataType,
    cache_file_name: str = None,
    shard_dim: int = 1,
) -> ttnn.Tensor:
    """Shard expert weights across devices.

    Each device gets a subset of experts (num_experts / num_devices experts per device).

    Args:
        weights: Full weight tensor (expert dimension will be sharded)
        num_devices: Number of devices to shard across
        mesh_device: TTNN mesh device
        dtype: Weight data type
        cache_file_name: Optional cache file path
        shard_dim: Dimension to shard across (default: 1 for weights, 3 for bias)

    Returns:
        Sharded ttnn.Tensor where each device has its local experts
    """
    # Shard along the expert dimension
    # Each device gets consecutive experts
    return ttnn.as_tensor(
        weights,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ShardTensorToMesh(
            mesh_device,
            dim=shard_dim,  # Shard expert dim across all devices
        ),
        cache_file_name=cache_file_name,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def load_throughput_expert_weights(
    mesh_device,
    config: ThroughputExpertConfig,
    state_dict: dict,
    weight_dtype: ttnn.DataType = ttnn.bfloat4_b,
    tensor_cache_path: str = None,
) -> ThroughputExpertWeights:
    """Load and shard expert weights for throughput mode.

    Args:
        mesh_device: TTNN mesh device
        config: Throughput expert configuration
        state_dict: Dictionary containing expert weights
        weight_dtype: Data type for weights (default: bfloat4_b)
        tensor_cache_path: Optional path for weight caching

    Returns:
        ThroughputExpertWeights with sharded tensors
    """
    num_experts = config.num_experts
    num_devices = config.num_devices
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    # Extract weights from state dict
    # Assume state_dict has keys like "gate_up_proj" (fused) and "down_proj"
    # or individual "gate_proj", "up_proj", "down_proj"

    if "gate_up_proj" in state_dict:
        # Fused gate/up projection - unfuse
        gate_up = state_dict["gate_up_proj"]
        # gate_up shape: [num_experts, hidden_size, 2 * intermediate_size] (interleaved)
        w1 = gate_up[..., ::2].reshape(1, num_experts, hidden_size, intermediate_size)
        w3 = gate_up[..., 1::2].reshape(1, num_experts, hidden_size, intermediate_size)

        # Unfuse bias if present
        if "gate_up_proj_bias" in state_dict:
            gate_up_bias = state_dict["gate_up_proj_bias"]
            w1_bias = gate_up_bias[..., ::2].reshape(1, num_experts, 1, intermediate_size)
            w3_bias = gate_up_bias[..., 1::2].reshape(1, num_experts, 1, intermediate_size)
        else:
            w1_bias = torch.zeros(1, num_experts, 1, intermediate_size)
            w3_bias = torch.zeros(1, num_experts, 1, intermediate_size)
    else:
        # Separate gate and up projections
        w1 = state_dict["gate_proj"].reshape(1, num_experts, hidden_size, intermediate_size)
        w3 = state_dict["up_proj"].reshape(1, num_experts, hidden_size, intermediate_size)
        w1_bias = state_dict.get("gate_proj_bias", torch.zeros(1, num_experts, 1, intermediate_size)).reshape(
            1, num_experts, 1, intermediate_size
        )
        w3_bias = state_dict.get("up_proj_bias", torch.zeros(1, num_experts, 1, intermediate_size)).reshape(
            1, num_experts, 1, intermediate_size
        )

    # Load down projection weights (always needed)
    w2 = state_dict["down_proj"].reshape(1, num_experts, intermediate_size, hidden_size)
    w2_bias = state_dict.get("down_proj_bias", torch.zeros(1, num_experts, 1, hidden_size)).reshape(
        1, num_experts, 1, hidden_size
    )
    if config.pad_w2:
        # Pad w2 output dimension for tile alignment in CCL operations.
        # Without padding, local_hidden = hidden_size / TP may not be tile-aligned (e.g., 2880/8 = 360),
        # causing CCL to do expensive Untilize->Pad->Tilize cycles internally.
        hidden_size = config.hidden_size
        scattered_local_hidden = hidden_size // mesh_device.shape[1]
        padded_scattered_local_hidden = ((scattered_local_hidden + 31) // 32) * 32  # Round up to tile boundary
        w2_pad_size = (padded_scattered_local_hidden * mesh_device.shape[1]) - hidden_size

        if w2_pad_size > 0 and mesh_device.shape[1] > 1:
            # Pad the output dimension of w2 weight: [input_dim, hidden_size] -> [input_dim, padded_hidden]
            w2 = torch.nn.functional.pad(w2, (0, w2_pad_size), "constant", value=0.0)
            # Pad bias similarly
            w2_bias = torch.nn.functional.pad(w2_bias, (0, w2_pad_size), "constant", value=0.0)

        # Use unique cache key when padding is applied
        w2_cache_suffix = f"_padded{w2_pad_size}" if w2_pad_size > 0 and mesh_device.shape[1] > 1 else ""
    else:
        w2_cache_suffix = ""

    w2_tt = _shard_experts_by_device(
        w2,
        num_devices,
        mesh_device,
        weight_dtype,
        get_cache_file_name(tensor_cache_path, f"throughput_w2{w2_cache_suffix}"),
    )

    w2_bias_tt = _shard_experts_by_device(
        w2_bias,
        num_devices,
        mesh_device,
        ttnn.bfloat16,
        get_cache_file_name(tensor_cache_path, "throughput_w2_bias"),
        shard_dim=-3,  # Expert dimension after reshape
    )

    # Load either fused or unfused weights based on config
    w1_tt = None
    w3_tt = None
    w1_bias_tt = None
    w3_bias_tt = None
    w1_w3_fused_tt = None
    w1_w3_bias_fused_tt = None

    if config.use_fused_gate_up:
        # ======================================================================
        # FUSED MODE: Load only fused weights to save memory
        # ======================================================================
        # Fuse w1 and w3 along the output dimension
        # w1: [1, num_experts, hidden_size, intermediate_size]
        # w3: [1, num_experts, hidden_size, intermediate_size]
        # -> w1_w3_fused: [1, num_experts, hidden_size, 2*intermediate_size]
        weights_cache_suffix = "throughput_w1_w3_fused"
        bias_cache_suffix = "throughput_w1_w3_bias_fused"
        w1_w3_fused = torch.cat([w1, w3], dim=-1)

        # Fuse biases
        # w1_bias: [1, num_experts, 1, intermediate_size]
        # w3_bias: [1, num_experts, 1, intermediate_size]
        # -> w1_w3_bias_fused: [1, num_experts, 1, 2*intermediate_size]
        w1_w3_bias_fused = torch.cat([w1_bias, w3_bias], dim=-1)

        if config.pad_w1_w3:
            ideal_core_grid_size = 64
            fused_hidden_size = 2 * intermediate_size
            fused_hidden_size_t = fused_hidden_size / ttnn.TILE_SIZE
            required_fused_hidden_size = (
                math.ceil(fused_hidden_size_t / ideal_core_grid_size) * ideal_core_grid_size * ttnn.TILE_SIZE
            )
            pad_size = int(required_fused_hidden_size - fused_hidden_size)
            w1_w3_fused = torch.nn.functional.pad(w1_w3_fused, (0, pad_size))
            w1_w3_bias_fused = torch.nn.functional.pad(w1_w3_bias_fused, (0, pad_size))
            weights_cache_suffix += f"_padded{pad_size}"
            bias_cache_suffix += f"_padded{pad_size}"

        w1_w3_fused_tt = _shard_experts_by_device(
            w1_w3_fused,
            num_devices,
            mesh_device,
            weight_dtype,
            get_cache_file_name(tensor_cache_path, weights_cache_suffix),
        )

        w1_w3_bias_fused_tt = _shard_experts_by_device(
            w1_w3_bias_fused,
            num_devices,
            mesh_device,
            ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, bias_cache_suffix),
            shard_dim=-3,  # Expert dimension after reshape
        )
    else:
        # ======================================================================
        # UNFUSED MODE: Load individual gate and up projection weights
        # ======================================================================
        w1_tt = _shard_experts_by_device(
            w1,
            num_devices,
            mesh_device,
            weight_dtype,
            get_cache_file_name(tensor_cache_path, "throughput_w1"),
        )

        w3_tt = _shard_experts_by_device(
            w3,
            num_devices,
            mesh_device,
            weight_dtype,
            get_cache_file_name(tensor_cache_path, "throughput_w3"),
        )

        # Load and shard bias (use bfloat16 for better precision)
        w1_bias_tt = _shard_experts_by_device(
            w1_bias,
            num_devices,
            mesh_device,
            ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, "throughput_w1_bias"),
            shard_dim=-3,  # Expert dimension after reshape
        )

        w3_bias_tt = _shard_experts_by_device(
            w3_bias,
            num_devices,
            mesh_device,
            ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, "throughput_w3_bias"),
            shard_dim=-3,  # Expert dimension after reshape
        )

    return ThroughputExpertWeights(
        w2=w2_tt,
        w2_bias=w2_bias_tt,
        w1=w1_tt,
        w3=w3_tt,
        w1_bias=w1_bias_tt,
        w3_bias=w3_bias_tt,
        w1_w3_fused=w1_w3_fused_tt,
        w1_w3_bias_fused=w1_w3_bias_fused_tt,
    )
