# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading and management for throughput-optimized MoE experts.

This module handles loading and sharding expert weights across devices for
the all_to_all-based throughput experts implementation.
"""

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

    Attributes:
        w1: Gate projection weights [num_experts_per_device, hidden_size, intermediate_size]
        w2: Down projection weights [num_experts_per_device, intermediate_size, hidden_size]
        w3: Up projection weights [num_experts_per_device, hidden_size, intermediate_size]
        w1_bias: Gate projection bias [num_experts_per_device, 1, intermediate_size]
        w2_bias: Down projection bias [num_experts_per_device, 1, hidden_size]
        w3_bias: Up projection bias [num_experts_per_device, 1, intermediate_size]
    """

    w1: ttnn.Tensor  # Gate projection
    w2: ttnn.Tensor  # Down projection
    w3: ttnn.Tensor  # Up projection
    w1_bias: ttnn.Tensor  # Gate projection bias
    w2_bias: ttnn.Tensor  # Down projection bias
    w3_bias: ttnn.Tensor  # Up projection bias


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

    w2 = state_dict["down_proj"].reshape(1, num_experts, intermediate_size, hidden_size)
    w2_bias = state_dict.get("down_proj_bias", torch.zeros(1, num_experts, 1, hidden_size)).reshape(
        1, num_experts, 1, hidden_size
    )

    # Transpose for matmul: [1, num_experts, out_features, in_features] -> [1, num_experts, in_features, out_features]
    # w1 = w1.transpose(-1, -2)
    # w2 = w2.transpose(-1, -2)
    # w3 = w3.transpose(-1, -2)

    # Reshape bias for proper broadcasting with sparse_matmul output
    # sparse_matmul output shape: [1, num_sparse_blocks, 1, num_local_experts, local_batch, dim]
    # We need bias shape: [1, 1, 1, num_local_experts, 1, dim] after sharding
    # So before sharding: [1, 1, 1, num_experts, 1, dim]
    w1_bias = w1_bias.reshape(1, 1, 1, num_experts, 1, intermediate_size)
    w3_bias = w3_bias.reshape(1, 1, 1, num_experts, 1, intermediate_size)
    # w2_bias = w2_bias.reshape(1, 1, 1, num_experts, 1, hidden_size)

    # Load and shard weights
    w1_tt = _shard_experts_by_device(
        w1,
        num_devices,
        mesh_device,
        weight_dtype,
        get_cache_file_name(tensor_cache_path, "throughput_w1"),
    )

    w2_tt = _shard_experts_by_device(
        w2,
        num_devices,
        mesh_device,
        weight_dtype,
        get_cache_file_name(tensor_cache_path, "throughput_w2"),
    )

    w3_tt = _shard_experts_by_device(
        w3,
        num_devices,
        mesh_device,
        weight_dtype,
        get_cache_file_name(tensor_cache_path, "throughput_w3"),
    )

    # Load and shard bias (use bfloat16 for better precision)
    # Shard on dim=3 (expert dimension for reshaped bias)
    # After sharding, shape becomes [1, 1, 1, num_experts_per_device, 1, dim]
    w1_bias_tt = _shard_experts_by_device(
        w1_bias,
        num_devices,
        mesh_device,
        ttnn.bfloat16,
        get_cache_file_name(tensor_cache_path, "throughput_w1_bias"),
        shard_dim=-3,  # Expert dimension after reshape
    )

    w2_bias_tt = _shard_experts_by_device(
        w2_bias,
        num_devices,
        mesh_device,
        ttnn.bfloat16,
        get_cache_file_name(tensor_cache_path, "throughput_w2_bias"),
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
        w1=w1_tt, w2=w2_tt, w3=w3_tt, w1_bias=w1_bias_tt, w2_bias=w2_bias_tt, w3_bias=w3_bias_tt
    )


def load_throughput_expert_weights_from_hf(
    mesh_device,
    config: ThroughputExpertConfig,
    state_dict: dict,
    weight_dtype: ttnn.DataType = ttnn.bfloat4_b,
    tensor_cache_path: str = None,
) -> ThroughputExpertWeights:
    """Load expert weights from HuggingFace format.

    HuggingFace MoE models typically store weights as:
    - experts.{i}.gate_proj.weight: [intermediate_size, hidden_size]
    - experts.{i}.up_proj.weight: [intermediate_size, hidden_size]
    - experts.{i}.down_proj.weight: [hidden_size, intermediate_size]

    Args:
        mesh_device: TTNN mesh device
        config: Throughput expert configuration
        state_dict: HuggingFace state dict with individual expert weights
        weight_dtype: Data type for weights
        tensor_cache_path: Optional cache path

    Returns:
        ThroughputExpertWeights with sharded tensors
    """
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    # Stack individual expert weights
    w1_list = []
    w2_list = []
    w3_list = []

    for i in range(num_experts):
        prefix = f"experts.{i}."

        # gate_proj: [intermediate_size, hidden_size] -> transpose for matmul
        gate = state_dict.get(f"{prefix}gate_proj.weight")
        if gate is not None:
            w1_list.append(gate.t())  # [hidden_size, intermediate_size]

        # down_proj: [hidden_size, intermediate_size] -> transpose for matmul
        down = state_dict.get(f"{prefix}down_proj.weight")
        if down is not None:
            w2_list.append(down.t())  # [intermediate_size, hidden_size]

        # up_proj: [intermediate_size, hidden_size] -> transpose for matmul
        up = state_dict.get(f"{prefix}up_proj.weight")
        if up is not None:
            w3_list.append(up.t())  # [hidden_size, intermediate_size]

    # Stack into [1, num_experts, in_features, out_features]
    w1 = torch.stack(w1_list, dim=0).unsqueeze(0)
    w2 = torch.stack(w2_list, dim=0).unsqueeze(0)
    w3 = torch.stack(w3_list, dim=0).unsqueeze(0)

    # Load and shard
    num_devices = config.num_devices

    w1_tt = _shard_experts_by_device(
        w1,
        num_devices,
        mesh_device,
        weight_dtype,
        get_cache_file_name(tensor_cache_path, "throughput_w1_hf"),
    )

    w2_tt = _shard_experts_by_device(
        w2,
        num_devices,
        mesh_device,
        weight_dtype,
        get_cache_file_name(tensor_cache_path, "throughput_w2_hf"),
    )

    w3_tt = _shard_experts_by_device(
        w3,
        num_devices,
        mesh_device,
        weight_dtype,
        get_cache_file_name(tensor_cache_path, "throughput_w3_hf"),
    )

    return ThroughputExpertWeights(w1=w1_tt, w2=w2_tt, w3=w3_tt)
