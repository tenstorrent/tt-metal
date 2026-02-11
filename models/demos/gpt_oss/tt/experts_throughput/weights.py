# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading and management for throughput-optimized MoE experts.

This module handles loading and sharding expert weights across devices for
the all_to_all-based throughput experts implementation.

Supports two storage modes:
- **DRAM sharded** (default): Each expert's weights are stored as individual 2D tensors
  with WIDTH_SHARDED DRAM memory configs. This maximizes DRAM read bandwidth during
  matmuls (~2-3x speedup on bandwidth-bound decode matmuls).
- **Legacy batched**: All experts are stored in a single 4D tensor with interleaved
  DRAM storage, used with batched matmuls.
"""

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .config import ThroughputExpertConfig, ThroughputProgramConfig


@dataclass(frozen=True)
class ThroughputExpertWeights:
    """Container for throughput expert weight tensors.

    Weights are sharded across devices such that each device holds
    num_experts_per_device complete expert weight sets.

    Two storage modes:
    - DRAM sharded: Weights are lists of per-expert 2D DRAM width-sharded tensors.
    - Legacy batched: Weights are single 4D tensors with interleaved DRAM.

    When use_fused_gate_up=True, only w2, w2_bias, w1_w3_fused, and w1_w3_bias_fused are loaded.
    When use_fused_gate_up=False, only w1, w2, w3, w1_bias, w2_bias, w3_bias are loaded.

    Attributes:
        w2: Down projection weights (4D tensor or list of 2D tensors)
        w2_bias: Down projection bias (4D tensor or list of 2D tensors)
        w1: Optional gate projection weights
        w3: Optional up projection weights
        w1_bias: Optional gate projection bias
        w3_bias: Optional up projection bias
        w1_w3_fused: Optional fused gate+up weights
        w1_w3_bias_fused: Optional fused gate+up bias
        dram_sharded: Whether weights are in DRAM sharded mode
    """

    # Required weights (always loaded)
    w2: ttnn.Tensor | list[ttnn.Tensor]  # Down projection
    w2_bias: ttnn.Tensor | list[ttnn.Tensor]  # Down projection bias

    # Unfused weights (loaded only when use_fused_gate_up=False)
    w1: ttnn.Tensor | list[ttnn.Tensor] | None = None  # Gate projection
    w3: ttnn.Tensor | list[ttnn.Tensor] | None = None  # Up projection
    w1_bias: ttnn.Tensor | list[ttnn.Tensor] | None = None  # Gate projection bias
    w3_bias: ttnn.Tensor | list[ttnn.Tensor] | None = None  # Up projection bias

    # Fused weights (loaded only when use_fused_gate_up=True)
    w1_w3_fused: ttnn.Tensor | list[ttnn.Tensor] | None = None  # Fused gate+up projection
    w1_w3_bias_fused: ttnn.Tensor | list[ttnn.Tensor] | None = None  # Fused gate+up bias

    # Mode flag
    dram_sharded: bool = False


def _shard_experts_by_device(
    weights: torch.Tensor,
    num_devices: int,
    mesh_device,
    dtype: ttnn.DataType,
    cache_file_name: str = None,
    shard_dim: int = 1,
) -> ttnn.Tensor:
    """Shard expert weights across devices (legacy batched mode).

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


def _shard_experts_by_device_dram_sharded(
    weights: torch.Tensor,
    num_devices: int,
    num_experts_per_device: int,
    mesh_device,
    dtype: ttnn.DataType,
    dram_weight_grid: ttnn.CoreRangeSet,
    dram_cores: int,
    cache_file_name: str = None,
) -> list[ttnn.Tensor]:
    """Shard expert weights across devices with DRAM width-sharded storage.

    Each expert's weight is stored as a separate 2D tensor, width-sharded across
    DRAM banks. This maximizes DRAM read bandwidth during matmuls.

    Args:
        weights: Full weight tensor [1, num_experts, K, N]
        num_devices: Number of devices
        num_experts_per_device: Experts per device
        mesh_device: TTNN mesh device
        dtype: Weight data type
        dram_weight_grid: CoreRangeSet spanning all DRAM cores
        dram_cores: Number of DRAM cores
        cache_file_name: Optional base cache file path

    Returns:
        List of num_experts_per_device DRAM width-sharded tensors, each [1, 1, K, N]
    """
    num_experts = weights.shape[1]
    k = weights.shape[2]
    n = weights.shape[3]

    # Create DRAM sharded memory config for this weight shape
    mem_config = ThroughputProgramConfig.create_dram_sharded_mem_config(k, n, dram_weight_grid, dram_cores)

    expert_tensors = []
    for expert_idx in range(num_experts_per_device):
        # Extract this expert's weight across all devices:
        # weights[:, expert_idx::num_experts_per_device, :, :] selects the expert_idx-th
        # expert from each device's group, giving one weight per device
        # But we need each device to get its own local expert, so we shard the expert dim.
        #
        # Select all instances of this local expert index across devices:
        # Device 0 gets expert expert_idx, device 1 gets expert expert_idx + num_experts_per_device, etc.
        expert_indices = list(range(expert_idx, num_experts, num_experts_per_device))
        expert_weight = weights[:, expert_indices, :, :]  # [1, num_devices, K, N]

        expert_cache = (
            get_cache_file_name(cache_file_name, f"_expert{expert_idx}_dram_sharded")
            if cache_file_name
            else None
        )

        expert_tt = ttnn.as_tensor(
            expert_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            cache_file_name=expert_cache,
            memory_config=mem_config,
        )
        expert_tensors.append(expert_tt)

    return expert_tensors


def load_throughput_expert_weights(
    mesh_device,
    config: ThroughputExpertConfig,
    state_dict: dict,
    mesh_config: MeshConfig,
    weight_dtype: ttnn.DataType = ttnn.bfloat4_b,
    tensor_cache_path: str = None,
    use_dram_sharded: bool = True,
) -> ThroughputExpertWeights:
    """Load and shard expert weights for throughput mode.

    Args:
        mesh_device: TTNN mesh device
        config: Throughput expert configuration
        state_dict: Dictionary containing expert weights
        mesh_config: Mesh configuration
        weight_dtype: Data type for weights (default: bfloat4_b)
        tensor_cache_path: Optional path for weight caching
        use_dram_sharded: If True, store weights as per-expert DRAM width-sharded tensors.
            If False, use legacy batched 4D interleaved DRAM tensors.

    Returns:
        ThroughputExpertWeights with sharded tensors
    """
    num_experts = config.num_experts
    num_devices = config.num_devices
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_experts_per_device = config.num_experts_per_device

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

    # ======================================================================
    # DRAM SHARDED MODE: Per-expert 2D tensors with DRAM width-sharded storage
    # ======================================================================
    if use_dram_sharded:
        dram_weight_grid, dram_cores = ThroughputProgramConfig.create_dram_weight_grid(mesh_device)

        w2_tt = _shard_experts_by_device_dram_sharded(
            w2, num_devices, num_experts_per_device, mesh_device, weight_dtype,
            dram_weight_grid, dram_cores,
            get_cache_file_name(tensor_cache_path, "throughput_w2"),
        )

        # Biases: stored as per-expert 2D tensors in interleaved DRAM
        # (biases are small, no benefit from DRAM sharding)
        w2_bias_tt = _shard_experts_by_device_dram_sharded(
            w2_bias, num_devices, num_experts_per_device, mesh_device, ttnn.bfloat16,
            dram_weight_grid, dram_cores,
            get_cache_file_name(tensor_cache_path, "throughput_w2_bias"),
        )

        w1_tt = None
        w3_tt = None
        w1_bias_tt = None
        w3_bias_tt = None
        w1_w3_fused_tt = None
        w1_w3_bias_fused_tt = None

        if config.use_fused_gate_up:
            w1_w3_fused = torch.cat([w1, w3], dim=-1)
            w1_w3_fused_tt = _shard_experts_by_device_dram_sharded(
                w1_w3_fused, num_devices, num_experts_per_device, mesh_device, weight_dtype,
                dram_weight_grid, dram_cores,
                get_cache_file_name(tensor_cache_path, "throughput_w1_w3_fused"),
            )

            w1_w3_bias_fused = torch.cat([w1_bias, w3_bias], dim=-1)
            w1_w3_bias_fused_tt = _shard_experts_by_device_dram_sharded(
                w1_w3_bias_fused, num_devices, num_experts_per_device, mesh_device, ttnn.bfloat16,
                dram_weight_grid, dram_cores,
                get_cache_file_name(tensor_cache_path, "throughput_w1_w3_bias_fused"),
            )
        else:
            w1_tt = _shard_experts_by_device_dram_sharded(
                w1, num_devices, num_experts_per_device, mesh_device, weight_dtype,
                dram_weight_grid, dram_cores,
                get_cache_file_name(tensor_cache_path, "throughput_w1"),
            )
            w3_tt = _shard_experts_by_device_dram_sharded(
                w3, num_devices, num_experts_per_device, mesh_device, weight_dtype,
                dram_weight_grid, dram_cores,
                get_cache_file_name(tensor_cache_path, "throughput_w3"),
            )
            w1_bias_tt = _shard_experts_by_device_dram_sharded(
                w1_bias, num_devices, num_experts_per_device, mesh_device, ttnn.bfloat16,
                dram_weight_grid, dram_cores,
                get_cache_file_name(tensor_cache_path, "throughput_w1_bias"),
            )
            w3_bias_tt = _shard_experts_by_device_dram_sharded(
                w3_bias, num_devices, num_experts_per_device, mesh_device, ttnn.bfloat16,
                dram_weight_grid, dram_cores,
                get_cache_file_name(tensor_cache_path, "throughput_w3_bias"),
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
            dram_sharded=True,
        )

    # ======================================================================
    # LEGACY BATCHED MODE: Single 4D tensors with interleaved DRAM
    # ======================================================================
    pad_w2 = False
    if pad_w2:
        # Pad w2 output dimension for tile alignment in CCL operations.
        scattered_local_hidden = hidden_size // mesh_device.shape[1]
        padded_scattered_local_hidden = ((scattered_local_hidden + 31) // 32) * 32
        w2_pad_size = (padded_scattered_local_hidden * mesh_device.shape[1]) - hidden_size

        if w2_pad_size > 0 and mesh_device.shape[1] > 1:
            w2 = torch.nn.functional.pad(w2, (0, w2_pad_size), "constant", value=0.0)
            w2_bias = torch.nn.functional.pad(w2_bias, (0, w2_pad_size), "constant", value=0.0)

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
        w1_w3_fused = torch.cat([w1, w3], dim=-1)

        w1_w3_fused_tt = _shard_experts_by_device(
            w1_w3_fused,
            num_devices,
            mesh_device,
            weight_dtype,
            get_cache_file_name(tensor_cache_path, "throughput_w1_w3_fused"),
        )

        w1_w3_bias_fused = torch.cat([w1_bias, w3_bias], dim=-1)

        w1_w3_bias_fused_tt = _shard_experts_by_device(
            w1_w3_bias_fused,
            num_devices,
            mesh_device,
            ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, "throughput_w1_w3_bias_fused"),
            shard_dim=-3,
        )
    else:
        w1_tt = _shard_experts_by_device(
            w1, num_devices, mesh_device, weight_dtype,
            get_cache_file_name(tensor_cache_path, "throughput_w1"),
        )
        w3_tt = _shard_experts_by_device(
            w3, num_devices, mesh_device, weight_dtype,
            get_cache_file_name(tensor_cache_path, "throughput_w3"),
        )
        w1_bias_tt = _shard_experts_by_device(
            w1_bias, num_devices, mesh_device, ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, "throughput_w1_bias"),
            shard_dim=-3,
        )
        w3_bias_tt = _shard_experts_by_device(
            w3_bias, num_devices, mesh_device, ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, "throughput_w3_bias"),
            shard_dim=-3,
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
        dram_sharded=False,
    )
