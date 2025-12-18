# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for throughput-optimized MoE experts with all_to_all operations.

This module implements expert distribution across 32 Galaxy devices (4 experts per device)
using all_to_all_dispatch and all_to_all_combine for dynamic batching based on expert routing.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

import ttnn


@dataclass
class ThroughputExpertConfig:
    """Configuration for throughput-optimized experts with all_to_all operations.

    This config extends the base expert config to support distributing experts
    across multiple devices and using all_to_all operations for dynamic batching.

    Attributes:
        intermediate_size: MLP intermediate dimension
        num_experts: Total number of experts (e.g., 128 for 32 devices × 4 experts)
        hidden_size: Model hidden dimension
        num_experts_per_tok: Number of experts activated per token
        num_devices: Total number of devices in the mesh (e.g., 32 for Galaxy)
        num_experts_per_device: Experts per device (num_experts / num_devices)
        sparsity_block_size: Block size for sparse operations (default: 32)
        swiglu_limit: Clamp limit for SwiGLU activation
        alpha: SwiGLU alpha parameter
    """

    intermediate_size: int
    num_experts: int
    hidden_size: int
    num_experts_per_tok: int
    num_devices: int
    sparsity_block_size: int = 32
    swiglu_limit: float = 7.0
    alpha: float = 1.702

    def __post_init__(self):
        """Validate and compute derived values."""
        if self.num_experts % self.num_devices != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by " f"num_devices ({self.num_devices})"
            )
        self.num_experts_per_device = self.num_experts // self.num_devices

    @classmethod
    def from_hf_config(cls, hf_config, mesh_device) -> "ThroughputExpertConfig":
        """Create config from HuggingFace model config.

        Args:
            hf_config: HuggingFace PretrainedConfig with MoE settings
            mesh_device: TTNN mesh device

        Returns:
            ThroughputExpertConfig instance
        """
        num_devices = mesh_device.get_num_devices()
        return cls(
            intermediate_size=hf_config.intermediate_size,
            num_experts=hf_config.num_local_experts,
            hidden_size=hf_config.hidden_size,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            num_devices=num_devices,
            swiglu_limit=getattr(hf_config, "swiglu_limit", 7.0),
            alpha=getattr(hf_config, "alpha", 1.702),
        )


@dataclass
class AllToAllDispatchConfig:
    """Configuration for ttnn.all_to_all_dispatch operation.

    This operation dispatches tokens to experts across devices based on routing.
    """

    cluster_axis: int = 0
    memory_config: ttnn.MemoryConfig = field(default_factory=lambda: ttnn.L1_MEMORY_CONFIG)
    num_links: int = 1
    topology: ttnn.Topology = field(default_factory=lambda: ttnn.Topology.Linear)
    subdevice_id: Optional[int] = None
    output_concat_dim: Optional[int] = 1  # 1 for decode 2 for prefill

    def as_dict(self):
        """Convert to kwargs dict for ttnn.all_to_all_dispatch."""
        result = {
            "cluster_axis": self.cluster_axis,
            "memory_config": self.memory_config,
            "num_links": self.num_links,
            "topology": self.topology,
            "output_concat_dim": self.output_concat_dim,
        }
        if self.subdevice_id is not None:
            result["subdevice_id"] = self.subdevice_id
        return result


@dataclass
class AllToAllCombineConfig:
    """Configuration for ttnn.all_to_all_combine operation.

    This operation combines expert outputs back to their original token positions.
    """

    cluster_axis: int = 0
    memory_config: ttnn.MemoryConfig = field(default_factory=lambda: ttnn.L1_MEMORY_CONFIG)
    num_links: int = 1
    topology: ttnn.Topology = field(default_factory=lambda: ttnn.Topology.Linear)

    def as_dict(self):
        """Convert to kwargs dict for ttnn.all_to_all_combine."""
        return {
            "cluster_axis": self.cluster_axis,
            "memory_config": self.memory_config,
            "num_links": self.num_links,
            "topology": self.topology,
        }


@dataclass
class ThroughputProgramConfig:
    """Program configuration for throughput-optimized expert computations.

    Provides matmul program configs for the MLP operations within each expert.
    """

    # Core grid sizes for sparse matmul
    gate_up_cores: tuple[int, int] = (5, 9)
    down_cores: tuple[int, int] = (5, 9)

    # Sparse matmul parameters
    # in0_block_w: int = 2
    in0_block_w: int = 10
    ## can be estimated by k // 32 (2880 / 32 = 90) therefore factors of 90 (30, 15, 10, 9 etc.)
    out_subblock_h: int = 1
    out_subblock_w: int = 1
    per_core_M: int = 1

    def __post_init__(self):
        """Validate configuration."""
        self._validate_cores("gate_up_cores", self.gate_up_cores)
        self._validate_cores("down_cores", self.down_cores)

    def _validate_cores(self, name: str, cores: tuple[int, int]):
        """Validate core grid dimensions."""
        if not isinstance(cores, tuple) or len(cores) != 2:
            raise ValueError(f"{name} must be a tuple of (x, y), got {cores}")
        core_x, core_y = cores
        if core_x <= 0 or core_y <= 0:
            raise ValueError(f"{name} must have positive dimensions, got {cores}")

    def get_gate_up_config(self, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for gate/up projections.

        Args:
            n: Output feature dimension

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig
        """
        core_x, core_y = self.gate_up_cores
        n_tiles = math.ceil(n / ttnn.TILE_SIZE)
        per_core_N = n_tiles // (core_x * core_y)
        # per_core_N = 2

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=self.in0_block_w,
            out_subblock_h=self.out_subblock_h,
            out_subblock_w=self.out_subblock_w,
            per_core_M=self.per_core_M,
            per_core_N=max(1, per_core_N),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_down_config(self, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for down projection.

        Args:
            n: Output feature dimension

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig
        """
        core_x, core_y = self.down_cores
        n_tiles = math.ceil(n / ttnn.TILE_SIZE)
        per_core_N = n_tiles // (core_x * core_y)
        per_core_N = 2

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=self.in0_block_w,
            out_subblock_h=self.out_subblock_h,
            out_subblock_w=self.out_subblock_w,
            per_core_M=self.per_core_M,
            per_core_N=max(1, per_core_N),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )


def create_expert_mapping_tensors(
    num_devices: int,
    num_experts_per_device: int,
    mesh_device,
    cluster_axis: Optional[int] = None,
    mesh_shape: tuple[int, int] = None,
) -> ttnn.Tensor:
    """Create expert-to-device mapping tensors for all_to_all operations.

    Args:
        num_devices: Total number of devices
        num_experts_per_device: Experts per device
        mesh_device: TTNN mesh device

    Returns:
        Mapping tensor [1, 1, num_experts, num_devices]
    """
    # Create identity matrix showing which device owns which expert
    # Shape: [num_experts, num_devices] where mapping[e, d] = 1 if expert e is on device d
    mapping = (
        torch.eye(num_devices, dtype=torch.int32)
        .repeat_interleave(num_experts_per_device, dim=0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    # if cluster_axis is not None:
    #     if cluster_axis == 0:
    #         mapping = mapping.repeat(1, 1, mesh_shape[1], 1)
    #     elif cluster_axis == 1:
    #         mapping = mapping.repeat(1, 1, 1, mesh_shape[0])
    #     else:
    #         raise ValueError(f"Invalid cluster axis: {cluster_axis}")

    return ttnn.from_torch(
        mapping,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def create_remap_topk_mask(
    num_dispatch_device_rows: int,
    num_experts: int,
    mesh_device,
) -> ttnn.Tensor:
    """Create mask for remapping top-k expert indices.

    Args:
        num_dispatch_device_rows: Number of device rows for dispatch
        num_experts: Total number of experts
        mesh_device: TTNN mesh device

    Returns:
        Remap mask tensor [1, num_dispatch_device_rows, 1, num_experts]
    """
    mask = torch.ones(
        (1, num_dispatch_device_rows, 1, num_experts),
        dtype=torch.bfloat16,
    )

    return ttnn.from_torch(
        mask,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
