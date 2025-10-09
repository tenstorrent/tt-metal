# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
All-Gather collective operation.

This module provides all-gather functionality for distributed tensor operations,
collecting tensors from all devices and concatenating them.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import ttnn

from .manager import CCLManager


@dataclass
class AllGatherSpec:
    """
    Mathematical specification for all-gather operation.

    All-gather collects tensors from all devices and concatenates them along
    the specified dimension.
    """

    mesh_shape: Tuple[int, int]  # (rows, columns) of device mesh
    cluster_axis: Optional[int] = None  # 0 for row-wise, 1 for column-wise, None for all
    gather_dim: int = 3  # Dimension along which to gather

    def validate(self):
        """Validate spec constraints."""
        assert len(self.mesh_shape) == 2, "mesh_shape must be (rows, columns)"
        assert self.mesh_shape[0] > 0 and self.mesh_shape[1] > 0, "mesh dimensions must be positive"
        if self.cluster_axis is not None:
            assert self.cluster_axis in [0, 1], "cluster_axis must be 0, 1, or None"
        assert 0 <= self.gather_dim <= 3, "gather_dim must be between 0 and 3"


@dataclass
class AllGatherImplConfig:
    """
    TTNN-specific implementation configuration for all-gather.

    Contains device-specific optimizations and performance tuning parameters
    for the all-gather operation.
    """

    # Communication links configuration
    num_all_gather_links: int = 2

    # Topology configuration
    topology: ttnn.Topology = ttnn.Topology.Linear

    # Memory configuration
    memory_config: Optional[ttnn.MemoryConfig] = None

    # Data type for communication
    dtype: ttnn.DataType = ttnn.bfloat16

    # Sharding configuration
    sharded: bool = False

    # Performance tuning parameters
    chunks_per_sync: int = 10
    num_workers_per_link: int = 2
    num_buffers_per_channel: int = 2


def get_default_impl_config(
    spec: AllGatherSpec, device: str, mode: Literal["prefill", "decode"] = "prefill", strategy: str = "default"
) -> AllGatherImplConfig:
    """
    Return default implementation configuration for all-gather.

    Args:
        spec: All-gather specification
        device: Target device (e.g., "N150", "N300", "T3000", "TG")
        mode: Execution mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Default AllGatherImplConfig for the specified device and mode
    """
    if device.startswith("N150"):
        # Single device, minimal configuration
        return AllGatherImplConfig(
            num_all_gather_links=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    elif device.startswith("N300"):
        # Two devices, optimize for bandwidth
        return AllGatherImplConfig(
            num_all_gather_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    elif device.startswith("T3000") or device == "T3K":
        # 8 devices, higher parallelism
        return AllGatherImplConfig(
            num_all_gather_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=20,
        )
    elif device == "TG":
        # Galaxy configuration with mesh topology
        return AllGatherImplConfig(
            num_all_gather_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=10,
        )
    else:
        # Default conservative configuration
        return AllGatherImplConfig()


def all_gather_forward(
    input_tensor: ttnn.Tensor,
    mesh_device: ttnn.Device,
    ccl_manager: CCLManager,
    spec: AllGatherSpec,
    impl_config: AllGatherImplConfig,
) -> ttnn.Tensor:
    """
    Perform all-gather operation across devices.

    Args:
        input_tensor: Input tensor to gather
        mesh_device: Device mesh for distributed operation
        ccl_manager: CCL manager instance for semaphore handling
        spec: All-gather specification
        impl_config: Implementation configuration

    Returns:
        Gathered tensor containing data from all devices
    """
    # Single device case - no gathering needed
    if list(mesh_device.shape) == (1, 1) or (spec.cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Ensure correct memory configuration
    if not impl_config.sharded:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    # Cast to CCL dtype if needed
    if input_tensor.dtype != impl_config.dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, impl_config.dtype)
        if impl_config.sharded and impl_config.memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, impl_config.memory_config, impl_config.dtype)

    # Perform all-gather operation
    if spec.cluster_axis is None:
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=spec.gather_dim,
            multi_device_global_semaphore=ccl_manager.get_and_cycle_ag_semaphore_handles(),
            num_links=impl_config.num_all_gather_links,
            topology=impl_config.topology,
            memory_config=impl_config.memory_config,
            barrier_semaphore=ccl_manager.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=impl_config.chunks_per_sync,
            num_workers_per_link=impl_config.num_workers_per_link,
            num_buffers_per_channel=impl_config.num_buffers_per_channel,
        )
    else:
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=spec.gather_dim,
            multi_device_global_semaphore=ccl_manager.get_and_cycle_ag_semaphore_handles(spec.cluster_axis),
            num_links=impl_config.num_all_gather_links,
            cluster_axis=spec.cluster_axis,
            topology=impl_config.topology,
            memory_config=impl_config.memory_config,
            barrier_semaphore=ccl_manager.get_and_cycle_barrier_semaphore_handle(spec.cluster_axis),
            chunks_per_sync=impl_config.chunks_per_sync,
            num_workers_per_link=impl_config.num_workers_per_link,
            num_buffers_per_channel=impl_config.num_buffers_per_channel,
        )

    input_tensor.deallocate(True)
    return gathered


# Alias for consistent naming pattern
prefill_forward = all_gather_forward
decode_forward = all_gather_forward
