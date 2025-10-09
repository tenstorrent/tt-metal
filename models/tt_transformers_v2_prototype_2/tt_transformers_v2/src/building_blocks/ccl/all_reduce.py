# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
All-Reduce collective operation.

This module provides all-reduce functionality for distributed tensor operations,
performing element-wise reduction across devices and broadcasting the result.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import ttnn

from .manager import CCLManager


@dataclass
class AllReduceSpec:
    """
    Mathematical specification for all-reduce operation.

    All-reduce performs element-wise reduction (sum) across all specified devices
    and broadcasts the result to all devices.
    """

    mesh_shape: Tuple[int, int]  # (rows, columns) of device mesh
    cluster_axis: Optional[int] = None  # 0 for row-wise, 1 for column-wise, None for all
    reduce_dim: int = 0  # Dimension along which to reduce

    def validate(self):
        """Validate spec constraints."""
        assert len(self.mesh_shape) == 2, "mesh_shape must be (rows, columns)"
        assert self.mesh_shape[0] > 0 and self.mesh_shape[1] > 0, "mesh dimensions must be positive"
        if self.cluster_axis is not None:
            assert self.cluster_axis in [0, 1], "cluster_axis must be 0, 1, or None"
        assert 0 <= self.reduce_dim <= 3, "reduce_dim must be between 0 and 3"


@dataclass
class AllReduceImplConfig:
    """
    TTNN-specific implementation configuration for all-reduce.

    Contains device-specific optimizations and performance tuning parameters
    for the all-reduce operation.
    """

    # Communication links configuration
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2

    # Topology configuration
    topology: ttnn.Topology = ttnn.Topology.Linear

    # Memory configuration
    memory_config: Optional[ttnn.MemoryConfig] = None
    intermediate_memory_config: Optional[ttnn.MemoryConfig] = None

    # Data type for communication
    dtype: ttnn.DataType = ttnn.bfloat16

    # Sharding configuration
    sharded: bool = False

    # Composite operation mode
    use_composite: bool = False

    # Performance tuning parameters
    chunks_per_sync: int = 10
    num_workers_per_link: int = 2
    num_buffers_per_channel: int = 2


def get_default_impl_config(
    spec: AllReduceSpec, device: str, mode: Literal["prefill", "decode"] = "prefill", strategy: str = "default"
) -> AllReduceImplConfig:
    """
    Return default implementation configuration for all-reduce.

    Args:
        spec: All-reduce specification
        device: Target device (e.g., "N150", "N300", "T3000", "TG")
        mode: Execution mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Default AllReduceImplConfig for the specified device and mode
    """
    if device.startswith("N150"):
        # Single device, minimal configuration
        return AllReduceImplConfig(
            num_reduce_scatter_links=1,
            num_all_gather_links=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    elif device.startswith("N300"):
        # Two devices, optimize for bandwidth
        return AllReduceImplConfig(
            num_reduce_scatter_links=1,
            num_all_gather_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    elif device.startswith("T3000") or device == "T3K":
        # 8 devices, higher parallelism
        return AllReduceImplConfig(
            num_reduce_scatter_links=2,
            num_all_gather_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=20,
        )
    elif device == "TG":
        # Galaxy configuration with mesh topology
        return AllReduceImplConfig(
            num_reduce_scatter_links=1,
            num_all_gather_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            use_composite=True,
            chunks_per_sync=10,
        )
    else:
        # Default conservative configuration
        return AllReduceImplConfig()


def all_reduce_forward(
    input_tensor: ttnn.Tensor,
    mesh_device: ttnn.Device,
    ccl_manager: CCLManager,
    spec: AllReduceSpec,
    impl_config: AllReduceImplConfig,
) -> ttnn.Tensor:
    """
    Perform all-reduce operation across devices.

    Args:
        input_tensor: Input tensor to reduce
        mesh_device: Device mesh for distributed operation
        ccl_manager: CCL manager instance for semaphore handling
        spec: All-reduce specification
        impl_config: Implementation configuration

    Returns:
        Reduced tensor available on all devices
    """
    # Single device case - no reduction needed
    if list(mesh_device.shape) == [1, 1] or (spec.cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Ensure dim 0 and 1 are 1 for proper operation
    original_shape = input_tensor.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        input_tensor = ttnn.reshape(
            input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # N300 and T3K: use reduce_scatter for linear topologies
    if 1 in list(mesh_device.shape):
        if input_tensor.is_sharded() and not impl_config.sharded:
            input_tensor_sharded = input_tensor
            input_tensor = ttnn.sharded_to_interleaved(input_tensor_sharded, ttnn.L1_MEMORY_CONFIG)
            input_tensor_sharded.deallocate(True)

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=spec.reduce_dim,
            multi_device_global_semaphore=ccl_manager.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=ccl_manager.get_and_cycle_barrier_semaphore_handle(),
            num_links=impl_config.num_reduce_scatter_links,
            memory_config=impl_config.memory_config,
            intermediate_memory_config=impl_config.intermediate_memory_config,
            topology=impl_config.topology,
            chunks_per_sync=impl_config.chunks_per_sync,
            num_workers_per_link=impl_config.num_workers_per_link,
            num_buffers_per_channel=impl_config.num_buffers_per_channel,
        )
        input_tensor.deallocate(True)
        return reduced

    # TG: full all_reduce for mesh topologies
    # Cast to CCL dtype if needed
    if input_tensor.dtype != impl_config.dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, impl_config.dtype)
        if impl_config.sharded and impl_config.memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, impl_config.memory_config, impl_config.dtype)

    # Ensure correct memory configuration
    if not impl_config.sharded:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    if not impl_config.use_composite:
        # Separate all-gather and reduce operations
        gathered_tensor = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=spec.reduce_dim,
            multi_device_global_semaphore=ccl_manager.get_and_cycle_ag_semaphore_handles(spec.cluster_axis),
            num_links=impl_config.num_all_gather_links,
            cluster_axis=spec.cluster_axis,
            topology=impl_config.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not impl_config.sharded else impl_config.memory_config,
            barrier_semaphore=ccl_manager.get_and_cycle_barrier_semaphore_handle(spec.cluster_axis),
            chunks_per_sync=impl_config.chunks_per_sync,
            num_workers_per_link=impl_config.num_workers_per_link,
            num_buffers_per_channel=impl_config.num_buffers_per_channel,
        )

        if impl_config.sharded:
            gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)

        reduced_tensor = ttnn.experimental.fast_reduce_nc(
            gathered_tensor,
            dims=[spec.reduce_dim],
            output=None,
            compute_kernel_config=None,
            memory_config=ttnn.L1_MEMORY_CONFIG if impl_config.sharded else ttnn.DRAM_MEMORY_CONFIG,
        )

        gathered_tensor.deallocate(True)
    else:
        # Composite reduce-scatter followed by all-gather
        input_mem_cfg = input_tensor.memory_config()

        reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=spec.reduce_dim,
            multi_device_global_semaphore=ccl_manager.get_and_cycle_rs_semaphore_handles(spec.cluster_axis),
            barrier_semaphore=ccl_manager.get_and_cycle_barrier_semaphore_handle(spec.cluster_axis),
            num_links=impl_config.num_reduce_scatter_links,
            cluster_axis=spec.cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not impl_config.sharded else impl_config.memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=impl_config.topology,
            chunks_per_sync=impl_config.chunks_per_sync,
            num_workers_per_link=impl_config.num_workers_per_link,
            num_buffers_per_channel=impl_config.num_buffers_per_channel,
        )

        reduced_tensor = ttnn.experimental.all_gather_async(
            reduced_tensor,
            persistent_output_buffer=None,
            dim=spec.reduce_dim,
            multi_device_global_semaphore=ccl_manager.get_and_cycle_ag_semaphore_handles(spec.cluster_axis),
            num_links=impl_config.num_all_gather_links,
            cluster_axis=spec.cluster_axis,
            topology=impl_config.topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=ccl_manager.get_and_cycle_barrier_semaphore_handle(spec.cluster_axis),
            chunks_per_sync=impl_config.chunks_per_sync,
            num_workers_per_link=impl_config.num_workers_per_link,
            num_buffers_per_channel=impl_config.num_buffers_per_channel,
        )

    # Reshape back to original shape
    reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)

    return reduced_tensor


# Alias for consistent naming pattern
prefill_forward = all_reduce_forward
decode_forward = all_reduce_forward
