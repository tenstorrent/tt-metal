# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Distributed RMS normalization using collective communication.

This module provides distributed RMS normalization that computes statistics
locally and gathers them across devices before applying normalization.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import ttnn

from .all_gather import AllGatherImplConfig, AllGatherSpec, all_gather_forward
from .manager import CCLManager


@dataclass
class DistributedRMSNormSpec:
    """
    Specification for distributed RMS normalization.

    Combines RMS normalization parameters with collective communication.
    """

    hidden_dim: int
    epsilon: float = 1e-5

    def validate(self):
        """Validate spec constraints."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.epsilon > 0, "epsilon must be positive"


@dataclass
class DistributedRMSNormImplConfig:
    """
    Implementation configuration for distributed RMS normalization.
    """

    compute_kernel_config: Optional[dict] = None
    stats_memory_config: Optional[ttnn.MemoryConfig] = None
    sharded_input_memory_config: Optional[ttnn.MemoryConfig] = None
    sharded_program_config: Optional[dict] = None
    sharded_stats_memory_config: Optional[ttnn.MemoryConfig] = None


def get_default_impl_config(
    spec: DistributedRMSNormSpec, device: str, mode: Literal["prefill", "decode"] = "prefill", strategy: str = "default"
) -> DistributedRMSNormImplConfig:
    """
    Return default implementation configuration for distributed RMS norm.

    Args:
        spec: Distributed RMS norm specification
        device: Target device (e.g., "N150", "N300", "T3000", "TG")
        mode: Execution mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Default DistributedRMSNormImplConfig for the specified device and mode
    """
    if device == "TG" or device.startswith("Galaxy"):
        # Galaxy configuration with special handling
        hidden_size = spec.hidden_dim
        core_grid = (
            min(4, hidden_size // 4 // 32 // 8),
            8,
        )  # dividing by 4 and 8 for num_cols/rows of mesh, and 32 for tile size

        return DistributedRMSNormImplConfig(
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            sharded_input_memory_config=ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size // 4),
                core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
            sharded_program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid[1], core_grid[0]),
                subblock_w=(hidden_size // 4 // (core_grid[0] * core_grid[1])) // 32,
                block_h=1,
                block_w=(hidden_size // 4 // (core_grid[0] * core_grid[1])) // 32,
                inplace=False,
            ),
            sharded_stats_memory_config=ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        )
    else:
        # Default configuration
        compute_kernel_config = (
            ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            if mode == "prefill"
            else None
        )

        return DistributedRMSNormImplConfig(
            compute_kernel_config=compute_kernel_config,
            stats_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def distributed_rmsnorm_forward(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    mesh_device: ttnn.Device,
    ccl_manager: CCLManager,
    spec: DistributedRMSNormSpec,
    impl_config: DistributedRMSNormImplConfig,
    mesh_shape: tuple,
    mode: Literal["prefill", "decode"] = "prefill",
) -> ttnn.Tensor:
    """
    Perform distributed RMS normalization across devices.

    This operation computes RMS normalization statistics locally, gathers them
    across devices, and applies the normalization using the aggregated statistics.

    Args:
        input_tensor: Input tensor to normalize
        weight: Normalization weights (gamma)
        mesh_device: Device mesh for distributed operation
        ccl_manager: CCL manager instance
        norm_spec: RMS normalization specification
        norm_impl_config: RMS normalization implementation config
        ccl_spec: CCL specification
        ccl_impl_config: CCL implementation config
        mode: Execution mode (prefill or decode)

    Returns:
        Normalized tensor
    """
    if mode == "decode" and impl_config.sharded_program_config is not None:
        # Sharded decode mode
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=impl_config.sharded_input_memory_config)

        # Compute local RMS statistics
        tt_stats = ttnn.rms_norm_pre_all_gather(input_tensor, program_config=impl_config.sharded_program_config)

        # Gather statistics across devices
        gather_spec = AllGatherSpec(mesh_shape=mesh_shape, cluster_axis=1, gather_dim=3)
        gather_impl = AllGatherImplConfig(
            num_all_gather_links=1,
            memory_config=impl_config.sharded_stats_memory_config,
        )

        tt_stats = all_gather_forward(
            tt_stats,
            mesh_device,
            ccl_manager,
            gather_spec,
            gather_impl,
        )

        # Apply normalization with gathered statistics
        tt_out = ttnn.rms_norm_post_all_gather(
            input_tensor,
            epsilon=spec.epsilon,
            weight=weight,
            program_config=impl_config.sharded_program_config,
            stats=tt_stats,
        )
        tt_stats.deallocate(True)
    else:
        # Non-sharded mode (prefill or non-sharded decode)
        # Compute local RMS statistics
        tt_stats = ttnn.rms_norm_pre_all_gather(
            input_tensor, compute_kernel_config=impl_config.compute_kernel_config, dtype=ttnn.bfloat16
        )

        # Reshape for gathering
        padded_shape = (1, 1, input_tensor.shape[-2], 32)
        tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))

        # Gather statistics
        gather_spec = AllGatherSpec(mesh_shape=mesh_shape, cluster_axis=1, gather_dim=3)
        gather_impl = AllGatherImplConfig(
            num_all_gather_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tt_stats_gathered = all_gather_forward(
            tt_stats,
            mesh_device,
            ccl_manager,
            gather_spec,
            gather_impl,
        )
        tt_stats.deallocate(True)

        # Apply normalization
        tt_out = ttnn.rms_norm_post_all_gather(
            input_tensor,
            tt_stats_gathered,
            epsilon=spec.epsilon,
            weight=weight,
            compute_kernel_config=impl_config.compute_kernel_config,
        )
        tt_stats_gathered.deallocate(True)

    return tt_out


# Alias for consistent naming pattern
prefill_forward = distributed_rmsnorm_forward
decode_forward = distributed_rmsnorm_forward
