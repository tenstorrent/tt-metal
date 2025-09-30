# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware configuration interface for TT devices"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import ttnn


class DeviceArch(Enum):
    """Supported device architectures"""

    GRAYSKULL = "grayskull"
    WORMHOLE_B0 = "wormhole_b0"
    BLACKHOLE = "blackhole"


class MemoryConfig(Enum):
    """Memory configuration options"""

    DRAM = "dram"
    L1 = "l1"
    L1_INTERLEAVED = "l1_interleaved"
    DRAM_INTERLEAVED = "dram_interleaved"


@dataclass
class DeviceConfig:
    """Configuration for a single TT device"""

    device_id: int
    arch: DeviceArch
    num_cores: int
    l1_memory_per_core: int
    dram_memory_per_channel: int
    dram_channels: int
    noc_bandwidth: float  # GB/s
    compute_with_storage_grid: Tuple[int, int]


@dataclass
class ComputeKernelConfig:
    """Compute kernel configuration"""

    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    math_approx_mode: bool = True
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = True


@dataclass
class ShardingConfig:
    """Configuration for tensor sharding across cores"""

    strategy: str = "block"  # Options: "block", "width", "height"
    core_grid: Optional[Tuple[int, int]] = None
    orientation: str = "row_major"  # Options: "row_major", "column_major"


class HWConfig:
    """
    Hardware configuration manager for TTT models.

    Provides device configuration, memory management, and optimization settings.
    """

    def __init__(
        self,
        device_arch: DeviceArch,
        num_devices: int = 1,
        device_ids: Optional[List[int]] = None,
    ):
        self.device_arch = device_arch
        self.num_devices = num_devices
        self.device_ids = device_ids or list(range(num_devices))

        # Initialize device mesh
        self._setup_device_mesh()

        # Default configurations
        self._setup_default_configs()

    def _setup_device_mesh(self):
        """Setup device mesh for multi-device execution"""
        if self.num_devices == 1:
            self.mesh_device = ttnn.open_device(device_id=self.device_ids[0])
        else:
            # Create mesh device for multi-chip
            self.mesh_device = ttnn.create_device_mesh(
                device_ids=self.device_ids,
                num_devices=self.num_devices,
            )

    def _setup_default_configs(self):
        """Setup default configurations based on architecture"""
        if self.device_arch == DeviceArch.WORMHOLE_B0:
            self.default_compute_kernel_config = ComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self.max_grid_size = (8, 8)
            self.l1_memory_size = 1024 * 1024  # 1MB per core
        elif self.device_arch == DeviceArch.BLACKHOLE:
            self.default_compute_kernel_config = ComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self.max_grid_size = (10, 10)
            self.l1_memory_size = 1536 * 1024  # 1.5MB per core
        else:
            # Grayskull defaults
            self.default_compute_kernel_config = ComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            self.max_grid_size = (8, 7)
            self.l1_memory_size = 512 * 1024  # 512KB per core

    def get_optimal_shard_config(
        self,
        tensor_shape: Tuple[int, ...],
        dtype: ttnn.DataType = ttnn.bfloat16,
        memory_config: MemoryConfig = MemoryConfig.L1,
    ) -> ShardingConfig:
        """
        Get optimal sharding configuration for a tensor.

        Args:
            tensor_shape: Shape of the tensor to shard
            dtype: Data type of the tensor
            memory_config: Target memory configuration

        Returns:
            Optimal sharding configuration
        """
        # Calculate tensor size
        element_size = self._get_dtype_size(dtype)
        tensor_size = 1
        for dim in tensor_shape:
            tensor_size *= dim
        tensor_size *= element_size

        # Determine optimal sharding based on size and memory
        if memory_config in [MemoryConfig.L1, MemoryConfig.L1_INTERLEAVED]:
            # For L1, we need to fit within per-core memory
            max_shard_size = self.l1_memory_size // 2  # Leave room for double buffering

            # Calculate required cores
            required_cores = (tensor_size + max_shard_size - 1) // max_shard_size

            # Find optimal grid
            core_grid = self._find_optimal_core_grid(required_cores)

            # Determine sharding strategy
            if len(tensor_shape) >= 2:
                height, width = tensor_shape[-2:]
                if height > width:
                    strategy = "height"
                else:
                    strategy = "width"
            else:
                strategy = "block"

            return ShardingConfig(
                strategy=strategy,
                core_grid=core_grid,
                orientation="row_major",
            )
        else:
            # For DRAM, use interleaved storage
            return ShardingConfig(
                strategy="block",
                core_grid=None,
                orientation="row_major",
            )

    def get_compute_kernel_config(
        self,
        operation: str,
        input_dtype: ttnn.DataType = ttnn.bfloat16,
        accumulate_dtype: Optional[ttnn.DataType] = None,
    ) -> ComputeKernelConfig:
        """
        Get compute kernel configuration for an operation.

        Args:
            operation: Type of operation (e.g., "matmul", "conv", "attention")
            input_dtype: Input data type
            accumulate_dtype: Accumulation data type

        Returns:
            Optimized compute kernel configuration
        """
        config = self.default_compute_kernel_config

        # Adjust based on operation type
        if operation == "attention":
            # Attention benefits from higher precision
            config.math_fidelity = ttnn.MathFidelity.HiFi4
            config.fp32_dest_acc_en = True
        elif operation == "matmul":
            # Standard matmul settings
            if input_dtype == ttnn.bfloat8_b:
                config.math_fidelity = ttnn.MathFidelity.HiFi2
        elif operation == "layernorm" or operation == "rmsnorm":
            # Normalization needs high precision
            config.fp32_dest_acc_en = True
            config.math_fidelity = ttnn.MathFidelity.HiFi4

        return config

    def get_memory_config(
        self,
        tensor_size: int,
        access_pattern: str = "sequential",
        double_buffer: bool = True,
    ) -> ttnn.MemoryConfig:
        """
        Get memory configuration for a tensor.

        Args:
            tensor_size: Size of tensor in bytes
            access_pattern: Access pattern ("sequential", "random", "broadcast")
            double_buffer: Whether to enable double buffering

        Returns:
            Memory configuration
        """
        # Determine if tensor fits in L1
        buffer_factor = 2 if double_buffer else 1
        required_l1 = tensor_size * buffer_factor

        if required_l1 <= self.l1_memory_size:
            # Use L1 for small tensors
            if access_pattern == "broadcast":
                # Use interleaved for broadcast
                buffer_type = ttnn.BufferType.L1
                layout = ttnn.TensorMemoryLayout.INTERLEAVED
            else:
                # Use block sharded for better locality
                buffer_type = ttnn.BufferType.L1
                layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

            return ttnn.MemoryConfig(
                memory_layout=layout,
                buffer_type=buffer_type,
            )
        else:
            # Use DRAM for large tensors
            return ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttnn.BufferType.DRAM,
            )

    def get_program_config(
        self,
        operation: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """
        Get program configuration for an operation.

        Args:
            operation: Operation type
            input_shapes: List of input tensor shapes
            output_shape: Output tensor shape

        Returns:
            Program configuration dictionary
        """
        # This would contain operation-specific optimizations
        # For now, return basic config
        return {
            "compute_kernel_config": self.get_compute_kernel_config(operation),
            "memory_config": self.get_memory_config(self._calculate_tensor_size(output_shape)),
            "core_grid": self.max_grid_size,
        }

    def _find_optimal_core_grid(self, required_cores: int) -> Tuple[int, int]:
        """Find optimal core grid for given number of cores"""
        max_x, max_y = self.max_grid_size

        # Find factors closest to square
        best_grid = (1, 1)
        best_diff = float("inf")

        for x in range(1, max_x + 1):
            for y in range(1, max_y + 1):
                if x * y >= required_cores:
                    diff = abs(x - y)
                    if diff < best_diff:
                        best_diff = diff
                        best_grid = (x, y)

        return best_grid

    def _get_dtype_size(self, dtype: ttnn.DataType) -> int:
        """Get size in bytes for a data type"""
        dtype_sizes = {
            ttnn.uint8: 1,
            ttnn.int8: 1,
            ttnn.uint16: 2,
            ttnn.int16: 2,
            ttnn.bfloat16: 2,
            ttnn.float16: 2,
            ttnn.uint32: 4,
            ttnn.int32: 4,
            ttnn.float32: 4,
            ttnn.bfloat8_b: 1,
            ttnn.bfloat4_b: 0.5,
        }
        return dtype_sizes.get(dtype, 2)  # Default to 2 bytes

    def _calculate_tensor_size(self, shape: Tuple[int, ...]) -> int:
        """Calculate tensor size in bytes"""
        size = 1
        for dim in shape:
            size *= dim
        return size * 2  # Assume bfloat16 by default

    def close(self):
        """Close device connections"""
        if hasattr(self, "mesh_device"):
            ttnn.close_device(self.mesh_device)
