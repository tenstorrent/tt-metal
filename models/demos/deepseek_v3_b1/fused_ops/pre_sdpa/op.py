# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.pre_sdpa.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_bfloat16_packed


class PreSDPA:
    """
    Pre-SDPA fused operation implementation using ttnn.generic_op.

    This class implements the pre-SDPA operations as a fused execution:
    - RMSNorm on a single core
    - Multicast of the result to a grid of cores
    """

    @staticmethod
    def golden(input_tensor, gamma_tensor, num_output_cores, epsilon=1e-6):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor)
            gamma_tensor: Gamma/weight tensor (torch.Tensor)
            num_output_cores: Number of cores to multicast to
            epsilon: Small value to avoid division by zero

        Returns:
            Output tensor with pre-SDPA operations applied (RMSNorm result repeated num_output_cores times)
        """
        # RMSNorm
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance + epsilon)
        rmsnorm_result = normalized * gamma_tensor
        # Repeat for multicast output
        return rmsnorm_result.repeat(num_output_cores, 1)

    @staticmethod
    def op(
        input_tensor,
        gamma_tensor,
        matmul_weights_tensor,
        output_tensor,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
    ):
        """
        Execute pre-SDPA fused operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded on single core)
            gamma_tensor: Gamma/weight tensor (must be sharded, same shape as input)
            matmul_weights_tensor: Matmul weights tensor (must be width sharded)
            output_tensor: Pre-allocated output tensor (must be sharded on single core)
            epsilon: Small value to avoid division by zero
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel

        Returns:
            Output tensor with RMSNorm applied
        """
        # Get tensor properties
        input_shape = input_tensor.shape
        data_format = input_tensor.dtype

        # Interpret N 1x32 tiles as full 32x32 or 16x32 tiles
        # eg. [1, 7168] = 7 full 32x32 tiles
        # eg. [1, 1536] = 3 half 16x32 tiles
        # eg. [1, 512] = 1 half 16x32 tile
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (input_shape[1] // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        tile_height, tile_width = interpreted_tile.tile_shape

        # Calculate single tile size in bytes (bfloat16 = 2 bytes per element)
        tile_size = interpreted_tile.get_tile_size(data_format)

        # Calculate num_tiles from tensor shape
        num_tiles = (input_shape[0] * input_shape[1]) // (tile_height * tile_width)

        # Get number of elements for RMS calculation
        numel = input_tensor.logical_volume()

        # Get core grid from input tensor's memory config
        input_memory_config = input_tensor.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        input_core_ranges = list(input_core_grid.ranges())
        rmsnorm_core = input_core_ranges[0].start
        rmsnorm_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(rmsnorm_core, rmsnorm_core)])

        # Get full device grid
        device = input_tensor.device()
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # Calculate runtime args
        epsilon_packed = float_to_bfloat16_packed(epsilon)

        # Compute 1/sqrt(num_elements) for RMS reduction
        inv_sqrt_numel = 1.0 / math.sqrt(float(numel))
        scalar_packed = float_to_bfloat16_packed(inv_sqrt_numel)

        # Define circular buffer page size
        cb_page_size = tile_size

        # CB indices
        input_cb = 0
        scalars_cb = 1
        interm_cb = 2
        gamma_cb = 3
        output_cb = 4
        matmul_weights_cb = 5

        # Create tile descriptor for proper tile dimensions
        tile_descriptor = ttnn.TileDescriptor(interpreted_tile)

        # Create circular buffer descriptors
        # CB 0: Input (created from sharded tensor)
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
        # Update the tile descriptor in the format descriptor
        in_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        in_cb_descriptor.format_descriptors[0].page_size = cb_page_size

        # CB 1: Scalars (epsilon and reduction scalar)
        scalars_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=scalars_cb,
            data_format=data_format,
            page_size=cb_page_size,
            tile=tile_descriptor,
        )
        scalars_cb_descriptor = ttnn.CBDescriptor(
            total_size=2 * cb_page_size,
            core_ranges=rmsnorm_core_grid,
            format_descriptors=[scalars_cb_format],
        )

        # CB 2: Intermediate buffer
        interm_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=interm_cb,
            data_format=data_format,
            page_size=cb_page_size,
            tile=tile_descriptor,
        )
        interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=(num_tiles + 1) * cb_page_size,
            core_ranges=rmsnorm_core_grid,
            format_descriptors=[interm_cb_format],
        )

        # CB 3: Gamma (created from sharded tensor)
        gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gamma_cb, gamma_tensor)
        # Update the tile descriptor in the format descriptor
        gamma_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        gamma_cb_descriptor.format_descriptors[0].page_size = cb_page_size

        # CB 4: Output (created from sharded tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)
        # Update the tile descriptor in the format descriptor
        out_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        out_cb_descriptor.format_descriptors[0].page_size = cb_page_size

        # CB 5: Matmul weights (created from sharded tensor) - not used yet
        matmul_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_weights_cb, matmul_weights_tensor)

        # ========================================================================
        # Kernel descriptors
        # ========================================================================

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/pre_sdpa_kernel.cpp",
            core_ranges=full_device_grid,
            # NCRISC compile-time args: [input_cb, scalars_cb, gamma_cb, num_tiles, tiny_tile]
            ncrisc_compile_time_args=[
                input_cb,
                scalars_cb,
                gamma_cb,
                num_tiles,
                is_16x32_tile,
            ],
            # NCRISC common runtime args: epsilon + scalar
            ncrisc_common_runtime_args=[
                epsilon_packed,
                scalar_packed,
            ],
            # BRISC compile-time args: [output_cb, num_tiles]
            brisc_compile_time_args=[
                output_cb,
                num_tiles,
            ],
            # TRISC compile-time args: [input_cb, scalars_cb, interm_cb, gamma_cb, output_cb,
            #                           fp32_acc, num_tiles, epsilon_index, scalar_index]
            trisc_compile_time_args=[
                input_cb,
                scalars_cb,
                interm_cb,
                gamma_cb,
                output_cb,
                1 if fp32_dest_acc_en else 0,
                num_tiles,
                0,  # epsilon_index
                1,  # scalar_index
            ],
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            # Per-core compile-time role differentiation
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_input_core",
                    core_range=rmsnorm_core,  # First core is the input core
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                in_cb_descriptor,
                scalars_cb_descriptor,
                interm_cb_descriptor,
                gamma_cb_descriptor,
                out_cb_descriptor,
                matmul_weights_cb_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [input_tensor, gamma_tensor, matmul_weights_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
