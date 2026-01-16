# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_bfloat16_packed, float_to_uint32


class RMSNormSingleCore:
    """
    Single-core RMS normalization implementation using ttnn.generic_op.

    This class implements RMS normalization as a static operation for single-core execution.
    """

    @staticmethod
    def golden(input_tensor, gamma_tensor, epsilon=1e-6):
        """
        PyTorch reference implementation of RMS norm for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor)
            gamma_tensor: Gamma/weight tensor (torch.Tensor)
            epsilon: Small value to avoid division by zero

        Returns:
            Output tensor with RMS norm applied
        """
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance + epsilon)
        return normalized * gamma_tensor

    @staticmethod
    def op(
        input_tensor,
        gamma_tensor,
        output_tensor,
        epsilon=1e-6,
        numel=None,
        fp32_dest_acc_en=False,
        rsqrt_fast_approx=False,
    ):
        """
        Execute RMS norm operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded)
            gamma_tensor: Gamma/weight tensor (must be sharded, same shape as input)
            output_tensor: Pre-allocated output tensor (must be sharded, same shape as input)
            epsilon: Small value to avoid division by zero
            numel: Number of elements to use for RMS calculation (defaults to input logical volume)
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel
            rsqrt_fast_approx: Whether to use fast approximation for rsqrt

        Returns:
            Output tensor with RMS norm applied
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
        num_faces = interpreted_tile.num_faces
        tile_height, tile_width = interpreted_tile.tile_shape

        # Calculate single tile size in bytes (bfloat16 = 2 bytes per element)
        tile_size = interpreted_tile.get_tile_size(data_format)

        # Calculate num_tiles from tensor shape
        num_tiles = (input_shape[0] * input_shape[1]) // (tile_height * tile_width)

        if numel is None:
            numel = input_tensor.logical_volume()

        # Hard-code to first core (0, 0)
        core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

        # Calculate runtime args
        epsilon_packed = float_to_uint32(epsilon)

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
            total_size=cb_page_size,
            core_ranges=core_grid,
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
            total_size=num_tiles * cb_page_size,
            core_ranges=core_grid,
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

        # Named compile-time args for NCRISC (reader)
        ncrisc_named_compile_time_args = [
            ("rmsnorm_input_cb", input_cb),
            ("rmsnorm_scalars_cb", scalars_cb),
            ("rmsnorm_gamma_cb", gamma_cb),
            ("rmsnorm_num_tiles", num_tiles),
            ("rmsnorm_num_faces", num_faces),
        ]

        # Named compile-time args for TRISC (compute)
        trisc_named_compile_time_args = [
            ("rmsnorm_input_cb", input_cb),
            ("rmsnorm_scalars_cb", scalars_cb),
            ("rmsnorm_interm_cb", interm_cb),
            ("rmsnorm_gamma_cb", gamma_cb),
            ("rmsnorm_output_cb", output_cb),
            ("rmsnorm_fp32_acc", 1 if fp32_dest_acc_en else 0),
            ("rmsnorm_num_tiles", num_tiles),
            ("rmsnorm_rsqrt_fast_approx", 1 if rsqrt_fast_approx else 0),
        ]

        # Unified kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/rmsnorm/kernels/rmsnorm_kernel.cpp",
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=[],
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            ncrisc_common_runtime_args=[scalar_packed],
            trisc_common_runtime_args=[epsilon_packed],
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_active_core",
                    core_range=core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[in_cb_descriptor, scalars_cb_descriptor, interm_cb_descriptor, gamma_cb_descriptor, out_cb_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, gamma_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
