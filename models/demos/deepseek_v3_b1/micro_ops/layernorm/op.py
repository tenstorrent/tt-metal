# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm single-core generic op implementation using ProgramDescriptor API."""


import torch

import ttnn

# =============================================================================
# Circular Buffer Indices (Step 1.3.1)
# =============================================================================
# Input/working CBs (0-15)
CB_INPUT_RM = 0  # Input row-major sticks from DRAM
CB_INPUT_TILED = 1  # Input after tilization
CB_GAMMA_RM = 2  # Gamma row-major sticks
CB_GAMMA_TILED = 3  # Gamma after tilization
CB_BETA_RM = 4  # Beta row-major sticks
CB_BETA_TILED = 5  # Beta after tilization
CB_SCALARS = 6  # Scalar values (epsilon, 1/W)
CB_INTERM = 7  # Intermediate computation results

# Output CBs (16+)
CB_OUTPUT_TILED = 16  # Output in tile format
CB_OUTPUT_RM = 17  # Output row-major sticks


def _calculate_sizes(input_shape, dtype):
    """
    Calculate buffer sizes for LayerNorm operation.

    Args:
        input_shape: Shape of the input tensor (list/tuple)
        dtype: Data type of the tensor (ttnn.bfloat16, etc.)

    Returns:
        dict with:
            - W: Final dimension (normalization dimension)
            - num_rows: Total number of rows to normalize
            - tiles_per_row: Number of tiles per row
            - stick_size: Size of one row-major stick in bytes (aligned to 32)
            - tile_size: Size of one 32x32 tile in bytes
            - element_size: Size of one element in bytes
    """
    # Final dimension is the normalization dimension
    W = input_shape[-1]

    # Total rows to normalize = product of all dims except last
    num_rows = 1
    for dim in input_shape[:-1]:
        num_rows *= dim

    # Tiles per row (32-wide tiles)
    tiles_per_row = (W + 31) // 32

    # Element size in bytes (bfloat16 = 2 bytes)
    if dtype == ttnn.bfloat16:
        element_size = 2
    elif dtype == ttnn.float32:
        element_size = 4
    else:
        # Default to bfloat16
        element_size = 2

    # Stick size = W * element_size, aligned to 32 bytes
    stick_size_unaligned = W * element_size
    stick_size = ((stick_size_unaligned + 31) // 32) * 32

    # Tile size = 32x32 elements * element_size
    tile_size = 32 * 32 * element_size

    return {
        "W": W,
        "num_rows": num_rows,
        "tiles_per_row": tiles_per_row,
        "stick_size": stick_size,
        "tile_size": tile_size,
        "element_size": element_size,
    }


def _create_cb_descriptors(core_grid, sizes, dtype):
    """
    Create circular buffer descriptors for LayerNorm operation.

    Args:
        core_grid: CoreRangeSet specifying which cores to create CBs on
        sizes: Dict from _calculate_sizes()
        dtype: Data type (ttnn.bfloat16, etc.)

    Returns:
        List of CBDescriptor objects for all 10 CBs
    """
    stick_size = sizes["stick_size"]
    tile_size = sizes["tile_size"]
    tiles_per_row = sizes["tiles_per_row"]

    # Standard 32x32 tile descriptor
    tile_32x32 = ttnn.Tile((32, 32))
    tile_descriptor = ttnn.TileDescriptor(tile_32x32)

    cb_descriptors = []

    # CB 0: Input row-major sticks (double buffer for pipelining)
    cb_input_rm_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_INPUT_RM,
        data_format=dtype,
        page_size=stick_size,
    )
    cb_input_rm_descriptor = ttnn.CBDescriptor(
        total_size=2 * stick_size,  # Double buffer
        core_ranges=core_grid,
        format_descriptors=[cb_input_rm_format],
    )
    cb_descriptors.append(cb_input_rm_descriptor)

    # CB 1: Input after tilization
    cb_input_tiled_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_INPUT_TILED,
        data_format=dtype,
        page_size=tile_size,
        tile=tile_descriptor,
    )
    cb_input_tiled_descriptor = ttnn.CBDescriptor(
        total_size=tiles_per_row * tile_size,
        core_ranges=core_grid,
        format_descriptors=[cb_input_tiled_format],
    )
    cb_descriptors.append(cb_input_tiled_descriptor)

    # CB 2: Gamma row-major sticks (single buffer, read once)
    cb_gamma_rm_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_GAMMA_RM,
        data_format=dtype,
        page_size=stick_size,
    )
    cb_gamma_rm_descriptor = ttnn.CBDescriptor(
        total_size=stick_size,  # Single buffer
        core_ranges=core_grid,
        format_descriptors=[cb_gamma_rm_format],
    )
    cb_descriptors.append(cb_gamma_rm_descriptor)

    # CB 3: Gamma after tilization
    cb_gamma_tiled_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_GAMMA_TILED,
        data_format=dtype,
        page_size=tile_size,
        tile=tile_descriptor,
    )
    cb_gamma_tiled_descriptor = ttnn.CBDescriptor(
        total_size=tiles_per_row * tile_size,
        core_ranges=core_grid,
        format_descriptors=[cb_gamma_tiled_format],
    )
    cb_descriptors.append(cb_gamma_tiled_descriptor)

    # CB 4: Beta row-major sticks (single buffer, read once)
    cb_beta_rm_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_BETA_RM,
        data_format=dtype,
        page_size=stick_size,
    )
    cb_beta_rm_descriptor = ttnn.CBDescriptor(
        total_size=stick_size,  # Single buffer
        core_ranges=core_grid,
        format_descriptors=[cb_beta_rm_format],
    )
    cb_descriptors.append(cb_beta_rm_descriptor)

    # CB 5: Beta after tilization
    cb_beta_tiled_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_BETA_TILED,
        data_format=dtype,
        page_size=tile_size,
        tile=tile_descriptor,
    )
    cb_beta_tiled_descriptor = ttnn.CBDescriptor(
        total_size=tiles_per_row * tile_size,
        core_ranges=core_grid,
        format_descriptors=[cb_beta_tiled_format],
    )
    cb_descriptors.append(cb_beta_tiled_descriptor)

    # CB 6: Scalars (epsilon, 1/W) - 1 tile
    cb_scalars_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_SCALARS,
        data_format=dtype,
        page_size=tile_size,
        tile=tile_descriptor,
    )
    cb_scalars_descriptor = ttnn.CBDescriptor(
        total_size=tile_size,  # 1 tile for scalars
        core_ranges=core_grid,
        format_descriptors=[cb_scalars_format],
    )
    cb_descriptors.append(cb_scalars_descriptor)

    # CB 7: Intermediate computation results
    cb_interm_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_INTERM,
        data_format=dtype,
        page_size=tile_size,
        tile=tile_descriptor,
    )
    cb_interm_descriptor = ttnn.CBDescriptor(
        total_size=tiles_per_row * tile_size,
        core_ranges=core_grid,
        format_descriptors=[cb_interm_format],
    )
    cb_descriptors.append(cb_interm_descriptor)

    # CB 16: Output in tile format
    cb_output_tiled_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_OUTPUT_TILED,
        data_format=dtype,
        page_size=tile_size,
        tile=tile_descriptor,
    )
    cb_output_tiled_descriptor = ttnn.CBDescriptor(
        total_size=tiles_per_row * tile_size,
        core_ranges=core_grid,
        format_descriptors=[cb_output_tiled_format],
    )
    cb_descriptors.append(cb_output_tiled_descriptor)

    # CB 17: Output row-major sticks (double buffer for pipelining)
    cb_output_rm_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_OUTPUT_RM,
        data_format=dtype,
        page_size=stick_size,
    )
    cb_output_rm_descriptor = ttnn.CBDescriptor(
        total_size=2 * stick_size,  # Double buffer
        core_ranges=core_grid,
        format_descriptors=[cb_output_rm_format],
    )
    cb_descriptors.append(cb_output_rm_descriptor)

    return cb_descriptors


def _create_reader_descriptor(core_grid, input_tensor, gamma_tensor, beta_tensor, sizes):
    """
    Create reader kernel descriptor for LayerNorm operation.

    The reader kernel reads:
    - Input tensor sticks (row-major) -> CB_INPUT_RM
    - Gamma tensor sticks (row-major, once) -> CB_GAMMA_RM
    - Beta tensor sticks (row-major, once) -> CB_BETA_RM

    Args:
        core_grid: CoreRangeSet specifying which cores to use
        input_tensor: Input tensor on device
        gamma_tensor: Gamma (scale) tensor on device
        beta_tensor: Beta (shift) tensor on device
        sizes: Dict from _calculate_sizes()

    Returns:
        KernelDescriptor for the reader kernel
    """
    import pathlib

    # Get kernel path
    kernel_dir = pathlib.Path(__file__).parent / "kernels"
    reader_kernel_path = str(kernel_dir / "reader.cpp")

    # Compile-time args:
    # [0] CB_INPUT_RM
    # [1] CB_GAMMA_RM
    # [2] CB_BETA_RM
    # [3] stick_size (page size for row-major data)
    # [4] num_rows (total rows to process)
    # [5..] TensorAccessorArgs for input
    # [..] TensorAccessorArgs for gamma
    # [..] TensorAccessorArgs for beta
    compile_time_args = [
        CB_INPUT_RM,
        CB_GAMMA_RM,
        CB_BETA_RM,
        sizes["stick_size"],
        sizes["num_rows"],
    ]

    # Append TensorAccessorArgs for each input tensor
    input_accessor = ttnn.TensorAccessorArgs(input_tensor)
    gamma_accessor = ttnn.TensorAccessorArgs(gamma_tensor)
    beta_accessor = ttnn.TensorAccessorArgs(beta_tensor)

    compile_time_args.extend(input_accessor.get_compile_time_args())
    compile_time_args.extend(gamma_accessor.get_compile_time_args())
    compile_time_args.extend(beta_accessor.get_compile_time_args())

    # Runtime args (per core):
    # [0] input buffer address
    # [1] gamma buffer address
    # [2] beta buffer address
    # [3] start_stick_id for this core (0 for single core)
    runtime_args = ttnn.RuntimeArgs()
    runtime_args[0][0] = [
        input_tensor.buffer_address(),
        gamma_tensor.buffer_address(),
        beta_tensor.buffer_address(),
        0,  # start_stick_id (single core starts at 0)
    ]

    # Create kernel descriptor
    reader_descriptor = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    return reader_descriptor


def _create_compute_descriptor(core_grid, sizes, epsilon):
    """
    Create compute kernel descriptor for LayerNorm operation.

    The compute kernel performs:
    - Tilization of input/gamma/beta
    - Mean and variance computation
    - rsqrt(variance + epsilon)
    - Standardization and affine transformation
    - Untilization of output

    Args:
        core_grid: CoreRangeSet specifying which cores to use
        sizes: Dict from _calculate_sizes()
        epsilon: Small constant for numerical stability

    Returns:
        KernelDescriptor for the compute kernel
    """
    import pathlib
    import struct

    # Get kernel path
    kernel_dir = pathlib.Path(__file__).parent / "kernels"
    compute_kernel_path = str(kernel_dir / "compute.cpp")

    # Pack epsilon as uint32 (bfloat16 representation packed in uint32)
    # Convert float to bfloat16 and pack as uint32
    def float_to_uint32(f):
        """Pack a float32 value into uint32 representation."""
        return struct.unpack("I", struct.pack("f", f))[0]

    epsilon_packed = float_to_uint32(epsilon)

    # Compile-time args:
    # [0] CB_INPUT_RM - input row-major CB
    # [1] CB_INPUT_TILED - input tiled CB
    # [2] CB_GAMMA_RM - gamma row-major CB
    # [3] CB_GAMMA_TILED - gamma tiled CB
    # [4] CB_BETA_RM - beta row-major CB
    # [5] CB_BETA_TILED - beta tiled CB
    # [6] CB_SCALARS - scalar values CB
    # [7] CB_INTERM - intermediate results CB
    # [8] CB_OUTPUT_TILED - output tiled CB
    # [9] CB_OUTPUT_RM - output row-major CB
    # [10] tiles_per_row - number of tiles per normalization row
    # [11] num_rows - total number of rows to normalize
    # [12] W - final dimension (normalization dimension)
    compile_time_args = [
        CB_INPUT_RM,
        CB_INPUT_TILED,
        CB_GAMMA_RM,
        CB_GAMMA_TILED,
        CB_BETA_RM,
        CB_BETA_TILED,
        CB_SCALARS,
        CB_INTERM,
        CB_OUTPUT_TILED,
        CB_OUTPUT_RM,
        sizes["tiles_per_row"],
        sizes["num_rows"],
        sizes["W"],
    ]

    # Runtime args (per core):
    # [0] epsilon (packed as uint32)
    runtime_args = ttnn.RuntimeArgs()
    runtime_args[0][0] = [
        epsilon_packed,
    ]

    # Create compute config with appropriate settings for LayerNorm
    # - Use HiFi4 math fidelity for better precision in normalization
    # - Enable FP32 dest accumulation for numerical stability in reduce operations
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
    )

    # Create kernel descriptor
    compute_descriptor = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=compute_config,
    )

    return compute_descriptor


class LayerNormSingleCore:
    """
    Single-core LayerNorm implementation using generic_op infrastructure.

    Performs row-wise layer normalization: output = (x - mean) * rsqrt(var + eps) * gamma + beta
    """

    @staticmethod
    def golden(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-6):
        """
        Golden reference implementation using PyTorch.

        Args:
            input_tensor: Input tensor (torch.Tensor)
            gamma_tensor: Scale parameter tensor (torch.Tensor)
            beta_tensor: Shift parameter tensor (torch.Tensor)
            epsilon: Small constant for numerical stability

        Returns:
            torch.Tensor: Normalized output
        """
        # Compute mean along the last dimension (row-wise)
        mean = input_tensor.mean(dim=-1, keepdim=True)

        # Compute variance along the last dimension (unbiased=False for population variance)
        var = input_tensor.var(dim=-1, unbiased=False, keepdim=True)

        # Standardize: (x - mean) / sqrt(var + epsilon)
        normalized = (input_tensor - mean) / torch.sqrt(var + epsilon)

        # Apply affine transformation: gamma * normalized + beta
        output = normalized * gamma_tensor + beta_tensor

        return output

    @staticmethod
    def op(input_tensor, gamma_tensor, beta_tensor, output_tensor, epsilon=1e-6):
        """
        Execute LayerNorm on device using generic_op.

        Args:
            input_tensor: Input tensor on device (row-major, interleaved, DRAM)
            gamma_tensor: Scale parameter tensor on device
            beta_tensor: Shift parameter tensor on device
            output_tensor: Pre-allocated output tensor on device
            epsilon: Small constant for numerical stability

        Returns:
            output_tensor with results
        """
        raise NotImplementedError("Op implementation pending - Step 1.5")
