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
