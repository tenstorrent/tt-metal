# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from .format_arg_mapping import format_dict
from .format_config import DataFormat


def tilize_block(input_tensor, dimensions, stimuli_format=DataFormat.Float16_b):
    if input_tensor.numel() != dimensions[0] * dimensions[1]:
        raise ValueError(
            f"Cannot reshape tensor of size {input_tensor.numel()} to shape {dimensions}."
        )

    input_reshaped = input_tensor.view(dimensions[0], dimensions[1])
    if input_reshaped.ndim != 2:
        raise ValueError(
            f"Expected a 2D tensor for tilize_block, got shape {input_tensor.shape}"
        )

    rows, cols = input_reshaped.shape
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(
            f"Input tensor dimensions must be divisible by 32. Got shape: {input_tensor.shape}"
        )

    # Calculate number of blocks in each dimension
    row_blocks = rows // 32
    col_blocks = cols // 32
    total_blocks = row_blocks * col_blocks

    blocked_tensor = input_reshaped.reshape(row_blocks, 32, col_blocks, 32)

    # Permute to get blocks in the right order: (row_blocks, col_blocks, 32, 32)
    blocked_tensor = blocked_tensor.permute(0, 2, 1, 3)

    # Reshape to get all blocks as sequential entities: (total_blocks, 32, 32)
    all_blocks = blocked_tensor.reshape(total_blocks, 32, 32)

    flat_blocks = all_blocks.reshape(total_blocks, -1)

    tilized_blocks = torch.stack(
        [tilize(block, stimuli_format=stimuli_format) for block in flat_blocks]
    )

    # Reshape tilized blocks back to original dimensions
    tilized_output = (
        tilized_blocks.flatten().reshape(rows, cols).to(format_dict[stimuli_format])
    )

    return tilized_output


def tilize(original_tensor, stimuli_format=DataFormat.Float16_b):

    if original_tensor.size(0) != 1024:
        raise ValueError("Input tensor must have 1024 elements.")

    matrix = original_tensor.view(32, 32)

    f0 = matrix[:16, :16]
    f1 = matrix[:16, 16:32]
    f2 = matrix[16:32, :16]
    f3 = matrix[16:32, 16:32]

    result = torch.cat((f0.reshape(-1), f1.reshape(-1), f2.reshape(-1), f3.reshape(-1)))

    return result.to(dtype=format_dict[stimuli_format])


def untilize(tilized_tensor, stimuli_format=DataFormat.Float16_b):

    if tilized_tensor.size(0) != 1024:
        raise ValueError(
            f"Input tensor must have 1024 elements. It has: {len(tilized_tensor)}"
        )

    tilized_tensor = tilized_tensor.view(-1)

    f0 = tilized_tensor[:256].view(16, 16)
    f1 = tilized_tensor[256:512].view(16, 16)
    f2 = tilized_tensor[512:768].view(16, 16)
    f3 = tilized_tensor[768:].view(16, 16)

    top = torch.cat((f0, f1), dim=1)
    bottom = torch.cat((f2, f3), dim=1)

    original_tensor = torch.cat((top, bottom), dim=0).view(1024)

    return original_tensor.to(dtype=format_dict[stimuli_format])


def untilize_block(
    input_tensor, stimuli_format=DataFormat.Float16_b, dimensions=[32, 32]
):
    """Optimized function to untilize blocks of data.

    Args:
        input_tensor: Input tensor to be untilized
        stimuli_format: Data format
        dimensions: Target dimensions for the output

    Returns:
        Untilized tensor with specified dimensions and data format
    """
    if input_tensor.numel() != dimensions[0] * dimensions[1]:
        raise ValueError(
            f"Cannot reshape tensor of size {input_tensor.numel()} to shape {dimensions}."
        )

    # Calculate number of 32x32 tiles
    rows, cols = dimensions
    row_blocks = rows // 32
    col_blocks = cols // 32
    total_blocks = row_blocks * col_blocks

    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(
            f"Dimensions must be divisible by 32. Got dimensions: {dimensions}"
        )

    # Reshape input to have one block per 1024 elements
    input_reshaped = input_tensor.reshape(total_blocks, 1024)

    untilized_blocks = torch.stack([untilize(block) for block in input_reshaped])

    output = untilized_blocks.reshape(row_blocks, col_blocks, 32, 32)

    # Then permute and reshape to get the final dimensions
    output = output.permute(0, 2, 1, 3).reshape(rows, cols)

    return output.to(dtype=format_dict[stimuli_format])
