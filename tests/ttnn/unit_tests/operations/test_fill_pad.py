# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, run_for_wormhole_b0


def create_rm_tensor(height, width):
    """
    Creates a 2D tensor of datatype float32 where each element is row * width + col.
    """
    # return torch.arange(height).unsqueeze(1).float() * width + torch.arange(width).float()
    return torch.arange(height).unsqueeze(1) * width + torch.arange(width)


def create_tile_tensor(height, width, tile_size):
    """
    Creates a 2D tensor where each element represents the tile it belongs to.

    Parameters:
        height (int): The height of the tensor (number of rows).
        width (int): The width of the tensor (number of columns).
        tile_size (int): The size of each square tile (tile_size x tile_size).

    Returns:
        torch.Tensor: A 2D tensor with tile indices.
    """
    # Calculate the number of tiles in each dimension
    tiles_per_row = (width + tile_size - 1) // tile_size
    tiles_per_col = (height + tile_size - 1) // tile_size

    # Create row and column indices
    row_indices = torch.arange(height).unsqueeze(1) // tile_size
    col_indices = torch.arange(width).unsqueeze(0) // tile_size

    # Calculate tile indices
    tile_tensor = row_indices * tiles_per_row + col_indices
    print(tile_tensor.shape)
    return tile_tensor


def create_padded_tiled_tensor(height, width, tile_size, fill_value):
    """
    Creates a 2D tensor of height x width with random values, creates another tensor of
    (height rounded up to the nearest multiple of tile_size) x (width rounded up to the nearest multiple of tile_size) with the same random values,
    and pads the original tensor with a fill_value to match the size of the second tensor.

    Parameters:
        height (int): The height of the tensor (number of rows).
        width (int): The width of the tensor (number of columns).
        tile_size (int): The size of each square tile (tile_size x tile_size).

    Returns:
        torch.Tensor: A 2D tensor with tile indices.
    """
    # Create a tensor with random values
    tensor = torch_random((height, width), 0, 10, dtype=torch.int32)
    # tensor = create_tile_tensor(height, width, tile_size)
    # tensor = torch_random((height, width), -15.0, 15.0, dtype=torch.float32)
    # tensor = torch.zeros((height, width), dtype=torch.int32)
    # tensor = create_rm_tensor(height, width)

    # Calculate the size of the padded tensor
    padded_height = (height + tile_size - 1) // tile_size * tile_size
    padded_width = (width + tile_size - 1) // tile_size * tile_size

    # Create a padded tensor padded with fill_value
    padded_tensor = torch.full((padded_height, padded_width), fill_value, dtype=torch.int32)

    # Copy the original tensor to the padded tensor
    padded_tensor[:height, :width] = tensor

    return tensor, padded_tensor


import pytest
import torch
import ttnn

torch.set_printoptions(threshold=torch.inf)


@pytest.mark.parametrize("height", [40])
@pytest.mark.parametrize("width", [40])
@pytest.mark.parametrize("fill_value", [1])
@pytest.mark.parametrize("dtype", [ttnn.uint32])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_fill_pad(
    device,
    height,
    width,
    fill_value,
    dtype,
    input_mem_config,
    output_mem_config,
):
    torch.manual_seed(1234)
    torch_input_tensor, padded_torch_tensor = create_padded_tiled_tensor(height, width, 32, fill_value)
    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT),
        device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.fill_pad(input_tensor, fill_value, memory_config=output_mem_config)
    # output_tensor.reshape(padded_torch_tensor.shape)
    ttnn.set_printoptions(profile="full")
    # print(output_tensor)
    padded_torch_output_tensor = ttnn.from_device(output_tensor).to_torch()
    print(padded_torch_output_tensor.shape)
    # print(ttnn.from_device(output_tensor).to_torch())
    output_tensor = ttnn.to_torch(output_tensor)
    # print the output tensor
    print(output_tensor)
    print(torch_input_tensor)

    assert_with_pcc(padded_torch_tensor, padded_torch_output_tensor)
