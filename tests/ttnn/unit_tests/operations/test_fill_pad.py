# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, run_for_wormhole_b0


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


import pytest
import torch
import ttnn


@pytest.mark.parametrize("height", [30])
@pytest.mark.parametrize("width", [30])
@pytest.mark.parametrize("fill_value", [0, 1, 3])
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

    torch_input_tensor = torch.randn((batch_size, height, width), dtype=torch.float32)
    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=dtype), device, memory_config=input_mem_config
    )

    output_tensor = ttnn.fill_pad(input_tensor, fill_value, memory_config=output_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
