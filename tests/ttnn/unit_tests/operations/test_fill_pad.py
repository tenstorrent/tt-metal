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


@pytest.mark.parametrize("shape", (32, 32))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_embedding(
    device,
    shape,
    dtype,
    input_mem_config,
    output_mem_config,
    layout,
):
    torch.manual_seed(1234)

    torch_input_tensor = torch.randint(0, vocabulary_size - 1, (batch_size, sentence_size))
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=input_mem_config)
    weights = ttnn.to_device(ttnn.from_torch(torch_weights, dtype=dtype), device, memory_config=input_mem_config)

    output_tensor = ttnn.embedding(input_tensor, weights, memory_config=output_mem_config, layout=layout)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
