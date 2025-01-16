# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, run_for_wormhole_b0


def create_nd_padded_tiled_tensor(shape, tile_size, fill_value, dtype):
    """
    Creates a tensor with shape `shape` with random values, and another tensor with the same values,
    but with the last 2 dimensions padded to the nearest multiple of tile_size using fill_value.

    Args:
        shape (tuple): Shape of the original tensor.
        tile_size (int): Size to which the last two dimensions will be padded.
        fill_value (float or int): Value used for padding.
        dtype (torch.dtype): Data type of the tensors.

    Returns:
        tuple: A tuple containing the original tensor and the padded tensor.
    """
    # Create a tensor with random values
    if dtype == torch.float32:
        tensor = torch_random(shape, -15.0, 15.0, dtype=dtype)
    else:
        tensor = torch.randint(0, 10, shape, dtype=dtype)

    # Calculate the padded sizes for the last two dimensions
    padded_shape = list(shape)
    padded_shape[-2] = (padded_shape[-2] + tile_size - 1) // tile_size * tile_size
    padded_shape[-1] = (padded_shape[-1] + tile_size - 1) // tile_size * tile_size

    # Create a padded tensor filled with fill_value
    padded_tensor = torch.full(padded_shape, fill_value, dtype=dtype)

    # Copy the original tensor into the padded tensor
    padded_tensor[..., : shape[-2], : shape[-1]] = tensor

    return tensor, padded_tensor


import pytest
import torch
import ttnn

ttnn_dtype_to_torch_dtype = {
    ttnn.uint32: torch.int32,
    ttnn.bfloat16: torch.float32,
}


# @pytest.mark.parametrize("shape", [(2, 32, 300, 256)])
@pytest.mark.parametrize(
    "shape",
    [
        # 2D shapes with edge cases for fill_pad
        (32, 32),
        (32, 1),
        (31, 32),
        (1, 32),
        # Same as above, but with larger dimensions for multiple pages of writing
        (64, 63),
        (64, 1),
        (63, 64),
        (1, 64),
        # Same as above, but with 3D where higher dim is not 1
        (5, 32, 31),
        (5, 32, 1),
        (5, 31, 32),
        (5, 1, 32),
        # Same as above, but using a prime number like 13 for the highest dimension
        (13, 32, 31),
        (13, 32, 1),
        (13, 31, 32),
        (13, 1, 32),
        (29, 13, 32, 31),
        (11, 13, 32, 1),
        (7, 13, 31, 32),
        (1, 13, 1, 32),
        (1, 1, 1, 13),
        (1, 2, 7, 73),
        (1, 2, 3, 5, 89),
        (1, 2, 3, 2, 7, 97),
        (1, 2, 3, 2, 1, 13, 253),
        (1, 2, 3, 2, 1, 2, 7, 1707),
    ],
)
@pytest.mark.parametrize("fill_value", [1])
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_fill_pad(
    device,
    shape,
    fill_value,
    dtype,
    input_mem_config,
    output_mem_config,
):
    torch.manual_seed(1234)
    torch_input_tensor, padded_torch_tensor = create_nd_padded_tiled_tensor(
        shape, 32, fill_value, ttnn_dtype_to_torch_dtype[dtype]
    )
    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.fill_pad(input_tensor, fill_value, memory_config=output_mem_config)
    padded_torch_output_tensor = ttnn.from_device(output_tensor).to_torch()

    assert_with_pcc(padded_torch_tensor, padded_torch_output_tensor)
