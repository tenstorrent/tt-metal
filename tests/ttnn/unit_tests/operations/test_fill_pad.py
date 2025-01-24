# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import math
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
        (1, 16),
        (16, 1),
        (1, 17),
        (17, 1),
        (16, 16),
        (17, 17),
        (31, 31),
        (33, 33),
        (65, 65),
        (1, 2, 3, 2, 1, 2, 97, 97),
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
    print(padded_torch_tensor.shape)
    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.fill_implicit_tile_padding(input_tensor, fill_value, memory_config=output_mem_config)
    padded_torch_output_tensor = ttnn.from_device(output_tensor).to_torch()

    assert_with_pcc(padded_torch_tensor, padded_torch_output_tensor)


@pytest.mark.parametrize(
    "shape",
    [
        # 2D shapes with edge cases for fill_pad
        # (1, 16),
        # (16, 1),
        # (1, 17),
        # (17, 1),
        # (16, 16),
        # (17, 17),
        # (31, 31),
        # (33, 33),
        # (65, 65),
        (1, 2, 3, 2, 1, 2, 97, 97),
    ],
)
@pytest.mark.parametrize(
    "shard_scheme", [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_fill_pad_sharded(device, shape, shard_scheme, dtype):
    torch.manual_seed(1234)
    torch_input_tensor, padded_torch_tensor = create_nd_padded_tiled_tensor(
        shape, 32, 1, ttnn_dtype_to_torch_dtype[dtype]
    )

    num_cores_x = 8
    num_cores_y = 7
    num_cores = num_cores_x * num_cores_y
    shard_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))]
    )

    print(padded_torch_tensor.shape)
    tiles_per_2d = padded_torch_tensor.shape[-2] * padded_torch_tensor.shape[-1] / (32 * 32)
    num_2d = 1
    for i in range(len(padded_torch_tensor.shape) - 2):
        num_2d *= padded_torch_tensor.shape[i]
    print(num_2d)
    num_tiles = tiles_per_2d * num_2d
    print(num_tiles)
    # tiles per core must make sure to cover all tiles so div up
    tiles_per_core = math.ceil(num_tiles / num_cores)

    shard_shape = [32, 32]

    if shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        shard_shape = (32, 32 * tiles_per_core)
    elif shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_shape = (32 * tiles_per_core, 32)
    else:
        shard_shape = (math.ceil(math.sqrt(tiles_per_core)), math.ceil(math.sqrt(tiles_per_core)))

    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        shard_scheme,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        device,
        memory_config=output_mem_config,
    )

    print(input_tensor.memory_config().memory_layout)

    output_tensor = ttnn.fill_implicit_tile_padding(input_tensor, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    padded_torch_output_tensor = ttnn.from_device(output_tensor).to_torch()

    assert_with_pcc(padded_torch_tensor, padded_torch_output_tensor)
