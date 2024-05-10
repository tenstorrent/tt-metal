# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random
import math

# parameters = {
#   "dtype": [ttnn.int32, ttnn.bfloat16, ttnn.bfloat8_b],
#   #    "height": [4, 8, 12, 16, 32, 64, 96, 128, 256, 512, 1024, 4096, 8192, 8196],
#   #    "width": [4, 8, 12, 16, 32, 64, 96, 128, 256, 512, 1024, 4096, 8192, 8196],
#   "height": [4, 8, 12, 16, 32, 64, 96, 128],
#   "width": [4, 8, 12, 16, 32, 64, 96, 128],
#   "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
#   "input_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
#   "input_num_cores_x": [1, 2, 4, 8],
#   "input_num_cores_y": [1, 2, 4, 8],
#   "input_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
#   "output_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
#   "output_num_cores_x": [1, 2, 4, 8],
#   "output_num_cores_y": [1, 2, 4, 8],
#   "output_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
# }

# parameters = {
#    "dtype": [ttnn.int32, ttnn.bfloat16, ttnn.bfloat8_b],
#    "height": [4, 16, 32],
#    "width": [4, 16, 32],
#    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
#    "input_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
#    "input_num_cores_x": [1, 8],
#    "input_num_cores_y": [1, 8],
#    "input_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
#    "output_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
#    "output_num_cores_x": [1, 4, 8],
#    "output_num_cores_y": [1, 4, 8],
#    "output_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
# }


parameters = {
    "dtype": [ttnn.bfloat16],
    "height": [32, 64],
    "width": [16, 64],
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "input_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    "input_num_cores_x": [1, 8],
    "input_num_cores_y": [1, 8],
    "input_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
    "output_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    "output_num_cores_x": [1, 8],
    "output_num_cores_y": [1, 8],
    "output_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
}


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def compute_lcm(x, y):
    # choose the greater number
    if x > y:
        greater = x
    else:
        greater = y

    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1

    return lcm


def roundTo32(x):
    if x < 32:
        return 32
    else:
        if x % 32 == 0:
            return x
        else:
            return ((x // 32) * 32) + 32


def run(
    dtype,
    height,
    width,
    layout,
    input_shard_orientation,
    input_num_cores_x,
    input_num_cores_y,
    input_shard_strategy,
    output_shard_orientation,
    output_num_cores_x,
    output_num_cores_y,
    output_shard_strategy,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_num_cores_height = 1
    input_num_cores_width = 1
    if input_shard_strategy == ttnn.ShardStrategy.HEIGHT:
        input_num_cores_height = input_num_cores_x * input_num_cores_y
    elif input_shard_strategy == ttnn.ShardStrategy.WIDTH:
        input_num_cores_width = input_num_cores_x * input_num_cores_y
    elif input_shard_strategy == ttnn.ShardStrategy.BLOCK:
        if input_shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
            input_num_cores_height = input_num_cores_y
            input_num_cores_width = input_num_cores_x
        else:
            input_num_cores_height = input_num_cores_x
            input_num_cores_width = input_num_cores_y

    output_num_cores_height = 1
    output_num_cores_width = 1
    if output_shard_strategy == ttnn.ShardStrategy.HEIGHT:
        output_num_cores_height = output_num_cores_x * output_num_cores_y
    elif output_shard_strategy == ttnn.ShardStrategy.WIDTH:
        output_num_cores_width = output_num_cores_x * output_num_cores_y
    elif output_shard_strategy == ttnn.ShardStrategy.BLOCK:
        if output_shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
            output_num_cores_height = output_num_cores_y
            output_num_cores_width = output_num_cores_x
        else:
            output_num_cores_height = output_num_cores_x
            output_num_cores_width = output_num_cores_y

    num_cores_height = max(input_num_cores_height, output_num_cores_height)
    num_cores_width = max(input_num_cores_width, output_num_cores_width)

    if layout == ttnn.TILE_LAYOUT:
        height = int(roundTo32(height))
        width = int(roundTo32(width))

    height = height * num_cores_height
    width = width * num_cores_width
    tensor_shape = [1, 1, int(height), int(width)]
    input_core_grid = ttnn.CoreGrid(y=input_num_cores_y, x=input_num_cores_x)
    output_core_grid = ttnn.CoreGrid(y=output_num_cores_y, x=output_num_cores_x)
    input_args = dict(
        core_grid=input_core_grid,
        strategy=input_shard_strategy,
        orientation=input_shard_orientation,
    )
    output_args = dict(
        core_grid=output_core_grid,
        strategy=output_shard_strategy,
        orientation=output_shard_orientation,
    )

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    interleaved_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, dtype=dtype, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_shard_memory_config = ttnn.create_sharded_memory_config(tensor_shape, **input_args)
    output_shard_memory_config = ttnn.create_sharded_memory_config(tensor_shape, **output_args)

    # interleaved_to_sharded
    sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)

    # reshard
    sharded_output_tensor = ttnn.to_memory_config(sharded_input_tensor, output_shard_memory_config)

    # sharded_to_interleaved
    interleaved_output_tensor = ttnn.to_memory_config(sharded_output_tensor, ttnn.DRAM_MEMORY_CONFIG)

    output = ttnn.to_torch(interleaved_output_tensor)

    return check_with_pcc(torch_input_tensor, output, 0.999)
