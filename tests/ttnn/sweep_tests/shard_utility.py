# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn


def roundTo32(x):
    if x < 32:
        return 32
    else:
        if x % 32 == 0:
            return x
        else:
            return ((x // 32) * 32) + 32


def sharded_run(
    dtype,
    height,
    width,
    layout,
    input_shard_orientation,
    input_num_cores_x,
    input_num_cores_y,
    input_shard_strategy,
    interleaved_to_sharded,
    sharded_to_interleaved,
    reshard,
    device,
    output_shard_orientation=None,
    output_num_cores_x=None,
    output_num_cores_y=None,
    output_shard_strategy=None,
):
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

    input_core_grid = ttnn.CoreGrid(y=input_num_cores_y, x=input_num_cores_x)
    input_args = dict(
        core_grid=input_core_grid,
        strategy=input_shard_strategy,
        orientation=input_shard_orientation,
    )
    num_cores_height = input_num_cores_height
    num_cores_width = input_num_cores_width

    if reshard:
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
        output_core_grid = ttnn.CoreGrid(y=output_num_cores_y, x=output_num_cores_x)
        output_args = dict(
            core_grid=output_core_grid,
            strategy=output_shard_strategy,
            orientation=output_shard_orientation,
        )

    if layout == ttnn.TILE_LAYOUT:
        height = int(roundTo32(height))
        width = int(roundTo32(width))

    height = height * num_cores_height
    width = width * num_cores_width
    tensor_shape = [1, 1, int(height), int(width)]

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_shard_memory_config = ttnn.create_sharded_memory_config(tensor_shape, **input_args)

    # interleaved_to_sharded
    if interleaved_to_sharded:
        interleaved_input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=layout, dtype=dtype, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)
    else:
        sharded_input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=layout, dtype=dtype, device=device, memory_config=input_shard_memory_config
        )

    # reshard
    if reshard:
        output_shard_memory_config = ttnn.create_sharded_memory_config(tensor_shape, **output_args)
        sharded_output_tensor = ttnn.to_memory_config(sharded_input_tensor, output_shard_memory_config)
    else:
        sharded_output_tensor = sharded_input_tensor

    # sharded_to_interleaved
    if sharded_to_interleaved:
        interleaved_output_tensor = ttnn.to_memory_config(sharded_output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.to_torch(interleaved_output_tensor)
    else:
        output = ttnn.to_torch(sharded_output_tensor)
    return torch_input_tensor, output
