# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import itertools
import random
import math
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import _gen_reshape_args_from_volume


def gen_sharded_spec_unary(num_shapes, max_tensor_size_per_core=32 * 1024, layouts=["TILE_LAYOUT", "ROW_MAJOR_LAYOUT"]):
    # device.compute_with_storage_grid_size()
    Y = 8
    X = 8

    # ["BLOCK", "WIDTH", "HEIGHT", "tensor_wh"]
    sharding_strategy_list = ["BLOCK", "WIDTH", "HEIGHT", "tensor_wh"]
    shard_orientation_list = ["COL_MAJOR", "ROW_MAJOR"]
    spec_list = []

    for sharding_strategy, shard_orientation, rank, layout in itertools.product(
        sharding_strategy_list, shard_orientation_list, [4, 3, 2], layouts
    ):
        if sharding_strategy == "tensor_wh":
            tensor_hw_as_shard_shape = True
            sharding_strategy = "BLOCK"
        else:
            tensor_hw_as_shard_shape = False

        for _ in range(num_shapes):
            x = random.randint(1, X)
            y = random.randint(1, Y)
            max_tensor_size = max_tensor_size_per_core * x * y

            if tensor_hw_as_shard_shape:
                if layout == "TILE_LAYOUT":
                    min_tensor_height = 32
                    min_tensor_width = 32
                    max_tensor_height = int(math.sqrt(max_tensor_size_per_core))
                    max_tensor_width = int(math.sqrt(max_tensor_size_per_core))
                    tensor_height = random.randrange(min_tensor_height, max_tensor_height + 1, 32)
                    tensor_width = random.randrange(min_tensor_width, max_tensor_width + 1, 32)
                    input_shape = [tensor_height, tensor_width]
                else:
                    shard_size = random.randrange(1024, max_tensor_size_per_core + 1, 1024)
                    tensor_width = random.randrange(16, shard_size // 2 + 1, 16)
                    tensor_height = shard_size // tensor_width
                    input_shape = [tensor_height, tensor_width]

                if rank != 2:
                    rest_volume = random.randint(1, max_tensor_size // math.prod(input_shape))
                    rest_dims = random.choice(_gen_reshape_args_from_volume(rest_volume, step=1, out_dims=rank - 2))
                    rest_dims = list(rest_dims["reshape_dims"])
                    input_shape = rest_dims + input_shape

            elif sharding_strategy == "BLOCK":
                min_shard_size_y = 32 * y
                min_shard_size_x = 32 * x

                rest_volume = random.randint(1, max_tensor_size // (min_shard_size_x * min_shard_size_y))
                physical_shape = random.choice(_gen_reshape_args_from_volume(rest_volume, step=1, out_dims=2))
                physical_shape = list(physical_shape["reshape_dims"])
                physical_shape[1] *= min_shard_size_y
                physical_shape[0] *= min_shard_size_x

                if shard_orientation == "ROW_MAJOR":
                    tmp = physical_shape[-2]
                    physical_shape[-2] = physical_shape[-1]
                    physical_shape[-1] = tmp

                input_shape = random.choice(_gen_reshape_args_from_volume(physical_shape[0], step=1, out_dims=rank - 1))
                input_shape = list(input_shape["reshape_dims"])
                input_shape.append(physical_shape[1])

            elif sharding_strategy == "WIDTH" or sharding_strategy == "HEIGHT":
                # if shard_width % total_cores != 0: raise RuntimeError("Invalid sharding core_grid")
                # Shard Size must be multiple of input_tile_size

                if layout == "TILE_LAYOUT":
                    # In shard mode ShardMode::PHYSICAL, physical shard shape {12, 13312} is not compatible with alignment Alignment([32, 32])!
                    min_shard_size_x = 32
                    min_shard_size_y = 32 * x * y
                else:  # if layout == "ROW_MAJOR_LAYOUT":
                    # Shard Size must be multiple of input_tile_size
                    # Shard width should be multiple of 16 to satisfy L1 alignment
                    mul_32_y = random.choice([16, 32, 64, 128, 256, 512, 1024])
                    mul_32_x = 1024 // mul_32_y

                    if sharding_strategy == "HEIGHT":
                        # Shard width should be multiple of 16 to satisfy L1 alignment
                        while mul_32_x % 16 != 0:
                            mul_32_x *= 2
                            mul_32_y //= 2

                    min_shard_size_x = mul_32_x
                    min_shard_size_y = mul_32_y * x * y

                rest_volume = random.randint(1, max_tensor_size // (min_shard_size_x * min_shard_size_y))
                input_shape = random.choice(_gen_reshape_args_from_volume(rest_volume, step=1, out_dims=rank))
                input_shape = list(input_shape["reshape_dims"])
                input_shape[-2] = input_shape[-2] * min_shard_size_x
                input_shape[-1] = input_shape[-1] * min_shard_size_y

                if sharding_strategy == "HEIGHT":
                    tmp = input_shape[-2]
                    input_shape[-2] = input_shape[-1]
                    input_shape[-1] = tmp

                # print(input_shape)

            spec_list.append(
                {
                    "input_shape": input_shape,
                    "X": x,
                    "Y": y,
                    "sharding_strategy": sharding_strategy,
                    "shard_orientation": shard_orientation,
                    "tensor_hw_as_shard_shape": tensor_hw_as_shard_shape,
                    "input_layout": layout,
                    "shard_height_mul_of_32": False,
                }
            )

    return spec_list


def parse_sharding_spec(input_spec):
    input_shape = input_spec["input_shape"]
    X = input_spec["X"]
    Y = input_spec["Y"]
    sharding_strategy = input_spec["sharding_strategy"]
    shard_orientation = input_spec["shard_orientation"]
    tensor_hw_as_shard_shape = input_spec["tensor_hw_as_shard_shape"]
    input_layout = input_spec["input_layout"]
    shard_height_mul_of_32 = input_spec["shard_height_mul_of_32"]

    if sharding_strategy == "HEIGHT":
        sharding_strategy = ttnn.ShardStrategy.HEIGHT
    elif sharding_strategy == "WIDTH":
        sharding_strategy = ttnn.ShardStrategy.WIDTH
    else:  # sharding_strategy == "BLOCK":
        sharding_strategy = ttnn.ShardStrategy.BLOCK

    if shard_orientation == "COL_MAJOR":
        shard_orientation = ttnn.ShardOrientation.COL_MAJOR
    else:
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    if input_layout == "TILE_LAYOUT":
        input_layout = ttnn.TILE_LAYOUT
    else:
        input_layout = ttnn.ROW_MAJOR_LAYOUT

    return (
        input_shape,
        ttnn.CoreGrid(y=Y, x=X),
        sharding_strategy,
        shard_orientation,
        tensor_hw_as_shard_shape,
        input_layout,
        shard_height_mul_of_32,
    )


def invalidate_vector_sharding(input_spec):
    input_shape, X, Y, _, shard_orientation, tensor_hw_as_shard_shape, input_layout, _ = input_spec.values()
    pre_sharded_height = math.prod(input_shape[:-1])
    pre_sharded_width = input_shape[-1]

    if tensor_hw_as_shard_shape:
        if input_shape[-2] % 32 != 0 or input_shape[-1] % 32 != 0:
            return (
                True,
                "Last two dimensions must be multiples of tile size when using tensor heght and width as shard shape",
            )

    if (
        input_layout == "ROW_MAJOR_LAYOUT"
        and shard_orientation == "COL_MAJOR"
        and (input_shape[-1] % input_shape[-2] != 0)
    ):
        return True, "Physical size <width, height> must be a multuple of page size <1, width>"

    return False, None
