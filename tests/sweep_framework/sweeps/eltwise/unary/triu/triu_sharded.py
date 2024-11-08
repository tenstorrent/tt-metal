# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import itertools

import torch
import random
import ttnn
import math
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape_rm, get_device_grid_size
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30
Y, X = get_device_grid_size()

random.seed(0)


def gen_sharded_spec(num_shapes, sharding_strategy, y, x, sanitize_args=True):
    shard_orientation_list = [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR]
    tensor_hw_as_shard_shape_list = [True, False]

    if sharding_strategy == ttnn.ShardStrategy.BLOCK:
        if not sanitize_args:
            interval_1 = 1
            interval_2 = 2
        else:
            interval_1 = 32 * y
            interval_2 = 32 * x

        input_shape_list = (
            gen_shapes([1, 1, 32 * y, 32 * x], [6, 12, 512, 512], [1, 1, interval_1, interval_2], num_shapes)
            + gen_shapes([1, 32 * y, 32 * x], [12, 512, 512], [1, interval_1, interval_2], num_shapes)
            + gen_shapes([32 * y, 32 * x], [512, 512], [interval_1, interval_2], num_shapes)
        )
    elif sharding_strategy == ttnn.ShardStrategy.WIDTH:
        if not sanitize_args:
            interval = 1
        else:
            interval = 32 * x * y
        input_shape_list = (
            gen_shapes([1, 1, 32, 32 * x * y], [4, 6, 64, 32 * x * y], [1, 1, 32, interval], num_shapes)
            + gen_shapes([1, 32, 32 * x * y], [6, 64, 32 * x * y], [1, 32, interval], num_shapes)
            + gen_shapes([32, 32 * x * y], [64, 32 * x * y], [32, interval], num_shapes)
        )
    else:
        if not sanitize_args:
            interval = 1
        else:
            interval = 32 * x * y
        input_shape_list = (
            gen_shapes([1, 1, 32 * x * y, 32], [4, 6, 32 * x * y, 64], [1, 1, interval, 32], num_shapes)
            + gen_shapes([1, 32 * x * y, 32], [6, 32 * x * y, 64], [1, interval, 32], num_shapes)
            + gen_shapes([32 * x * y, 32], [32 * x * y, 64], [interval, 32], num_shapes)
        )

    for input_shape, shard_orientation, tensor_hw_as_shard_shape in itertools.product(
        input_shape_list, shard_orientation_list, tensor_hw_as_shard_shape_list
    ):
        yield {
            "input_shape": input_shape,
            "sharding_strategy": sharding_strategy,
            "shard_orientation": shard_orientation,
            "tensor_hw_as_shard_shape": tensor_hw_as_shard_shape,
        }


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_spec": list(gen_sharded_spec(4, ttnn.ShardStrategy.BLOCK, Y, X))
        + list(gen_sharded_spec(4, ttnn.ShardStrategy.HEIGHT, Y, X))
        + list(gen_sharded_spec(4, ttnn.ShardStrategy.WIDTH, Y, X)),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
    "xfail": {
        "input_spec": list(gen_sharded_spec(16, ttnn.ShardStrategy.BLOCK, Y, X, sanitize_args=False))
        + list(gen_sharded_spec(16, ttnn.ShardStrategy.HEIGHT, Y, X, sanitize_args=False))
        + list(gen_sharded_spec(16, ttnn.ShardStrategy.WIDTH, Y, X, sanitize_args=False)),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_spec = test_vector["input_spec"]

    if input_spec["sharding_strategy"] == ttnn.ShardStrategy.BLOCK:
        if math.prod(input_spec["input_shape"][:-1]) % Y != 0:
            return (
                True,
                "Prod of all dimensions except the innermost must be divisible by the y coordinate of coregrid when using block sharding",
            )
        if input_spec["input_shape"][-1] % X != 0:
            return (
                True,
                "Innermost dimension must be divisible by the x coordinate of coregrid when using block sharding",
            )
        if (math.prod(input_spec["input_shape"][:-1]) // Y) // 32 <= 0:
            return True, "Shard height must be greater than 32"
        if (input_spec["input_shape"][-1] // X) // 32 <= 0:
            return True, "Shard wdith must be greater than 32"

    if input_spec["sharding_strategy"] == ttnn.ShardStrategy.WIDTH:
        if input_spec["input_shape"][-1] % (Y * X) != 0:
            return True, "Last dimension must be divisible by a total number of cores when using width sharding"
        if (input_spec["input_shape"][-1] // (Y * X)) // 32 <= 0:
            return True, "Shard wdith must be greater than 32"

    if input_spec["sharding_strategy"] == ttnn.ShardStrategy.HEIGHT:
        if math.prod(input_spec["input_shape"][:-1]) % (Y * X) != 0:
            return (
                True,
                "Prod of all dimensions except the innermost must be divisible by a total number of cores when using height sharding",
            )
        if (math.prod(input_spec["input_shape"][:-1]) // (Y * X)) // 32 <= 0:
            return True, "Shard heght must be greater than 32"

    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Input to eltwise binary must be tilized"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and input_spec["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is only supported on tiled layout"

    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_spec,
    input_a_dtype,
    input_layout,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    input_shape, sharding_strategy, shard_orientation, tensor_hw_as_shard_shape = input_spec.values()

    device_grid_size = ttnn.CoreGrid(y=Y, x=X)

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    low = -(input_shape[-2] - 2)
    high = input_shape[-1]
    diagonal = torch.randint(low, high, (1,)).item()

    torch_output_tensor = torch.triu(torch_input_tensor_a, diagonal)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=device_grid_size,
        strategy=sharding_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=sharded_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.triu(input_tensor_a, diagonal=diagonal, memory_config=sharded_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
