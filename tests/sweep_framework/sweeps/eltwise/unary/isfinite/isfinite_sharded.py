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
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt, gen_rand_inf

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 120
Y, X = get_device_grid_size()

random.seed(0)


def gen_sharded_spec(num_shapes, sharding_strategy, y, x, sanitize_args=True):
    assert sharding_strategy in ["block", "width", "height"]

    shard_orientation_list = ["col_major", "row_major"]
    tensor_hw_as_shard_shape_list = [True, False]

    for shard_orientation, tensor_hw_as_shard_shape in itertools.product(
        shard_orientation_list, tensor_hw_as_shard_shape_list
    ):
        if sharding_strategy == "block":
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
        elif sharding_strategy == "width":
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

        for input_shape in input_shape_list:
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
        "input_spec": list(gen_sharded_spec(4, "block", Y, X))
        + list(gen_sharded_spec(4, "height", Y, X))
        + list(gen_sharded_spec(4, "width", Y, X)),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
    "xfail": {
        "input_spec": list(gen_sharded_spec(16, "block", Y, X, sanitize_args=False))
        + list(gen_sharded_spec(16, "height", Y, X, sanitize_args=False))
        + list(gen_sharded_spec(16, "width", Y, X, sanitize_args=False)),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
    "test": {
        "input_spec": [
            {
                "input_shape": [2048, 32],
                "sharding_strategy": "height",
                "shard_orientation": "row_major",
                "tensor_hw_as_shard_shape": False,
            }
        ],
        "input_a_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.TILE_LAYOUT],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_shape, sharding_strategy, _, _ = test_vector["input_spec"].values()
    pre_sharded_height = math.prod(input_shape[:-1])
    pre_sharded_width = input_shape[-1]

    if sharding_strategy == "block":
        if pre_sharded_height % Y != 0:
            return (
                True,
                "Prod of all dimensions except the innermost must be divisible by the y coordinate of coregrid when using block sharding",
            )
        if pre_sharded_width % X != 0:
            return (
                True,
                "Innermost dimension must be divisible by the x coordinate of coregrid when using block sharding",
            )
        if (pre_sharded_height // Y) // 32 <= 0:
            return True, "Shard height must be greater than 32"
        if (pre_sharded_width // X) // 32 <= 0:
            return True, "Shard wdith must be greater than 32"

    if sharding_strategy == "width":
        if pre_sharded_width % (Y * X) != 0:
            return True, "Last dimension must be divisible by a total number of cores when using width sharding"
        if (pre_sharded_width // (Y * X)) // 32 <= 0:
            return True, "Shard wdith must be greater than 32"

    if sharding_strategy == "height":
        if pre_sharded_height % (Y * X) != 0:
            return (
                True,
                "Prod of all dimensions except the innermost must be divisible by a total number of cores when using height sharding",
            )
        if (pre_sharded_height // (Y * X)) // 32 <= 0:
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
    if shard_orientation == "col_major":
        shard_orientation = ttnn.ShardOrientation.COL_MAJOR
    else:
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    if sharding_strategy == "block":
        sharding_strategy = ttnn.ShardStrategy.BLOCK
    elif sharding_strategy == "width":
        sharding_strategy = ttnn.ShardStrategy.WIDTH
    else:
        sharding_strategy = ttnn.ShardStrategy.HEIGHT

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_rand_inf(input_shape, low=-100, high=100)
    torch_output_tensor = torch.isfinite(torch_input_tensor_a)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=ttnn.CoreGrid(y=Y, x=X),
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
    output_tensor = ttnn.isfinite(input_tensor_a, memory_config=sharded_config)
    e2e_perf = stop_measuring_time(start_time)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    # print(pcc)
    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]


# Run sweeps locally
from tests.sweep_framework.framework.permutations import *

start_time = start_measuring_time()
for suite in parameters.keys():
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    suite_vectors = list(permutations(parameters[suite]))
    print(len(suite_vectors))
    for vector in suite_vectors:
        invalidate_res = invalidate_vector(vector)
        if invalidate_res[0]:
            print(f"Invalidated: {invalidate_res[1]}")
            continue
        try:
            passed, _ = run(**vector, device=device)
            if passed[0] != True:
                print(passed)
        except Exception as e:
            print(e)

    ttnn.close_device(device)

e2e_perf = stop_measuring_time(start_time)
print(f"time {e2e_perf / 1000000000}s")
