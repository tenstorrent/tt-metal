# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import itertools

import torch
import random
import ttnn
import math
from tests.sweep_framework.sweep_utils.utils import (
    gen_shapes,
    sanitize_shape_rm,
    get_device_grid_size,
    get_sharded_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import (
    gen_func_with_cast_tt,
    _gen_reshape_args_from_volume,
    _get_factors,
)

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


TIMEOUT = 120

Y, X = get_device_grid_size()

random.seed(0)


def gen_sharded_spec(
    num_shapes, num_core_samples, shard_orientation, sharding_strategy, max_tensor_size_per_core=480 * 480
):
    assert sharding_strategy in ["block", "width", "height", "tensor_hw"]

    assert shard_orientation in ["col_major", "row_major"]

    for i in range(num_core_samples):
        y = random.randint(1, Y)
        x = random.randint(1, X)
        max_tensor_size = y * x * max_tensor_size_per_core
        for j in range(num_shapes):
            for rank in [2, 3, 4]:
                if sharding_strategy == "tensor_hw":
                    min_tensor_height = 32
                    min_tensor_width = 32
                    mul_height = random.randint(1, 10)
                    mul_width = random.randint(1, 10)
                    tensor_height = min_tensor_height * mul_height
                    tensor_width = min_tensor_width * mul_width
                    input_shape = [tensor_height, tensor_width]
                    if rank != 2:
                        rest_volume = random.randint(1, max_tensor_size // (tensor_height * tensor_width))
                        rest_dims = random.choice(_gen_reshape_args_from_volume(rest_volume, step=1, out_dims=rank - 2))
                        rest_dims = list(rest_dims["reshape_dims"])
                        input_shape = rest_dims + input_shape

                elif sharding_strategy == "block":
                    min_pre_sharded_height = 32 * y
                    min_pre_sharded_width = 32 * x

                    mul_1 = random.randint(1, y * 2)
                    mul_2 = random.randint(1, x * 2)

                    if shard_orientation == "row_major":
                        pre_sharded_width = mul_1 * min_pre_sharded_width
                        pre_sharded_height = mul_2 * min_pre_sharded_height
                    else:
                        pre_sharded_width = mul_1 * min_pre_sharded_height
                        pre_sharded_height = mul_2 * min_pre_sharded_width

                    input_shape = random.choice(
                        _gen_reshape_args_from_volume(pre_sharded_height, step=1, out_dims=rank - 1)
                    )
                    input_shape = list(input_shape["reshape_dims"])
                    input_shape.append(pre_sharded_width)

                elif sharding_strategy == "height":
                    min_pre_sharded_height = 32 * y * x
                    min_pre_sharded_width = 32

                    mul_1 = random.randint(1, 16)

                    pre_sharded_width = mul_1 * min_pre_sharded_width
                    pre_sharded_height = random.randrange(
                        min_pre_sharded_height, max_tensor_size // pre_sharded_width + 1, 32 * y * x
                    )

                    input_shape = random.choice(
                        _gen_reshape_args_from_volume(pre_sharded_height, step=1, out_dims=rank - 1)
                    )
                    input_shape = list(input_shape["reshape_dims"])
                    input_shape.append(pre_sharded_width)

                else:
                    min_pre_sharded_height = 32
                    min_pre_sharded_width = 32 * y * x

                    mul_1 = random.randint(1, 16)

                    pre_sharded_height = mul_1 * min_pre_sharded_height
                    pre_sharded_width = random.randrange(
                        min_pre_sharded_width, max_tensor_size // pre_sharded_height + 1, 32 * y * x
                    )

                    input_shape = random.choice(
                        _gen_reshape_args_from_volume(pre_sharded_height, step=1, out_dims=rank - 1)
                    )
                    input_shape = list(input_shape["reshape_dims"])
                    input_shape.append(pre_sharded_width)

                yield {
                    "input_shape": input_shape,
                    "core_size": (y, x),
                    "sharding_strategy": sharding_strategy,
                    "sharding_orientation": shard_orientation,
                }


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "xfail": {
        "input_spec":  # list(gen_sharded_spec(16, 4, "row_major", "tensor_hw"))
        # + list(gen_sharded_spec(16, 4, "col_major", "tensor_hw"))
        # + list(gen_sharded_spec(16, 4, "row_major", "block"))
        # + list(gen_sharded_spec(16, 4, "col_major", "block"))
        # + list(gen_sharded_spec(16, 4, "row_major", "height"))
        # + list(gen_sharded_spec(16, 4, "col_major", "height"))
        list(gen_sharded_spec(16, 4, "row_major", "width")) + list(gen_sharded_spec(16, 4, "col_major", "width")),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
    "test_3": {
        "input_spec": [
            {
                "input_shape": [6080, 672],
                "core_size": [7, 2],
                "sharding_strategy": "block",
                "shard_orientation": "row_major",
            }
        ],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_spec = test_vector["input_spec"]
    input_shape, core_grid_size, sharding_strategy, shard_orientation = input_spec.values()
    y, x = core_grid_size

    pre_sharded_height = math.prod(input_shape[:-1])
    pre_sharded_width = input_shape[-1]

    if sharding_strategy == "block":
        if shard_orientation == "row_major":
            if pre_sharded_height % y != 0:
                return (
                    True,
                    "Prod of all dimensions except the innermost must be divisible by the y coordinate of coregrid when using block sharding",
                )
            if pre_sharded_width % x != 0:
                return (
                    True,
                    "Innermost dimension must be divisible by the x coordinate of coregrid when using block sharding",
                )
            if (pre_sharded_height // y) // 32 <= 0:
                return True, "Shard height must be a atleast 32"
            if (pre_sharded_width // x) // 32 <= 0:
                return True, "Shard width must be a atleast 32"
        else:
            if pre_sharded_height % x != 0:
                return (
                    True,
                    "Prod of all dimensions except the innermost must be divisible by the x coordinate of coregrid when using block sharding",
                )
            if pre_sharded_width % y != 0:
                return (
                    True,
                    "Innermost dimension must be divisible by the y coordinate of coregrid when using block sharding",
                )
            if (pre_sharded_height // x) // 32 <= 0:
                return True, "Shard height must be a atleast 32"
            if (pre_sharded_width // y) // 32 <= 0:
                return True, "Shard width must be a atleast 32"

    if sharding_strategy == "width":
        if pre_sharded_width % (y * x) != 0:
            return True, "Last dimension must be divisible by a total number of cores when using width sharding"
        if pre_sharded_height // 32 <= 0:
            return True, "Shard height must be a atleast 32"
        if (pre_sharded_width // (x * y)) // 32 <= 0:
            return True, "Shard width must be a atleast 32"

    if sharding_strategy == "height":
        if pre_sharded_height % (y * x) != 0:
            return (
                True,
                "Prod of all dimensions except the innermost must be divisible by a total number of cores when using width sharding",
            )
        if (pre_sharded_height // (x * y)) // 32 <= 0:
            return True, "Shard height must be a atleast 32"
        if pre_sharded_width // 32 <= 0:
            return True, "Shard width must be a atleast 32"

    else:
        if input_shape[-2] % 32 != 0 or input_shape[-1] % 32 != 0:
            True, "Shard dimensions must be divisible by 32"

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

    input_shape, core_grid_size, sharding_strategy, shard_orientation = input_spec.values()
    y, x = core_grid_size
    device_grid_size = ttnn.CoreGrid(y=y, x=x)

    sharded_config = get_sharded_config(
        input_shape,
        sharding_strategy,
        device_grid_size,
        shard_orientation,
    )

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    low = -(input_shape[-2] - 2)
    high = input_shape[-1]
    diagonal = torch.randint(low, high, (1,)).item()

    torch_output_tensor = torch.triu(torch_input_tensor_a, diagonal)

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


from tests.sweep_framework.framework.permutations import *

for suite in parameters.keys():
    if suite != "xfail":
        continue
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    suite_vectors = list(permutations(parameters[suite]))
    passes = 0
    lowpcc = 0
    print(len(suite_vectors))
    for vector in suite_vectors:
        if invalidate_vector(vector)[0]:
            continue
        try:
            passed, _ = run(**vector, device=device)
            if passed[0] != True:
                lowpcc += 1
                print(passed)
                print(vector)
            else:
                passes += 1
        except Exception as e:
            print(str(e)[:60])
            print(vector)
    print(lowpcc)
    print(passes)
    ttnn.close_device(device)
