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
)

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 120

Y, X = get_device_grid_size()

random.seed(0)


def gen_sharded_spec(num_shapes, shard_orientation, sharding_strategy=None):
    if sharding_strategy:
        assert sharding_strategy in ["block", "width", "height"]

    assert shard_orientation in ["col_major", "row_major"]

    if not sharding_strategy:
        input_shape_list = (
            gen_shapes([1, 1, 32, 32], [6, 12, 256, 256], [1, 1, 32, 32], num_shapes)
            + gen_shapes([1, 32, 32], [12, 256, 256], [1, 32, 32], num_shapes)
            + gen_shapes([32, 32], [256, 256], [32, 32], num_shapes)
        )
        use_tensor_hw_as_shard_shape = True
        sharding_strategy = random.choice(["block", "width", "height"])

    else:
        input_shape_list = []
        use_tensor_hw_as_shard_shape = False
        for i in range(num_shapes):
            for rank in [2, 3, 4]:
                if sharding_strategy == "block":
                    min_shard_size_y = 32 * Y
                    min_shard_size_x = 32 * X

                    mul_x = random.randint(1, 10)
                    mul_y = random.randint(1, 64 // mul_x)

                    shape = random.choice(
                        _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                    )
                    shape = list(shape["reshape_dims"])
                    shape.append(mul_x * min_shard_size_x)

                    input_shape_list.append(shape)

                elif sharding_strategy == "height":
                    min_shard_size_y = 32 * X * Y
                    min_shard_size_x = 32

                    mul_x = random.randint(1, 10)
                    mul_y = random.randint(1, 2)

                    shape = random.choice(
                        _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                    )
                    shape = list(shape["reshape_dims"])
                    shape.append(mul_x * min_shard_size_x)

                    input_shape_list.append(shape)

                elif sharding_strategy == "width":
                    min_shard_size_x = 32 * X * Y
                    min_shard_size_y = 32

                    mul_y = random.randint(1, 10)
                    mul_x = random.randint(1, 2)

                    shape = [mul_y * min_shard_size_y]
                    wdith = random.choice(
                        _gen_reshape_args_from_volume(mul_x * min_shard_size_x, step=1, out_dims=rank - 1)
                    )
                    width = list(wdith["reshape_dims"])
                    shape.append(width[0])

                    input_shape_list.append(shape)

    for input_shape in input_shape_list:
        yield {
            "input_shape": input_shape,
            "sharding_strategy": sharding_strategy,
            "shard_orientation": shard_orientation,
            "use_tensor_hw_as_shard_shape": use_tensor_hw_as_shard_shape,
        }


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "xfail": {
        "input_spec": list(gen_sharded_spec(16, "row_major", "block"))
        + list(gen_sharded_spec(16, "col_major", "block"))
        + list(gen_sharded_spec(16, "row_major", "height"))
        + list(gen_sharded_spec(16, "col_major", "height"))
        + list(gen_sharded_spec(16, "row_major", "width"))
        + list(gen_sharded_spec(16, "col_major", "width"))
        + list(gen_sharded_spec(16, "row_major", None))
        + list(gen_sharded_spec(16, "col_major", None)),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_spec = test_vector["input_spec"]
    input_shape, sharding_strategy, _, tensor_hw_as_shard_shape = input_spec.values()

    pre_sharded_height = math.prod(input_shape[:-1])
    pre_sharded_width = input_shape[-1]

    if not tensor_hw_as_shard_shape:
        if sharding_strategy == "block":
            if pre_sharded_height % Y != 0:
                return (
                    True,
                    "Prod of all dimensions except the innermost must be divisible by the y coordinate of coregrid when using block sharding",
                )
            if pre_sharded_width % X != 0:
                return (
                    True,
                    "Innermost dimension must be divisible by the y coordinate of coregrid when using block sharding",
                )
            if (pre_sharded_height // Y) % 32 != 0:
                return True, "Shard height must be a multiple of input tile size"
            if (pre_sharded_width // X) % 32 != 0:
                return True, "Shard width must be a multiple of input tile size"

        if sharding_strategy == "width":
            if pre_sharded_width % (Y * X) != 0:
                return True, "Last dimension must be divisible by a total number of cores when using width sharding"
            if pre_sharded_height % 32 != 0:
                return True, "Shard height must be a multiple of input tile size"
            if (pre_sharded_width // (X * Y)) % 32 != 0:
                return True, "Shard width must be a multiple of input tile size"

        else:
            if pre_sharded_height % (Y * X) != 0:
                return (
                    True,
                    "Prod of all dimensions except the innermost must be divisible by a total number of cores when using width sharding",
                )
            if (pre_sharded_height // (X * Y)) % 32 != 0:
                return True, "Shard height must be a multiple of input tile size"
            if pre_sharded_width % 32 != 0:
                return True, "Shard width must be a multiple of input tile size"

    else:
        if input_shape[-2] % 32 != 0 or input_shape[-1] % 32 != 0:
            return (
                True,
                "Last two dimensions must be multiples of tile size when using tensor heght and width as shard shape",
            )

    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT or test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "Row Major layout and bfloat8_b are not supported"

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
    sharded_config = get_sharded_config(
        input_shape, sharding_strategy, device_grid_size, shard_orientation, tensor_hw_as_shard_shape
    )

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    scalar = torch.tensor(1, dtype=torch.bfloat16).uniform_(0, 100).item()

    golden_function = ttnn.get_golden_function(ttnn.hardshrink)
    torch_output_tensor = golden_function(torch_input_tensor_a, lambd=scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=sharded_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.hardshrink(input_tensor_a, lambd=scalar, memory_config=sharded_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
