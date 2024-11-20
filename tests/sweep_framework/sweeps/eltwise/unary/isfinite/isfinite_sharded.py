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
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_inf, _gen_reshape_args_from_volume

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 120
Y, X = get_device_grid_size()

random.seed(0)


def gen_sharded_spec(num_shapes, y, x, max_tensor_size=8 * 1024 * 1024):
    # [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.HEIGHT, "tensor_wh"]
    sharding_strategy_list = ["tensor_wh"]
    shard_orientation_list = [ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardOrientation.ROW_MAJOR]
    spec_list = []

    for sharding_strategy, shard_orientation, rank, layout in itertools.product(
        sharding_strategy_list, shard_orientation_list, [4, 3, 2], [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
    ):
        if sharding_strategy == "tensor_wh":
            tensor_hw_as_shard_shape = True
            sharding_strategy = ttnn.ShardStrategy.BLOCK
        else:
            tensor_hw_as_shard_shape = False

        for _ in range(num_shapes):
            if tensor_hw_as_shard_shape:
                # Gets stuck:
                # X 8 Y 8 input_shape [1, 17792, 8] DataType.BFLOAT8_B Layout.TILE ShardStrategy.BLOCK ShardOrientation.COL_MAJOR tensor_hw_as_shard_shape True

                if layout == ttnn.TILE_LAYOUT:
                    # In shard mode ShardMode::PHYSICAL, physical shard shape {12, 13312} is not compatible with alignment Alignment([32, 32])!
                    min_shard_size_x = 32
                    min_shard_size_y = 32
                else:  # if layout == ttnn.ROW_MAJOR_LAYOUT:
                    # Shard Size must be multiple of input_tile_size (width * height is multiple of 1024)
                    min_shard_size_x = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
                    min_shard_size_y = 1024 // min_shard_size_x

                rest_volume = random.randint(1, max_tensor_size // (min_shard_size_x * min_shard_size_y * x * y))
                input_shape = random.choice(_gen_reshape_args_from_volume(rest_volume, step=1, out_dims=rank))
                input_shape = list(input_shape["reshape_dims"])
                input_shape[-2] = input_shape[-2] * min_shard_size_x
                input_shape[-1] = input_shape[-1] * min_shard_size_y

                # Shard width should be multiple of 16 to satisfy L1 alignment (width = multiple 8 for bfloat16)
                while input_shape[-1] % 16 != 0:
                    input_shape[-1] *= 2
                    input_shape[-2] //= 2

                if shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
                    tmp = input_shape[-2]
                    input_shape[-2] = input_shape[-1]
                    input_shape[-1] = tmp

            elif sharding_strategy == ttnn.ShardStrategy.BLOCK:
                min_shard_size_y = 32 * y
                min_shard_size_x = 32 * x
                mul_x = random.randint(1, 10)
                mul_y = random.randint(1, 64 // mul_x)

                input_shape = random.choice(
                    _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                )
                input_shape = list(input_shape["reshape_dims"])
                input_shape.append(mul_x * min_shard_size_x)

            elif sharding_strategy == ttnn.ShardStrategy.WIDTH or sharding_strategy == ttnn.ShardStrategy.HEIGHT:
                # if shard_width % total_cores != 0: raise RuntimeError("Invalid sharding core_grid")
                # Shard Size must be multiple of input_tile_size

                if layout == ttnn.TILE_LAYOUT:
                    # In shard mode ShardMode::PHYSICAL, physical shard shape {12, 13312} is not compatible with alignment Alignment([32, 32])!
                    min_shard_size_x = 32
                    min_shard_size_y = 32 * x * y
                else:  # if layout == ttnn.ROW_MAJOR_LAYOUT:
                    min_shard_size_x = 1
                    min_shard_size_y = x * y

                    # Shard Size must be multiple of input_tile_size
                    mul_32_x = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
                    mul_32_y = 1024 // mul_32_x

                    min_shard_size_x *= mul_32_x
                    min_shard_size_y *= mul_32_y

                    if sharding_strategy == ttnn.ShardStrategy.HEIGHT:
                        # Shard width should be multiple of 16 to satisfy L1 alignment
                        while min_shard_size_x % 16 != 0:
                            min_shard_size_x *= 2

                rest_volume = random.randint(1, max_tensor_size // (min_shard_size_x * min_shard_size_y))
                input_shape = random.choice(_gen_reshape_args_from_volume(rest_volume, step=1, out_dims=rank))
                input_shape = list(input_shape["reshape_dims"])
                input_shape[-2] = input_shape[-2] * min_shard_size_x
                input_shape[-1] = input_shape[-1] * min_shard_size_y

                if sharding_strategy == ttnn.ShardStrategy.HEIGHT:
                    tmp = input_shape[-2]
                    input_shape[-2] = input_shape[-1]
                    input_shape[-1] = tmp

                # print(input_shape)

            spec_list.append(
                {
                    "input_shape": input_shape,
                    "sharding_strategy": sharding_strategy,
                    "shard_orientation": shard_orientation,
                    "tensor_hw_as_shard_shape": tensor_hw_as_shard_shape,
                    "input_layout": layout,
                }
            )

    return spec_list


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_spec": gen_sharded_spec(16, Y, X),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_shape, sharding_strategy, _, _, input_layout = test_vector["input_spec"].values()
    pre_sharded_height = math.prod(input_shape[:-1])
    pre_sharded_width = input_shape[-1]

    if input_layout == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is only supported on tiled layout"

    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_spec,
    input_a_dtype,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)
    input_shape, sharding_strategy, shard_orientation, tensor_hw_as_shard_shape, input_layout = input_spec.values()

    print(
        f"X {X} Y {Y} input_shape {input_shape} {input_a_dtype} {input_layout} {sharding_strategy} {shard_orientation} tensor_hw_as_shard_shape {tensor_hw_as_shard_shape}"
    )

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
    print(pcc)
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
            # if passed[0] != True:
            #     print(passed)
        except Exception as e:
            print(e)

        # break

    ttnn.close_device(device)

e2e_perf = stop_measuring_time(start_time)
print(f"time {e2e_perf / 1000000000}s")
