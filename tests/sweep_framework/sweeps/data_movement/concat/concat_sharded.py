# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import (
    check_with_pcc,
    start_measuring_time,
    stop_measuring_time,
    get_per_core_size_and_num_cores,
)
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 20
random.seed(0)


def wrap_dim(dim, rank):
    return dim if dim >= 0 else rank + dim


def generate_concat_block_config(nonvariable_dim, variable_dims, num_cores_choices_h, num_cores_choices_w):
    for dim in range(-2, 2):
        for var_dim1 in variable_dims:
            for var_dim2 in variable_dims:
                for nonvar_dim in nonvariable_dim:
                    shape1 = [None, None]
                    shape2 = [None, None]

                    shape1[dim] = var_dim1
                    shape2[dim] = var_dim2
                    shape1[1 - wrap_dim(dim, 2)] = nonvar_dim
                    shape2[1 - wrap_dim(dim, 2)] = nonvar_dim

                    for num_cores_height in num_cores_choices_h:
                        if shape1[0] % num_cores_height != 0:
                            continue
                        per_core_height = shape1[0] // num_cores_height

                        for num_cores_width in num_cores_choices_w:
                            if shape1[1] % num_cores_width != 0:
                                continue
                            per_core_width = shape1[1] // num_cores_width

                            for num_cores_height2 in num_cores_choices_h:
                                if shape2[0] % num_cores_height2 != 0:
                                    continue
                                per_core_height2 = shape2[0] // num_cores_height2

                                for num_cores_width2 in num_cores_choices_w:
                                    if shape2[1] % num_cores_width2 != 0:
                                        continue
                                    per_core_width2 = shape2[1] // num_cores_width2
                                    yield {
                                        "dim": dim,
                                        "shape1": shape1,
                                        "shape2": shape2,
                                        "core_shape1": [per_core_height, per_core_width],
                                        "core_shape2": [per_core_height2, per_core_width2],
                                        "num_cores1": [num_cores_height, num_cores_width],
                                        "num_cores2": [num_cores_height2, num_cores_width2],
                                    }


def generate_concat_height_config(nonvariable_dim, variable_dims, num_cores_choices_h):
    for dim in range(-2, 2):
        for var_dim1 in variable_dims:
            for var_dim2 in variable_dims:
                for nonvar_dim in nonvariable_dim:
                    shape1 = [None, None]
                    shape2 = [None, None]

                    shape1[dim] = var_dim1
                    shape2[dim] = var_dim2
                    shape1[1 - wrap_dim(dim, 2)] = nonvar_dim
                    shape2[1 - wrap_dim(dim, 2)] = nonvar_dim

                    for num_cores_height in num_cores_choices_h:
                        if shape1[0] % num_cores_height != 0:
                            continue
                        per_core_height = shape1[0] // num_cores_height
                        for num_cores_height2 in num_cores_choices_h:
                            if shape2[0] % num_cores_height2 != 0:
                                continue
                            per_core_height2 = shape2[0] // num_cores_height2
                            if per_core_height2 > shape2[0]:
                                continue  # function rounded up to 32 as shape doesn't divide evenly into cores
                            yield {
                                "dim": dim,
                                "shape1": shape1,
                                "shape2": shape2,
                                "core_shape1": [per_core_height, shape1[1]],
                                "core_shape2": [
                                    per_core_height2,
                                    shape2[1],
                                ],
                                "num_cores1": [
                                    num_cores_height,
                                    1,
                                ],
                                "num_cores2": [
                                    num_cores_height2,
                                    1,
                                ],
                            }


def generate_concat_width_config(nonvariable_dim, variable_dims, num_cores_choices_w):
    for dim in range(-2, 2):
        for var_dim1 in variable_dims:
            for var_dim2 in variable_dims:
                for nonvar_dim in nonvariable_dim:
                    shape1 = [None, None]
                    shape2 = [None, None]

                    shape1[dim] = var_dim1
                    shape2[dim] = var_dim2
                    shape1[1 - wrap_dim(dim, 2)] = nonvar_dim
                    shape2[1 - wrap_dim(dim, 2)] = nonvar_dim

                    for num_cores_width in num_cores_choices_w:
                        if shape1[1] % num_cores_width != 0:
                            continue
                        per_core_width = shape1[1] // num_cores_width
                        for num_cores_width2 in num_cores_choices_w:
                            if shape2[1] % num_cores_width2 != 0:
                                continue
                            per_core_width2 = shape2[1] // num_cores_width2
                            yield {
                                "dim": dim,
                                "shape1": shape1,
                                "shape2": shape2,
                                "core_shape1": [shape1[0], per_core_width],
                                "core_shape2": [
                                    shape2[0],
                                    per_core_width2,
                                ],
                                "num_cores1": [
                                    1,
                                    num_cores_width,
                                ],
                                "num_cores2": [
                                    1,
                                    num_cores_width2,
                                ],
                            }


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameter_block_sharded = {
    f"block_sharded_suite_2_tensors": {
        "concat_specs": list(
            generate_concat_block_config(
                [x for x in range(64, 320, 64)],
                [x for x in range(64, 320, 64)],
                [2],
                [2],
            )
        ),
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_mem_config": [
            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ],
        "output_mem_config": [
            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ],
    }
}

parameters_height_sharded = {
    f"height_sharded_suite_2_tensors": {
        "concat_specs": list(
            generate_concat_height_config(
                [x for x in range(64, 320, 64)],
                [x for x in range(64, 320, 64)],
                [2],
            )
        ),
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_mem_config": [
            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ],
        "output_mem_config": [
            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ],
    }
}

parameters_width_sharded = {
    f"width_sharded_suite_2_tensors": {
        "concat_specs": list(
            generate_concat_width_config(
                [x for x in range(64, 320, 64)],
                [x for x in range(64, 320, 64)],
                [2],
            )
        ),
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_mem_config": [
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ],
        "output_mem_config": [
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ],
    }
}

parameters = {**parameter_block_sharded, **parameters_height_sharded, **parameters_width_sharded}
print(f"parameter keys: {parameters.keys()}")


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"

    def height(l):
        result = 1
        for i in range(0, len(l) - 1):
            result *= l[i]
        return result

    if test_vector["layout"] == ttnn.TILE_LAYOUT:
        for shape in [
            test_vector["concat_specs"]["shape1"],
            test_vector["concat_specs"]["shape2"],
            test_vector["concat_specs"]["core_shape1"],
            test_vector["concat_specs"]["core_shape2"],
        ]:
            if shape[-1] % 32 != 0:
                return True, "TILE_LAYOUT requires the last dimension to be a multiple of 32"
            elif len(shape) > 1 and height(shape) % 32 != 0:
                return True, "TILE_LAYOUT requires the product of all dimensions except the last to be a multiple of 32"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.


def run(
    concat_specs,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    *,
    device,
) -> list:
    torch_input_tensors = []
    device.enable_async(False)
    if input_mem_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
        input_mem_config_a = ttnn.create_sharded_memory_config(
            shape=concat_specs["shape1"],
            core_grid=ttnn.CoreGrid(x=concat_specs["num_cores1"][1], y=concat_specs["num_cores1"][0]),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
        input_mem_config_b = ttnn.create_sharded_memory_config(
            shape=concat_specs["shape2"],
            core_grid=ttnn.CoreGrid(x=concat_specs["num_cores2"][1], y=concat_specs["num_cores2"][0]),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
    elif input_mem_config == ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG:
        input_mem_config_a = ttnn.create_sharded_memory_config(
            shape=concat_specs["shape1"],
            core_grid=ttnn.CoreGrid(x=concat_specs["num_cores1"][1], y=concat_specs["num_cores1"][0]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
        input_mem_config_b = ttnn.create_sharded_memory_config(
            shape=concat_specs["shape2"],
            core_grid=ttnn.CoreGrid(x=concat_specs["num_cores2"][1], y=concat_specs["num_cores2"][0]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
    else:
        input_mem_config_a = ttnn.create_sharded_memory_config(
            shape=concat_specs["shape1"],
            core_grid=ttnn.CoreGrid(x=concat_specs["num_cores1"][1], y=concat_specs["num_cores1"][0]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
        input_mem_config_b = ttnn.create_sharded_memory_config(
            shape=concat_specs["shape1"],
            core_grid=ttnn.CoreGrid(x=concat_specs["num_cores2"][1], y=concat_specs["num_cores2"][0]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
    torch_input_tensors.append(torch_random(concat_specs["shape1"], -0.1, 0.1, dtype=torch.bfloat16))
    torch_input_tensors.append(torch_random(concat_specs["shape2"], -0.1, 0.1, dtype=torch.bfloat16))
    torch_output_tensor = torch.concat(torch_input_tensors, dim=concat_specs["dim"])

    if (
        output_mem_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        or output_mem_config == ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        or output_mem_config == ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    ):
        shape_list = list(torch_output_tensor.size())
        output_mem_config = ttnn.create_sharded_memory_config(
            shape=shape_list,
            core_grid=ttnn.CoreGrid(x=concat_specs["num_cores1"][1], y=concat_specs["num_cores1"][0]),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )

    input_tensors = []
    input_tensors.append(
        ttnn.from_torch(
            torch_input_tensors[0],
            device=device,
            layout=layout,
            dtype=dtype,
            memory_config=input_mem_config_a,
        )
    )
    input_tensors.append(
        ttnn.from_torch(
            torch_input_tensors[1],
            device=device,
            layout=layout,
            dtype=dtype,
            memory_config=input_mem_config_b,
        )
    )
    start_time = start_measuring_time()
    result_tensor = ttnn.concat(input_tensors, dim=concat_specs["dim"], memory_config=output_mem_config)
    e2e_perf = stop_measuring_time(start_time)
    output_tensor = ttnn.to_torch(result_tensor)
    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
