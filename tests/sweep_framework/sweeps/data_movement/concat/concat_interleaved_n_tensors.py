# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30  # formatting on host and torch CPU call are slow
random.seed(0)
tiled_dim_unpadded = [32, 64, 96]
tiled_dim_padded = [7, 16]

rm_dim_even = [8, 10, 12, 16, 32, 46]
rm_dim_odd = [1, 33]


def get_random_value(valid_dims):
    return random.choice(valid_dims)


def generate_shapes(tensor_counts, rank, concat_dim, variable_dims, other_dims):
    shape = [get_random_value(other_dims) for _ in range(rank)]  # nonvariable dims will sample from other_dims
    result = []
    for _ in range(tensor_counts):
        new_shape = shape.copy()
        new_shape[concat_dim] = get_random_value(variable_dims)
        result.append(new_shape)
    return result


def generate_concat_config(tensor_counts, ranks, variable_dim, other_dims):
    for rank in ranks:
        for concat_dim in range(-rank, rank):
            shapes = generate_shapes(
                tensor_counts, rank, concat_dim, variable_dim, other_dims
            )  # variable dims will sample from variable_dim
            config = (
                tensor_counts,
                rank,
                concat_dim,
                shapes,
            )
            yield config


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameter_tiled_interleaved = {
    f"tiled_interleaved_suite_{n}_tensors": {
        "concat_specs": list(
            generate_concat_config(n, [4, 3], tiled_dim_unpadded, tiled_dim_unpadded + tiled_dim_padded)
        )
        + list(
            generate_concat_config(n, [4, 3], tiled_dim_padded, [32, 33])
        ),  # variable dim doesn't support padding, other dims can be anything
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.TILE_LAYOUT],
        "input_mem_config": [
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ],
        "output_mem_config": [
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ],
    }
    for n in range(4, 2, -1)
}

parameters_row_major_interleaved = {
    f"row_major_interleaved_suite_{n}_tensors": {
        "concat_specs": list(generate_concat_config(n, [4, 3], rm_dim_even, rm_dim_even))
        + list(generate_concat_config(n, [4, 3], rm_dim_even, rm_dim_odd))
        + list(
            generate_concat_config(n, [4, 3], rm_dim_odd, rm_dim_even)
        ),  # variable dim doesn't support padding, other dims can be anything
        "dtype": [ttnn.bfloat16],
        "layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_mem_config": [
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ],
        "output_mem_config": [
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ],
    }
    for n in range(4, 2, -1)
}

parameters = {**parameter_tiled_interleaved, **parameters_row_major_interleaved}
print(f"parameter keys: {parameters.keys()}")


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"
        for shape in test_vector["concat_specs"][3]:
            if shape[-1] % 2 != 0:
                return True, "ROW_MAJOR_LAYOUT requires the last dimension to be aligned to uint32"
    #

    # def height(l):
    #     result = 1
    #     for i in range(0, len(l) - 1):
    #         result *= l[i]
    #     return result
    # if test_vector["layout"] == ttnn.TILE_LAYOUT:
    #     for shape in test_vector["concat_specs"][3]:
    #         if shape[-1] % 32 != 0:
    #             return True, "TILE_LAYOUT requires the last dimension to be a multiple of 32"
    #         elif len(shape) > 1 and height(shape) % 32 != 0:
    #             return True, "TILE_LAYOUT requires the product of all dimensions except the last to be a multiple of 32"
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
    for i in range(0, concat_specs[0]):
        torch_input_tensors.append(torch_random(concat_specs[3][i], -0.1, 0.1, dtype=torch.bfloat16))

    input_tensors = [
        ttnn.from_torch(
            torch_input_tensor,
            device=device,
            layout=layout,
            dtype=dtype,
            memory_config=input_mem_config,
        )
        for torch_input_tensor in torch_input_tensors
    ]

    start_time = start_measuring_time()
    result_tensor = ttnn.concat(input_tensors, dim=concat_specs[2], memory_config=output_mem_config)
    e2e_perf = stop_measuring_time(start_time)
    output_tensor = ttnn.to_torch(result_tensor)

    torch_output_tensor = torch.concat(torch_input_tensors, dim=concat_specs[2])
    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
