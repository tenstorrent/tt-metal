# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.


parameters = {
    "nightly": {
        "input_shape": [
            {"self": [120, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [128], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [128, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [160], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [240], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [240, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [27], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [27, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [30], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [30, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [320], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [320, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [3234, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [3234, 2], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [40], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [480], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [480, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [60], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [60, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [640], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [6, 2], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [80], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 184, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 200, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 240, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 240, 40, 40], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 480, 10, 10], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 480, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 672, 10, 10], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 672, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 1], "other": [1, 1], "input_dtype": "ttnn.int32"},
            # {"self": [1], "other": [1, 480, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 672, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 72, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 116, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 1280, 7, 7], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 128, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 128, 2, 2], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 128, 3, 3], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 128, 56, 56], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 128, 5, 5], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 134, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 144, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 144, 56, 56], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 14, 56, 56], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 160, 7, 7], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 168, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 16, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 192, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 192, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 196, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 20, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 24, 56, 56], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 256, 10, 10], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 256, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 256, 2, 2], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 256, 3, 3], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 256, 5, 5], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 272, 7, 7], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 28, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 320, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 32, 112, 112], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 334, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 34, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 384, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 40, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 40, 56, 56], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 462, 7, 7], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 46, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 480, 10, 10], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 512, 5, 5], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 576, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 576, 7, 7], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 58, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 640, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 64, 112, 112], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 64, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 64, 2, 2], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 672, 20, 20], "input_dtype": "ttnn.float32"},
            # {"self": [1], "other": [1, 68, 14, 14], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 68, 56, 56], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 78, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 960, 7, 7], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 96, 112, 112], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 96, 56, 56], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 98, 28, 28], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 128, 128, 128], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 128, 32, 32], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 128, 64, 64], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 256, 16, 16], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 256, 32, 32], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 256, 64, 64], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 32, 256, 256], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 32, 512, 512], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 512, 16, 16], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 512, 32, 32], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 64, 128, 128], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1], "other": [1, 64, 256, 256], "input_dtype": "ttnn.bfloat16"},
            # {"self": [15, 15], "other": [15, 15], "input_dtype": "ttnn.int32"},
        ],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT or test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row Major layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)
    if input_shape["input_dtype"] == "ttnn.bfloat16":
        input_dtype = ttnn.bfloat16
    elif input_shape["input_dtype"] == "ttnn.float32":
        input_dtype = ttnn.float32
    elif input_shape["input_dtype"] == "ttnn.int32":
        input_dtype = ttnn.int32

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape["self"])
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape["other"])

    golden_function = ttnn.get_golden_function(ttnn.minimum)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.minimum(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    # ToDo: Update it once the tensor layout support with rank < 2 is supported in mid of Jan
    input_a_rank = len(input_shape["self"])
    input_b_rank = len(input_shape["other"])
    output_rank = max(input_a_rank, input_b_rank)
    output_tensor = ttnn.to_torch(result, torch_rank=output_rank)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
