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
            {"self": [1, 12, 1, 1], "other": [1, 12, 1, 1], "input_dtype": "ttnn.float32"},
            {"self": [1, 16, 1, 1], "other": [1, 16, 1, 1], "input_dtype": "ttnn.float32"},
            {"self": [1, 6, 1, 1], "other": [1, 6, 1, 1], "input_dtype": "ttnn.float32"},
            {"self": [1, 8, 1, 1], "other": [1, 8, 1, 1], "input_dtype": "ttnn.float32"},
            {"self": [120, 1], "other": [120, 1], "input_dtype": "ttnn.float32"},
            {"self": [128], "other": [128], "input_dtype": "ttnn.float32"},
            {"self": [128, 1], "other": [128, 1], "input_dtype": "ttnn.float32"},
            {"self": [160], "other": [160], "input_dtype": "ttnn.float32"},
            {"self": [1, 1024, 1, 1], "other": [1, 1024, 1, 1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 128, 1, 1], "other": [1, 128, 1, 1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 12, 27, 27], "other": [1, 12, 27, 27], "input_dtype": "ttnn.float32"},
            {"self": [1, 16, 27, 27], "other": [1, 16, 27, 27], "input_dtype": "ttnn.float32"},
            {"self": [1, 2048, 1, 1], "other": [1, 2048, 1, 1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 256, 128, 128], "other": [1, 256, 128, 128], "input_dtype": "ttnn.float32"},
            {"self": [1, 256, 1, 1], "other": [1, 256, 1, 1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 3, 320, 320], "other": [1, 3, 320, 320], "input_dtype": "ttnn.float32"},
            {"self": [1, 512, 1, 1], "other": [1, 512, 1, 1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 64, 120, 160], "other": [1, 64, 120, 160], "input_dtype": "ttnn.float32"},
            {"self": [1, 64, 1, 1], "other": [1, 64, 1, 1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 64, 240, 320], "other": [1, 64, 240, 320], "input_dtype": "ttnn.float32"},
            {"self": [1, 64, 30, 40], "other": [1, 64, 30, 40], "input_dtype": "ttnn.float32"},
            {"self": [1, 64, 480, 640], "other": [1, 64, 480, 640], "input_dtype": "ttnn.float32"},
            {"self": [1, 64, 60, 80], "other": [1, 64, 60, 80], "input_dtype": "ttnn.float32"},
            {"self": [240, 1], "other": [240, 1], "input_dtype": "ttnn.float32"},
            {"self": [27], "other": [27], "input_dtype": "ttnn.float32"},
            {"self": [27, 1], "other": [27, 1], "input_dtype": "ttnn.float32"},
            {"self": [30, 1], "other": [30, 1], "input_dtype": "ttnn.float32"},
            {"self": [320], "other": [320], "input_dtype": "ttnn.float32"},
            {"self": [320, 1], "other": [320, 1], "input_dtype": "ttnn.float32"},
            {"self": [3234], "other": [3234], "input_dtype": "ttnn.float32"},
            {"self": [3234, 1], "other": [3234, 1], "input_dtype": "ttnn.float32"},
            {"self": [3234, 2], "other": [3234, 2], "input_dtype": "ttnn.float32"},
            {"self": [40], "other": [40], "input_dtype": "ttnn.float32"},
            {"self": [480, 1], "other": [480, 1], "input_dtype": "ttnn.float32"},
            {"self": [60, 1], "other": [60, 1], "input_dtype": "ttnn.float32"},
            {"self": [640], "other": [640], "input_dtype": "ttnn.float32"},
            {"self": [80], "other": [80], "input_dtype": "ttnn.float32"},
            # {"self": [197], "other": [197], "input_dtype": "ttnn.int32"},
            # {"self": [19], "other": [19], "input_dtype": "ttnn.int32"},
            # {"self": [1], "other": [1], "input_dtype": "ttnn.int32"},
            # {"self": [1, 1], "other": [1, 1], "input_dtype": "ttnn.int32"},
            # {"self": [1, 32], "other": [1, 32], "input_dtype": "ttnn.int32"},
            # {"self": [1, 45], "other": [1, 45], "input_dtype": "ttnn.int32"},
            # {"self": [1, 5], "other": [1, 5], "input_dtype": "ttnn.int32"},
            # {"self": [1, 6], "other": [1, 6], "input_dtype": "ttnn.int32"},
            # {"self": [45], "other": [45], "input_dtype": "ttnn.int32"},
            # {"self": [5], "other": [5], "input_dtype": "ttnn.int32"},
            # {"self": [6], "other": [6], "input_dtype": "ttnn.int32"},
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

    golden_function = ttnn.get_golden_function(ttnn.subtract)
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
    result = ttnn.subtract(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    # ToDo: Update it once the tensor layout support with rank < 2 is supported in mid of Jan
    output_tensor = ttnn.to_torch(result, torch_rank=len(input_shape["self"]))
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
