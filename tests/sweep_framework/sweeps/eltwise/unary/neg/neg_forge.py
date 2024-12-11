# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
# TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.


parameters = {
    "nightly": {
        "input_shape": [
            [1, 14, 1],
            [1, 5, 1200, 1],
            [1, 5, 16, 16],
            [1, 16, 32, 1],
            [1, 1, 9, 9],
            [1, 1, 19, 19],
            [8, 920, 1],
            [240, 1, 1],
            [512, 1, 1],
            [1, 32, 32, 64],
            [1, 8, 10, 1],
            [116, 1, 1],
            [1, 12, 1, 1],
            [1, 12, 201, 1],
            [462, 1, 1],
            [78, 1, 1],
            [2, 1, 7, 7],
            [1, 8, 2048, 1],
            [1, 6, 1],
            [1, 1, 16, 16],
            [1, 6, 1, 1],
            [58, 1, 1],
            [1, 5, 1024, 1],
            [1, 7, 1],
            [960, 1, 1],
            [1024, 1, 1],
            [1, 1200, 1],
            [2, 7, 1],
            [1, 64, 1],
            [1, 32, 1, 1],
            [1, 16, 1, 1],
            [1, 12, 197, 1],
            [1],
            [1, 16, 5, 1],
            [1, 1, 1, 2048],
            [128, 1, 1],
            [1, 19200, 1],
            [334, 1, 1],
            [672, 1, 1],
            [1, 1, 16384, 1],
            [40, 1, 1],
            [1, 1445, 1],
            [80, 1, 1],
            [68, 1, 1],
            [1, 8, 300, 1],
            [8, 100, 1],
            [98, 1, 1],
            [1, 1, 1, 201],
            [320, 1, 1],
            [1, 16, 1],
            [1, 5, 1],
            [134, 1, 1],
            [20, 1, 1],
            [480, 1, 1],
            [1, 16, 197, 1],
            [32, 1, 1],
            [1, 1, 19200, 1],
            [1280, 1, 1],
            [1, 9, 1],
            [640, 1, 1],
            [2048, 1, 1],
            [1, 12, 10, 1],
            [120, 1, 1],
            [1, 6, 15, 1],
            [1, 12, 8, 1],
            [168, 1, 1],
            [3, 1, 1],
            [16, 1, 1],
            [1, 50, 1],
            [920, 1, 1],
            [1, 1, 25, 25],
            [384, 1, 1],
            [1, 8, 1, 1],
            [34, 1, 1],
            [24, 1, 1],
            [1, 8, 256, 1],
            [1, 12, 1],
            [19, 1],
            [1, 71, 7, 32],
            [1, 1, 1, 15],
            [1, 1, 256, 256],
            [1, 25, 1],
            [1, 201, 1],
            [1, 1024, 1],
            [1, 256, 1],
            [72, 1, 1],
            [576, 1, 1],
            [1, 16384, 1],
            [192, 1, 1],
            [1, 16, 10, 1],
            [1, 1, 1, 8],
            [1, 4096, 1],
            [256, 1, 1],
            [1, 1, 14, 14],
            [160, 1, 1],
            [1, 1, 10, 10],
            [144, 1, 1],
            [1, 2, 4096, 1],
            [1, 1, 6, 6],
            [200, 1, 1],
            [1, 1, 1, 10],
            [1, 4800, 1],
            [28, 1, 1],
            [1, 19, 1],
            [1, 1, 7, 32],
            [1, 8, 1],
            [272, 1, 1],
            [46, 1, 1],
            [1, 10, 1],
            [112, 1, 1],
            [1, 32, 1],
            [96, 1, 1],
            [1, 2, 4800, 1],
            [1, 1, 7, 7],
            [1, 197, 1],
            [1, 1],
            [1, 1, 12, 12],
            [1, 2048, 1],
            [14, 1, 1],
            [196, 1, 1],
            [64, 1, 1],
            [1, 300, 1],
            [184, 1, 1],
        ],
        "input_a_dtype": [ttnn.float32, ttnn.bfloat16],  # ttnn.int32],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row Major layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    golden_function = ttnn.get_golden_function(ttnn.neg)
    torch_output_tensor = golden_function(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.neg(input_tensor_a, memory_config=output_memory_config)
    # ToDo: Update it once the tensor layout support with rank < 2 is supported in mid of Jan
    output_tensor = ttnn.to_torch(result, torch_rank=len(input_shape))
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
