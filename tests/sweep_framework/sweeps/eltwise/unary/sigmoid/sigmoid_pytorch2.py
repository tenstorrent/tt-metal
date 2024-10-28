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
TIMEOUT = 30

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": [
            [1, 1, 256, 256],
            [1, 1, 480, 640],
            [1, 100, 4],
            [1, 104, 1, 1],
            [1, 1056, 1, 1],
            [1, 12, 64, 64],
            [1, 120, 1, 1],
            [1, 120, 14, 14],
            [1, 1232, 1, 1],
            [1, 1392, 1, 1],
            [1, 144, 1, 1],
            [1, 1512, 1, 1],
            [1, 16, 64, 64],
            [1, 184, 7, 7],
            [1, 2, 120, 160],
            [1, 2, 30, 40],
            [1, 2, 60, 80],
            [1, 200, 7, 7],
            [1, 2016, 1, 1],
            [1, 208, 1, 1],
            [1, 216, 1, 1],
            [1, 224, 1, 1],
            [1, 232, 1, 1],
            [1, 24, 64, 64],
            [1, 240, 14, 14],
            [1, 2904, 1, 1],
            [1, 3, 16, 16, 85],
            [1, 3, 32, 32, 85],
            [1, 3, 64, 64, 85],
            [1, 3, 64, 64],
            [1, 3024, 1, 1],
            [1, 32, 64, 64],
            [1, 320, 1, 1],
            [1, 336, 1, 1],
            [1, 3712, 1, 1],
            [1, 4, 64, 64],
            [1, 440, 1, 1],
            [1, 448, 1, 1],
            [1, 48, 1, 1],
            [1, 480, 7, 7],
            [1, 50, 3072],
            [1, 528, 1, 1],
            [1, 576, 1, 1],
            [1, 6, 64, 64],
            [1, 64, 1, 1],
            [1, 672, 7, 7],
            [1, 696, 1, 1],
            [1, 72, 1, 1],
            [1, 72, 28, 28],
            [1, 7392, 1, 1],
            [1, 784, 1, 1],
            [1, 8, 64, 64],
            [1, 888, 1, 1],
            [1, 896, 1, 1],
            [1, 960, 3, 3],
            [2, 7, 2048],
            [6, 1, 100, 4],
        ],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    golden_function = ttnn.get_golden_function(ttnn.sigmoid)
    torch_output_tensor = golden_function(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.sigmoid(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
