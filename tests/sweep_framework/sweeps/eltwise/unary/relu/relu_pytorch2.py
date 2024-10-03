# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.utils import gen_shapes
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
            [1, 1, 2048],
            [1, 1, 256],
            [1, 1, 3072],
            [1, 1, 4096],
            [1, 1, 768],
            [1, 10, 2048],
            [1, 10, 3072],
            [1, 10, 4096],
            [1, 100, 14, 14],
            [1, 100, 192],
            [1, 1008, 14, 14],
            [1, 1008, 7, 7],
            [1, 1024, 10, 10],
            [1, 1024, 14, 14],
            [1, 1024, 19, 19],
            [1, 1024, 28, 28],
            [1, 1024, 45, 80],
            [1, 1024, 50, 68],
            [1, 1024, 7, 7],
            [1, 104, 28, 28],
            [1, 104, 56, 56],
            [1, 1056, 14, 14],
            [1, 1056, 48, 48],
            [1, 1056, 7, 7],
            [1, 1056, 96, 96],
            [1, 1088, 14, 14],
            [1, 1088, 7, 7],
            [1, 110, 1, 1],
            [1, 1104, 14, 14],
            [1, 1104, 7, 7],
            [1, 112, 1, 1],
            [1, 112, 14, 14],
            [1, 1120, 14, 14],
            [1, 1120, 7, 7],
            [1, 1152, 14, 14],
            [1, 1152, 7, 7],
            [1, 1184, 14, 14],
            [1, 1184, 7, 7],
            [1, 12, 1, 1],
            [1, 120, 1, 1],
            [1, 120, 28, 28],
            [1, 120, 40, 40],
            [1, 120, 56, 56],
            [1, 1200, 14, 14],
            [1, 1200, 7, 7],
            [1, 1216, 14, 14],
            [1, 1216, 7, 7],
            [1, 1232, 14, 14],
            [1, 1232, 28, 28],
            [1, 1248, 14, 14],
            [1, 1248, 7, 7],
            [1, 128, 10, 10],
            [1, 128, 100, 136],
            [1, 128, 112, 112],
            [1, 128, 14, 14],
            [1, 128, 150, 150],
            [1, 128, 17, 17],
            [1, 128, 180, 320],
            [1, 128, 200, 272],
            [1, 128, 28, 28],
            [1, 128, 3, 3],
            [1, 128, 5, 5],
            [1, 128, 56, 56],
            [1, 128, 64, 64],
            [1, 128, 7, 7],
            [1, 128, 75, 75],
            [1, 128, 90, 160],
            [1, 1280, 1, 1],
            [1, 1280, 14, 14],
            [1, 1280, 7, 7],
            [1, 128],
            [1, 1296, 14, 14],
            [1, 1296, 7, 7],
            [1, 12],
            [1, 1312, 14, 14],
            [1, 1312, 7, 7],
            [1, 132, 1, 1],
            [1, 1344, 14, 14],
            [1, 1344, 28, 28],
            [1, 1344, 7, 7],
            [1, 1376, 14, 14],
            [1, 1376, 7, 7],
            [1, 1392, 14, 14],
            [1, 1392, 28, 28],
            [1, 1392, 7, 7],
            [1, 1408, 14, 14],
            [1, 1408, 7, 7],
            [1, 144, 1, 1],
            [1, 144, 14, 14],
            [1, 144, 28, 28],
            [1, 144, 56, 56],
            [1, 144, 7, 7],
            [1, 1440, 14, 14],
            [1, 1440, 7, 7],
            [1, 1472, 14, 14],
            [1, 1472, 7, 7],
            [1, 1488, 14, 14],
            [1, 1488, 7, 7],
            [1, 15, 15, 512],
            [1, 1504, 14, 14],
            [1, 1504, 7, 7],
            [1, 1512, 14, 14],
            [1, 1512, 7, 7],
            [1, 1536, 10, 10],
            [1, 1536, 14, 14],
            [1, 1536, 7, 7],
            [1, 1568, 14, 14],
            [1, 1568, 7, 7],
            [1, 1584, 14, 14],
            [1, 1584, 7, 7],
            [1, 16, 1, 1],
            [1, 16, 112, 112],
            [1, 16, 14, 14],
            [1, 16, 160, 160],
            [1, 16, 224, 224],
            [1, 16, 28, 28],
            [1, 16, 56, 56],
            [1, 160, 14, 14],
            [1, 160, 28, 28],
            [1, 160, 56, 56],
            [1, 160, 7, 7],
            [1, 1600, 14, 14],
            [1, 1600, 7, 7],
            [1, 1632, 14, 14],
            [1, 1632, 7, 7],
            [1, 1664, 14, 14],
            [1, 1664, 7, 7],
            [1, 168, 1, 1],
            [1, 168, 28, 28],
            [1, 168, 56, 56],
            [1, 1680, 14, 14],
            [1, 1680, 7, 7],
            [1, 1696, 14, 14],
            [1, 1696, 7, 7],
            [1, 1728, 14, 14],
            [1, 1728, 7, 7],
            [1, 174, 1, 1],
            [1, 1760, 14, 14],
            [1, 1760, 7, 7],
            [1, 1776, 14, 14],
            [1, 1776, 7, 7],
            [1, 1792, 14, 14],
            [1, 1792, 7, 7],
            [1, 18, 1, 1],
            [1, 18, 14, 14],
            [1, 18, 28, 28],
            [1, 18, 56, 56],
            [1, 1824, 14, 14],
            [1, 1824, 7, 7],
            [1, 1856, 7, 7],
            [1, 1872, 14, 14],
            [1, 1872, 7, 7],
            [1, 1888, 7, 7],
            [1, 192, 14, 14],
            [1, 192, 17, 17],
            [1, 192, 28, 28],
            [1, 192, 35, 35],
            [1, 192, 56, 56],
            [1, 192, 7, 7],
            [1, 192, 8, 8],
            [1, 1920, 14, 14],
            [1, 1920, 7, 7],
            [1, 196, 1, 1],
            [1, 1968, 14, 14],
            [1, 1968, 7, 7],
            [1, 1984, 7, 7],
            [1, 1],
            [1, 20, 7, 7],
            [1, 200, 28, 28],
            [1, 200, 56, 56],
            [1, 200, 7, 7],
            [1, 2016, 14, 14],
            [1, 2016, 7, 7],
            [1, 2048, 10, 10],
            [1, 2048, 12, 12],
            [1, 2048, 14, 14],
            [1, 2048, 28, 28],
            [1, 2048, 7, 7],
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
    torch_output_tensor = torch.nn.functional.relu(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.relu(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
