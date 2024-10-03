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
            [1, 1, 3072],
            [1, 10, 3072],
            [1, 10, 768],
            [1, 1024, 2560],
            [1, 1024, 512],
            [1, 1024, 640],
            [1, 1200, 1280],
            [1, 1370, 5120],
            [1, 14, 14, 1536],
            [1, 14, 14, 2048],
            [1, 1445, 768],
            [1, 1500, 3072],
            [1, 1536],
            [1, 16, 16, 1536],
            [1, 16, 16, 2048],
            [1, 16, 3072],
            [1, 16384, 128],
            [1, 19, 4096],
            [1, 19200, 256],
            [1, 196, 3072],
            [1, 197, 3072],
            [1, 197, 4096],
            [1, 201, 3072],
            [1, 2048, 768],
            [1, 24, 3072],
            [1, 25, 3072],
            [1, 256, 1024],
            [1, 256, 1280],
            [1, 256, 256],
            [1, 256, 4096],
            [1, 256, 5120],
            [1, 28, 28, 1024],
            [1, 28, 28, 768],
            [1, 300, 2048],
            [1, 32, 32, 1024],
            [1, 32, 32, 768],
            [1, 4, 3072],
            [1, 4096, 1280],
            [1, 4096, 256],
            [1, 4800, 512],
            [1, 50, 3072],
            [1, 50, 4096],
            [1, 56, 56, 384],
            [1, 56, 56, 512],
            [1, 64, 5120],
            [1, 64, 64, 384],
            [1, 64, 64, 512],
            [1, 7, 18176],
            [1, 7, 7, 3072],
            [1, 7, 7, 4096],
            [1, 768, 1500],
            [1, 768, 3000],
            [1, 768, 384],
            [1, 8, 8, 3072],
            [1, 8, 8, 4096],
        ],
        "input_a_dtype": [ttnn.bfloat16],
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
        partial(torch_random, low=-100, high=100, dtype=torch.float16), input_a_dtype
    )(input_shape)
    torch_output_tensor = torch.nn.functional.gelu(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.gelu(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
