# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "suite_1": {
        "input_shape": [
            [8, 1, 33, 256],
            [8, 1, 256, 32],
            [8, 8, 256, 384],
            [8, 5, 13, 512],
            [8, 5, 32, 512],
            [1, 1, 32, 16384],
        ],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    *,
    device,
) -> list:
    low = -1e6
    high = 1e6
    torch_input_tensor_a = torch.rand(input_shape).bfloat16() * (high - low) + low
    torch_input_tensor_b = torch.rand(input_shape).bfloat16() * (high - low) + low
    torch_output_tensor = torch_input_tensor_a.logical_and_(torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_b_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    start_time = start_measuring_time()
    ttnn.logical_and_(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(input_tensor_a)
    e2e_perf = stop_measuring_time(start_time)

    return [assert_equal(torch_output_tensor, output_tensor), e2e_perf]
