# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Ref: https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/docs/operations/aten.pow.Tensor_Scalar.md


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": [
            {"shape": [1, 1, 1024], "exponent": 2},
            {"shape": [1, 1, 1024], "exponent": 3.0},
            {"shape": [1, 1, 3072], "exponent": 3.0},
            {"shape": [1, 1, 4096], "exponent": 3.0},
            {"shape": [1, 1, 512], "exponent": 2},
            {"shape": [1, 1, 768], "exponent": 2},
            {"shape": [1, 10, 1024], "exponent": 2},
            {"shape": [1, 10, 512], "exponent": 2},
            {"shape": [1, 10, 768], "exponent": 2},
            {"shape": [1, 12, 3072], "exponent": 3.0},
            {"shape": [1, 14, 3072], "exponent": 3.0},
            {"shape": [1, 15, 1024], "exponent": 3.0},
            {"shape": [1, 15, 512], "exponent": 2},
            {"shape": [1, 3, 16, 16, 2], "exponent": 2},
            {"shape": [1, 3, 32, 32, 2], "exponent": 2},
            {"shape": [1, 3, 64, 64, 2], "exponent": 2},
            {"shape": [1, 45, 3072], "exponent": 3.0},
            {"shape": [1, 5, 4096], "exponent": 3.0},
            {"shape": [1, 7, 3072], "exponent": 3.0},
            {"shape": [1, 9, 128], "exponent": 3.0},
            {"shape": [1, 9, 16384], "exponent": 3.0},
            {"shape": [1, 9, 3072], "exponent": 3.0},
            {"shape": [1, 9, 4096], "exponent": 3.0},
            {"shape": [1, 9, 8192], "exponent": 3.0},
        ],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape["shape"])

    value = input_shape["exponent"]
    torch_output_tensor = torch.pow(torch_input_tensor_a, value)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.pow(input_tensor_a, exponent=value, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
