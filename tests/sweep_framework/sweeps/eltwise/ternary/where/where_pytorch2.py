# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt, gen_bin

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "where_5": {
        "input_shape": [
            {"shape1": [1, 1, 1, 46], "shape2": [1, 12, 1, 46], "shape3": []},
            {"shape1": [1, 1, 1, 6], "shape2": [1, 16, 1, 6], "shape3": []},
            # {"shape1": [1, 1, 1, "s10 + 1"], "shape2": [1, 12, 1, "s10 + 1"], "shape3": []},
            # {"shape1": [1, 1, 1, "s10 + 1"], "shape2": [1, 16, 1, "s10 + 1"], "shape3": []},
            {"shape1": [1, 1, 256], "shape2": [1, 1, 256], "shape3": []},
            {"shape1": [1, 1, 45, 45], "shape2": [1, 12, 45, 45], "shape3": []},
            {"shape1": [1, 1, 5, 5], "shape2": [1, 16, 5, 5], "shape3": []},
            {"shape1": [1, 1, 7, 7], "shape2": [1, 12, 7, 7], "shape3": []},
            {"shape1": [1, 1], "shape2": [1, 1], "shape3": [1, 1]},
            # {"shape1": [1, "s0", 256], "shape2": [1, "s0", 256], "shape3": []},
            {"shape1": [10, 10], "shape2": [10, 10], "shape3": [10, 10]},
            {"shape1": [15, 15], "shape2": [15, 15], "shape3": [15, 15]},
            {"shape1": [17, 17], "shape2": [17, 17], "shape3": [17, 17]},
            {"shape1": [2, 2], "shape2": [2, 2], "shape3": [2, 2]},
            # {"shape1": ["s0 + 1", "s0 + 1"], "shape2": ["s0 + 1", "s0 + 1"], "shape3": []},
        ],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_c_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
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
    input_c_dtype,
    input_a_layout,
    input_b_layout,
    input_c_layout,
    input_a_memory_config,
    input_b_memory_config,
    input_c_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = gen_func_with_cast_tt(gen_bin, input_a_dtype)(input_shape["shape1"])
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(input_shape["shape2"])
    torch_input_tensor_c = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_c_dtype
    )(input_shape["shape3"])

    torch_output_tensor = torch.where(torch_input_tensor_a > 0, torch_input_tensor_b, torch_input_tensor_c)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    input_tensor_c = ttnn.from_torch(
        torch_input_tensor_c,
        dtype=input_c_dtype,
        layout=input_c_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.where(input_tensor_a, input_tensor_b, input_tensor_c, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
