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
from models.common.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 130

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 64, 64], [6, 12, 256, 256], [1, 1, 32, 32], 16),
        "op": [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.SUB, ttnn.BcastOpMath.MUL],
        "axis": [ttnn.BcastOpDim.H, ttnn.BcastOpDim.W, ttnn.BcastOpDim.HW],
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
    op,
    axis,
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
    input_shape_2 = [input_shape[-4], input_shape[-3], input_shape[-2], input_shape[-1]]

    if axis == ttnn.BcastOpDim.H or axis == ttnn.BcastOpDim.HW:
        input_shape_2[-2] = 1

    if axis == ttnn.BcastOpDim.W or axis == ttnn.BcastOpDim.HW:
        input_shape_2[-1] = 1

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(input_shape_2)

    if op == ttnn.BcastOpMath.ADD:
        torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)
    elif op == ttnn.BcastOpMath.SUB:
        torch_output_tensor = torch.sub(torch_input_tensor_a, torch_input_tensor_b)
    else:
        torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)

    if axis == ttnn.BcastOpDim.H:
        torch_input_tensor_b = torch_input_tensor_b.repeat(1, 1, 32, 1)
    elif axis == ttnn.BcastOpDim.W:
        torch_input_tensor_b = torch_input_tensor_b.repeat(1, 1, 1, 32)
    else:
        torch_input_tensor_b = torch_input_tensor_b.repeat(1, 1, 32, 32)

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

    start_time = start_measuring_time()
    output_tensor = ttnn.bcast(input_tensor_a, input_tensor_b, op, axis)
    output_tensor = ttnn.to_torch(output_tensor)

    e2e_perf = stop_measuring_time(start_time)
    pcc_res = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    # print(f"e2e_perf {e2e_perf / 1000000}ms")
    # print(f"pcc_res {pcc_res}")

    return [pcc_res, e2e_perf]
