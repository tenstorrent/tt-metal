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
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "xfail": {
        "input_shape": gen_shapes([1, 1, 1, 2], [6, 12, 256, 256], [1, 1, 1, 2], 2)
        + gen_shapes([1, 1, 2], [12, 256, 256], [1, 1, 2], 2)
        + gen_shapes([1, 2], [256, 256], [1, 2], 2),
        "grad_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_c_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "grad_layout": [ttnn.TILE_LAYOUT],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "grad_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def str_to_float(x):
    try:
        return float(x)
    except:
        return 0.0


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    grad_dtype,
    input_a_dtype,
    input_b_dtype,
    input_c_dtype,
    grad_layout,
    input_a_layout,
    input_b_layout,
    input_c_layout,
    grad_memory_config,
    input_a_memory_config,
    input_b_memory_config,
    input_c_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_grad_tensor = gen_func_with_cast_tt(partial(torch_random, low=-10, high=10, dtype=torch.float32), grad_dtype)(
        input_shape
    )

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)
    torch_input_tensor_a.requires_grad = True
    torch_input_tensor_a.retain_grad()

    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(input_shape)
    torch_input_tensor_b.requires_grad = True
    torch_input_tensor_b.retain_grad()

    torch_input_tensor_c = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_c_dtype
    )(input_shape)
    torch_input_tensor_c.requires_grad = True
    torch_input_tensor_c.retain_grad()

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    torch_output_tensor = golden_function(
        torch_grad_tensor, torch_input_tensor_a, torch_input_tensor_b, torch_input_tensor_c
    )

    grad_tensor = ttnn.from_torch(
        torch_grad_tensor,
        dtype=grad_dtype,
        layout=grad_layout,
        device=device,
        memory_config=grad_memory_config,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a.detach().clone(),
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b.detach().clone(),
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    input_tensor_c = ttnn.from_torch(
        torch_input_tensor_c.detach().clone(),
        dtype=input_c_dtype,
        layout=input_c_layout,
        device=device,
        memory_config=input_c_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.lerp_bw(
        grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c, memory_config=output_memory_config
    )

    for i in range(len(output_tensor)):
        output_tensor[i] = ttnn.to_torch(output_tensor[i])
    e2e_perf = stop_measuring_time(start_time)

    pcc = [True, 1.0]

    for i in range(len(output_tensor)):
        pcc_tmp = check_with_pcc(torch_output_tensor[i], output_tensor[i], 0.999)
        pcc[0] = pcc[0] and pcc_tmp[0]
        pcc[1] = min(pcc[1], str_to_float(pcc_tmp[1]))

    pcc[1] = str(pcc[1])
    # print(f"pcc {pcc} - {grad_dtype}, {input_a_dtype}, {input_b_dtype}")
    return [pcc, e2e_perf]
