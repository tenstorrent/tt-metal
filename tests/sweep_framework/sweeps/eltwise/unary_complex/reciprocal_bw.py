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


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 16)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 16)
        + gen_shapes([1, 1], [256, 256], [1, 1], 16),
        "grad_dtype": [ttnn.bfloat16],
        "input_a_dtype": [ttnn.bfloat16],
        "grad_layout": [ttnn.TILE_LAYOUT],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "grad_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
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
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    grad_dtype,
    input_a_dtype,
    grad_layout,
    input_a_layout,
    grad_memory_config,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_grad_tensor_r = gen_func_with_cast_tt(
        partial(torch_random, low=0.01, high=100, dtype=torch.float32), grad_dtype
    )(input_shape)
    torch_grad_tensor_r.requires_grad = True
    torch_grad_tensor_r.retain_grad()

    torch_grad_tensor_c = gen_func_with_cast_tt(
        partial(torch_random, low=0.01, high=100, dtype=torch.float32), grad_dtype
    )(input_shape)
    torch_grad_tensor_c.requires_grad = True
    torch_grad_tensor_c.retain_grad()

    torch_input_tensor_ar = gen_func_with_cast_tt(
        partial(torch_random, low=0.01, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)
    torch_input_tensor_ar.requires_grad = True
    torch_input_tensor_ar.retain_grad()

    torch_input_tensor_ac = gen_func_with_cast_tt(
        partial(torch_random, low=0.01, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)
    torch_input_tensor_ac.requires_grad = True
    torch_input_tensor_ac.retain_grad()

    torch_grad_tensor = torch.complex(torch_grad_tensor_r.to(torch.float32), torch_grad_tensor_c.to(torch.float32))
    torch_input_tensor_a = torch.complex(
        torch_input_tensor_ar.to(torch.float32), torch_input_tensor_ac.to(torch.float32)
    )

    golden_function = ttnn.get_golden_function(ttnn.reciprocal_bw)
    torch_output_tensor = golden_function(torch_grad_tensor, torch_input_tensor_a)

    grad_tensor_r = ttnn.from_torch(
        torch_grad_tensor_r,
        dtype=grad_dtype,
        layout=grad_layout,
        device=device,
        memory_config=grad_memory_config,
    )

    grad_tensor_c = ttnn.from_torch(
        torch_grad_tensor_c, dtype=grad_dtype, layout=grad_layout, device=device, memory_config=grad_memory_config
    )

    input_tensor_ar = ttnn.from_torch(
        torch_input_tensor_ar,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_ac = ttnn.from_torch(
        torch_input_tensor_ac,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    grad_tensor = ttnn.complex_tensor(grad_tensor_r, grad_tensor_c)
    input_tensor_a = ttnn.complex_tensor(input_tensor_ar, input_tensor_ac)

    start_time = start_measuring_time()
    output_tensor = ttnn.reciprocal_bw(grad_tensor, input_tensor_a, memory_config=output_memory_config)

    for i in range(len(output_tensor)):
        result_real = ttnn.to_torch(output_tensor[i].real).to(torch.float32)
        result_imag = ttnn.to_torch(output_tensor[i].imag).to(torch.float32)
        output_tensor[i] = torch.complex(result_real, result_imag)
    e2e_perf = stop_measuring_time(start_time)
    pcc = [True, 1.0]

    for i in range(len(output_tensor)):
        pcc_tmp = check_with_pcc(torch.view_as_real(torch_output_tensor[i]), torch.view_as_real(output_tensor[i]), 0.99)
        pcc[0] = pcc[0] and pcc_tmp[0]
        pcc[1] = min(pcc[1], str_to_float(pcc_tmp[1]))

    # print(f"pcc {pcc} input_a_dtype {input_a_dtype}")
    return [pcc, e2e_perf]
