# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import (
    gen_shapes,
    sanitize_shape_rm,
    gen_complex_tensor,
    complex_from_torch,
)
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
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 8)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 8)
        + gen_shapes([1, 1], [256, 256], [1, 1], 8),
        "grad_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "grad_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Input to eltwise binary must be tilized"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and (
        test_vector["input_a_dtype"] == ttnn.bfloat8_b
        or test_vector["input_b_dtype"] == ttnn.bfloat8_b
        or test_vector["grad_dtype"] == ttnn.bfloat8_b
    ):
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    grad_dtype,
    input_a_dtype,
    input_b_dtype,
    input_layout,
    grad_memory_config,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_grad_tensor = gen_complex_tensor(input_shape, -100, 100, grad_dtype)
    torch_input_tensor_a = gen_complex_tensor(input_shape, -100, 100, input_a_dtype)
    torch_input_tensor_b = gen_complex_tensor(input_shape, -100, 100, input_b_dtype)

    torch_input_tensor_a.requires_grad = True
    torch_input_tensor_b.requires_grad = True

    torch_input_tensor_a.retain_grad()
    torch_input_tensor_b.retain_grad()

    intermediate_result = torch.sub(torch_input_tensor_a, torch_input_tensor_b)
    intermediate_result.backward(gradient=torch_grad_tensor)
    torch_output_tensors = [torch_input_tensor_a.grad, torch_input_tensor_b.grad]

    grad_tensor = complex_from_torch(
        torch_grad_tensor,
        dtype=grad_dtype,
        layout=input_layout,
        device=device,
        memory_config=grad_memory_config,
    )
    input_tensor_a = complex_from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = complex_from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    output_tensors = ttnn.sub_bw(
        grad_tensor, input_tensor_a, input_tensor_b, alpha=0.0, memory_config=output_memory_config
    )
    e2e_perf = stop_measuring_time(start_time)

    passed = []
    output_string = ""
    for i in range(len(torch_output_tensors)):
        torch_output_tensor = torch_output_tensors[i]
        output_tensor = torch.complex(
            ttnn.to_torch(output_tensors[i].real).to(torch.float32),
            ttnn.to_torch(output_tensors[i].imag).to(torch.float32),
        )
        passed_, output_string_ = check_with_pcc(
            torch.view_as_real(torch_output_tensor.clone()), torch.view_as_real(output_tensor.clone()), 0.999
        )
        passed.append(passed_)
        output_string += output_string_ + ", "

    if all(passed):
        passed = True
    else:
        passed = False

    return [(passed, output_string), e2e_perf]
