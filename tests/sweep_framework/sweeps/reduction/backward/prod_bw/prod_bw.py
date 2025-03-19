# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape_rm
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
        "input_shape": gen_shapes([1, 1, 32, 32], [6, 12, 256, 256], [1, 1, 32, 32], 16)
        + gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 2)
        + gen_shapes([3, 4, 5, 6], [6, 12, 256, 256], [7, 8, 9, 10], 2)
        + gen_shapes([1, 1, 1, 1], [6, 12, 187, 188], [1, 1, 1, 1], 7)
        + gen_shapes([1, 32, 64], [6, 48, 128], [1, 1, 1], 2)
        + gen_shapes([1, 32, 64], [6, 77, 128], [1, 1, 1], 7)
        + gen_shapes([1, 32, 64], [6, 10222, 1023], [1, 1, 1], 8)
        + gen_shapes([1, 1], [6, 6], [1, 1], 2)
        + gen_shapes([1, 1], [7, 7], [1, 2], 3)
        + gen_shapes([1, 1], [8, 8], [1, 3], 4)
        + gen_shapes([1], [4], [1], 2)
        + gen_shapes([1], [14], [11], 12)
        + gen_shapes([1], [24], [21], 22),
        "dim": [
            0,
            1,
            2,
            3,
            None,
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
        ],
        "keepdim": [True, False],
        "grad_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "grad_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if (test_vector["input_a_dtype"] == ttnn.float32 and test_vector["grad_dtype"] == ttnn.float32) or (
            test_vector["input_a_dtype"] == ttnn.bfloat16 or test_vector["grad_dtype"] == ttnn.bfloat16
        ):
            return False, None
        else:
            return True, "Row major is only supported for fp32 & fp16"
    if not test_vector["keepdim"]:
        return True, "keepdim = false is not supported"
    if not isinstance(test_vector["dim"], int):
        return True, "dim can only be integer value"

    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. pOtherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    dim,
    keepdim,
    grad_dtype,
    input_a_dtype,
    input_layout,
    grad_memory_config,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if (input_a_dtype == ttnn.float32 or grad_dtype == ttnn.float32) and ttnn.device.is_grayskull(device):
        return [(False, "Dest Fp32 mode is not supported for arch grayskull"), 0]

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=0, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)
    torch_input_tensor_a.requires_grad = True
    torch_input_tensor_a.retain_grad()

    intermediate_result = torch.prod(torch_input_tensor_a, dim=dim, keepdim=keepdim)
    torch_grad_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), grad_dtype
    )(intermediate_result.shape)

    intermediate_result.backward(gradient=torch_grad_tensor)
    torch_output_tensor = torch_input_tensor_a.grad

    grad_tensor = ttnn.from_torch(
        torch_grad_tensor,
        dtype=grad_dtype,
        layout=input_layout,
        device=device,
        memory_config=grad_memory_config,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.prod_bw(grad_tensor, input_tensor_a, dim=dim, memory_config=output_memory_config)[0]
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
