# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt, gen_bin

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.


parameters = {
    "nightly": {
        "input_shape": [
            {
                "tensor1": [19, 19],
                "tensor2": [1],
                "tensor3": [19, 19],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.float32",
                "input_dtype_c": "ttnn.float32",
            },
            {
                "tensor1": [1, 1, 10, 10],
                "tensor2": [1],
                "tensor3": [1, 1, 10, 10],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 12, 12],
                "tensor2": [1],
                "tensor3": [1, 1, 12, 12],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 14, 14],
                "tensor2": [1],
                "tensor3": [1, 1, 14, 14],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 16, 16],
                "tensor2": [1],
                "tensor3": [1, 1, 16, 16],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 19, 19],
                "tensor2": [1],
                "tensor3": [1, 1, 19, 19],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 256, 256],
                "tensor2": [1],
                "tensor3": [1, 1, 256, 256],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 25, 25],
                "tensor2": [1],
                "tensor3": [1, 1, 25, 25],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 32, 32],
                "tensor2": [1],
                "tensor3": [1, 1, 32, 32],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 5, 5],
                "tensor2": [1],
                "tensor3": [1, 1, 5, 5],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 6, 6],
                "tensor2": [1],
                "tensor3": [1, 1, 6, 6],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 7, 7],
                "tensor2": [1],
                "tensor3": [1, 1, 7, 7],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 1, 9, 9],
                "tensor2": [1],
                "tensor3": [1, 1, 9, 9],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [1, 920],
                "tensor2": [1],
                "tensor3": [1, 920],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [2, 1, 7, 7],
                "tensor2": [1],
                "tensor3": [2, 1, 7, 7],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.bfloat16",
                "input_dtype_c": "ttnn.bfloat16",
            },
            {
                "tensor1": [6, 6],
                "tensor2": [1],
                "tensor3": [6, 6],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.float32",
                "input_dtype_c": "ttnn.float32",
            },
            {
                "tensor1": [7, 7],
                "tensor2": [1],
                "tensor3": [7, 7],
                "input_dtype_a": "ttnn.bfloat16",
                "input_dtype_b": "ttnn.float32",
                "input_dtype_c": "ttnn.float32",
            },
            # {
            #     "tensor1": [1, 45],
            #     "tensor2": [1],
            #     "tensor3": [1, 45],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
            # {
            #     "tensor1": [1, 5],
            #     "tensor2": [1],
            #     "tensor3": [1, 5],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
            # {
            #     "tensor1": [1, 6],
            #     "tensor2": [1],
            #     "tensor3": [1, 6],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
            # {
            #     "tensor1": [1, 197],
            #     "tensor2": [196, 197],
            #     "tensor3": [196, 197],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
            # {
            #     "tensor1": [1, 19],
            #     "tensor2": [1, 19],
            #     "tensor3": [1, 19],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
            # {
            #     "tensor1": [1, 1],
            #     "tensor2": [1, 1],
            #     "tensor3": [1, 1],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
            # {
            #     "tensor1": [15, 15],
            #     "tensor2": [15, 15],
            #     "tensor3": [15, 15],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
            # {
            #     "tensor1": [197, 1],
            #     "tensor2": [197, 197],
            #     "tensor3": [197, 197],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
            # {
            #     "tensor1": [19],
            #     "tensor2": [19],
            #     "tensor3": [1],
            #     "input_dtype_a": "ttnn.bfloat16",
            #     "input_dtype_b": "ttnn.int32",
            #     "input_dtype_c": "ttnn.int32",
            # },
        ],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if (
        test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT
        or test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT
        or test_vector["input_c_layout"] == ttnn.ROW_MAJOR_LAYOUT
    ):
        return True, "Row Major layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
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
    torch.manual_seed(0)

    if input_shape["input_dtype_a"] == "ttnn.bfloat16":
        input_dtype_a = ttnn.bfloat16
    elif input_shape["input_dtype_a"] == "ttnn.float32":
        input_dtype_a = ttnn.float32
    elif input_shape["input_dtype_a"] == "ttnn.int32":
        input_dtype_a = ttnn.int32

    if input_shape["input_dtype_b"] == "ttnn.bfloat16":
        input_dtype_b = ttnn.bfloat16
    elif input_shape["input_dtype_b"] == "ttnn.float32":
        input_dtype_b = ttnn.float32
    elif input_shape["input_dtype_b"] == "ttnn.int32":
        input_dtype_b = ttnn.int32

    if input_shape["input_dtype_c"] == "ttnn.bfloat16":
        input_dtype_c = ttnn.bfloat16
    elif input_shape["input_dtype_c"] == "ttnn.float32":
        input_dtype_c = ttnn.float32
    elif input_shape["input_dtype_c"] == "ttnn.int32":
        input_dtype_c = ttnn.int32

    torch_input_tensor_a = gen_func_with_cast_tt(gen_bin, input_dtype_a)(input_shape["tensor1"])
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype_b
    )(input_shape["tensor2"])
    torch_input_tensor_c = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype_c
    )(input_shape["tensor3"])

    torch_output_tensor = torch.where(torch_input_tensor_a > 0, torch_input_tensor_b, torch_input_tensor_c)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype_a,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_dtype_b,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    input_tensor_c = ttnn.from_torch(
        torch_input_tensor_c,
        dtype=input_dtype_c,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.where(input_tensor_a, input_tensor_b, input_tensor_c, memory_config=output_memory_config)
    # ToDo: Update it once the tensor layout support with rank < 2 is supported in mid of Jan
    output_tensor = ttnn.to_torch(result, torch_rank=len(input_shape["tensor1"]))
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
