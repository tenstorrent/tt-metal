# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
import random

from tests.sweep_framework.sweep_utils.utils import (
    gen_shapes,
    sanitize_shape_rm,
    unary_ops_dict,
    get_unary_op_args,
    gen_unary_chain_spec,
    tensor_to_dtype,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 8)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 8)
        + gen_shapes([1, 1], [256, 256], [1, 1], 8),
        "unary_chain_spec": list(gen_unary_chain_spec()),
        "input_a_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    unary_ops, param_list, input_range = test_vector["unary_chain_spec"].values()
    starting_ops = ["acos", "asin", "atan"]
    ending_ops = ["isinf", "isfinite", "isposinf", "isneginf", "unary_ne", "unary_gt", "unary_lt", "isnan"]
    if len(unary_ops) > 1:
        starting_ops_invalid = list(set(starting_ops).intersection(set(unary_ops[1:])))
        ending_ops_invalid = list(set(ending_ops).intersection(set(unary_ops[:-1])))
        if starting_ops_invalid:
            return (
                True,
                f"Unary op(s) {starting_ops_invalid} must appear at the beginning of the ops sequence to ensure that they are applied on a tensor with values within a valid range",
            )
        if ending_ops_invalid:
            return (
                True,
                f"Unary op(s) {ending_ops_invalid} must appear at the ending of the ops sequence because they return tensor with boolean values",
            )
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is only supported on tile layout"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    unary_chain_spec,
    input_a_dtype,
    input_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    unary_ops, param_list, input_range = unary_chain_spec.values()

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=input_range[0], high=input_range[1], dtype=torch.float32), input_a_dtype
    )(input_shape)
    torch_output_tensor = torch_input_tensor_a.detach().clone()

    unary_op_types = []
    for i in range(len(unary_ops)):
        unary_op = unary_ops[i]
        param = param_list[i]
        golden_function, unary_op_type = get_unary_op_args(unary_op, torch_output_tensor, param)
        unary_op_types.append(unary_op_type)
        torch_output_tensor = tensor_to_dtype(golden_function(), input_a_dtype)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.unary_chain(input_tensor_a, unary_op_types, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
