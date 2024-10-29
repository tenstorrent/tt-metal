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
    "nightly": {
        "input_shape": gen_shapes([1, 1, 32 * 3], [6, 2048, 1024 * 3], [1, 1, 32 * 3], 16),
        "num_heads": list(range(1, 12)),
        "transpose_key": [True, False],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_shape"][-1] % (test_vector["num_heads"] * 3) != 0:
        return True, "Hidden size should be divisible by a total number of heads"
    if (test_vector["input_shape"][-1] // (test_vector["num_heads"] * 3)) % 32 != 0:
        return True, "Head size must be a multiple of 32"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row major layout is not supported"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    num_heads,
    transpose_key,
    input_dtype,
    input_layout,
    input_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    batch_size, sequence_size, hidden_size = input_shape
    head_size = hidden_size // (3 * num_heads)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape)

    intermediate_result = torch.reshape(torch_input_tensor, (batch_size, sequence_size, num_heads * 3, head_size))
    query, key, value = (
        intermediate_result[..., :num_heads, :],
        intermediate_result[..., num_heads : num_heads + num_heads, :],
        intermediate_result[..., num_heads + num_heads :, :],
    )

    query = torch.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    key = torch.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    value = torch.reshape(value, (batch_size, sequence_size, num_heads, head_size))

    query = torch.permute(query, (0, 2, 1, 3)).contiguous().clone()
    key = torch.permute(key, (0, 2, 1, 3)).contiguous().clone()
    value = torch.permute(value, (0, 2, 1, 3)).contiguous().clone()

    if transpose_key:
        key = torch.permute(key, (0, 1, 3, 2)).contiguous().clone()

    torch_output_tensors = (query, key, value)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    output_tensors = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor,
        kv_input_tensor=None,
        num_heads=num_heads,
        num_kv_heads=None,
        transpose_key=transpose_key,
        memory_config=output_memory_config,
    )
    e2e_perf = stop_measuring_time(start_time)

    passed = []
    output_string = ""
    for i in range(len(torch_output_tensors)):
        output_tensor = ttnn.to_torch(output_tensors[i])
        passed_, output_string_ = check_with_pcc(torch_output_tensors[i], output_tensor, 0.999)
        passed.append(passed_)
        output_string += output_string_ + ", "

    if all(passed):
        passed = True
    else:
        passed = False

    output_string = output_string[:-2]

    return [(passed, output_string), e2e_perf]
