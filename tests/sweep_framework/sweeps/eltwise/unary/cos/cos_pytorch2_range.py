# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import assert_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random, is_close
from tests.sweep_framework.sweep_utils.utils import gen_shapes

# Ref: https://github.com/tenstorrent/tt-torch/blob/main/docs/ops/ttnn/ttnn.cos.md

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 32, 32], [6, 12, 256, 256], [1, 1, 30, 30], 16)
        + gen_shapes([1, 32, 32], [12, 256, 256], [1, 32, 32], 16)
        + gen_shapes([32, 32], [256, 256], [32, 32], 32),
        "input_a_dtype": [ttnn.float32, ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Adjust input range based on dtype.
def get_input_range(dtype: str) -> Tuple[float, float]:
    if dtype == ttnn.float32:
        return -1e3, 1e3
    elif dtype == ttnn.bfloat16:
        return -5.0, 5.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    input_range = get_input_range(input_a_dtype)
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=input_range[0], high=input_range[1], dtype=torch.float32), input_a_dtype
    )(input_shape)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.cos(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [
        assert_with_pcc(torch_output_tensor, output_tensor, 0.999),
        e2e_perf,
    ]
