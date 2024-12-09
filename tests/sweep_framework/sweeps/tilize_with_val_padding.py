# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
import math
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import comp_equal, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

from tt_lib.utils import tilize as tilize_util

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


def tilize_with_val_padding(x, output_tensor_shape, pad_value):
    pad = torch.nn.functional.pad(
        x,
        tuple(j for i in reversed(range(len(x.shape))) for j in (0, output_tensor_shape[i] - x.shape[i])),
        value=pad_value,
    )
    tilized = tilize_util(pad)
    return tilized


def _nearest_32(x):
    return math.ceil(x / 32) * 32


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "xfail": {
        "input_shape": gen_shapes([1, 1, 32, 32], [6, 12, 128, 128], [1, 1, 32, 32], 32),
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.TILE_LAYOUT:
        return True, "Tile layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    number_generated = torch.tensor(1, dtype=torch.bfloat16).uniform_(-100, 100).item()

    padded_shape = [
        input_shape[0],
        input_shape[1],
        _nearest_32(input_shape[2]),
        _nearest_32(input_shape[3]),
    ]

    torch_output_tensor = tilize_with_val_padding(torch_input_tensor_a, padded_shape, number_generated)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.tilize_with_val_padding(
        input_tensor_a, padded_shape, number_generated, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [comp_equal(torch_output_tensor, output_tensor), e2e_perf]
