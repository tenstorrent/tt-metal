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
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 16)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 16)
        + gen_shapes([1, 1], [256, 256], [1, 1], 16),
        "input_a_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def mesh_device_fixture():
    device = ttnn.open_device(device_id=0)
    assert ttnn.device.is_wormhole_b0(device), "This op is supported on Wormhole"
    yield (device, "Wormhole_B0")
    ttnn.close_device(device)
    del device


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Input to eltwise binary must be tilized"
    if test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is not supported"
    if test_vector["input_layout"] == "ROW_MAJOR_LAYOUT" and test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    scalar = torch.tensor(1, dtype=torch.bfloat16).uniform_(0.1, 10.0).item()
    signs_b = torch.randint(0, 2, (1,)).item() * 2 - 1
    scalar *= signs_b

    golden_function = ttnn.get_golden_function(ttnn.floor_div)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.floor_div(input_tensor_a, scalar, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
