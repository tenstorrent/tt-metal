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
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 128, 128], [1, 1, 1, 1], 16)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 132, 1], 16)
        + gen_shapes([1, 1], [256, 256], [1, 1], 16),
        "input_a_dtype": [ttnn.int32],
        "output_dtype": [ttnn.int32],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def mesh_device_fixture():
    device = ttnn.open_device(device_id=0)
    assert ttnn.device.is_wormhole_b0(device), "This op is available for Wormhole_B0 only"
    yield (device, "Wormhole_B0")
    ttnn.close_device(device)
    del device


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    output_dtype,
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
        partial(torch_random, low=0, high=2147483648, dtype=torch.int64), input_a_dtype
    )(input_shape)
    torch_optional_output = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.int32), output_dtype
    )(input_shape)
    scalar = torch.randint(0, 2147483648, (1,), dtype=torch.int32).item()

    golden_function = ttnn.get_golden_function(ttnn.bitwise_xor)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    output_tensor = ttnn.from_torch(
        torch_optional_output,
        dtype=output_dtype,
        layout=input_layout,
        device=device,
        memory_config=output_memory_config,
    )

    start_time = start_measuring_time()
    ttnn.bitwise_xor(input_tensor_a, scalar, output_tensor=output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
