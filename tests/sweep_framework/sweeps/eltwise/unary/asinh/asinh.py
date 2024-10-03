# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.utils import gen_shapes
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
        "batch_sizes": [(1,)],
        "height": [384, 1024],
        "width": [1024, 4096],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "layout": [ttnn.TILE_LAYOUT],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row Major layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    batch_sizes,
    height,
    width,
    input_shape,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    input_shape = (*batch_sizes, height, width)

    low = -1e-6
    high = 1e6

    torch_input_tensor = torch_random(input_shape, low, high, dtype=torch.float32)
    torch_output_tensor = torch.asinh(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, memory_config=input_memory_config, layout=layout
    )
    start_time = start_measuring_time()

    output_tensor = ttnn.asinh(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
