# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 32)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 32)
        + gen_shapes([1, 1], [256, 256], [1, 1], 32),
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_dtype,
    input_layout,
    input_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=0.0001, high=100, dtype=torch.float32), input_dtype
    )(input_shape)
    golden_function = ttnn.get_golden_function(ttnn.digamma)
    torch_output_tensor = golden_function(torch_input_tensor)

    print(f"{input_shape} {input_dtype} {input_layout} {input_memory_config} {output_memory_config}")

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.digamma(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    print(pcc)
    return [pcc, e2e_perf]


# Run sweeps locally
from tests.sweep_framework.framework.permutations import *

start_time = start_measuring_time()
for suite in parameters.keys():
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    suite_vectors = list(permutations(parameters[suite]))
    print(len(suite_vectors))
    for vector in suite_vectors:
        # invalidate_res = invalidate_vector(vector)
        # if invalidate_res[0]:
        #     print(f"Invalidated: {invalidate_res[1]}")
        #     continue
        try:
            passed, _ = run(**vector, device=device)
            # if passed[0] != True:
            #     print(passed)
        except Exception as e:
            print(e)

    ttnn.close_device(device)

e2e_perf = stop_measuring_time(start_time)
print(f"time {e2e_perf / 1000000000}s")
