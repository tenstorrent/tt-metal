# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "batch_sizes": [(1, 2), (3, 6)],
        "height": [384, 1024],
        "width": [1024, 4096],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_dtype"] == ttnn.bfloat8_b:
        return True, "Skipped as ROW_MAJOR_LAYOUT and ttnn.bfloat8_b not supported"
    return False, None


def check_output(torch_output_tensor, output_tensor):
    status = list(torch_output_tensor.shape) == list(output_tensor.shape)
    msg = ""
    msg = "pass" if status else "fail"

    return status, msg


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    batch_sizes,
    height,
    width,
    input_dtype,
    output_memory_config,
    layout,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    input_shape = (*batch_sizes, height, width)

    torch_output_tensor = torch.empty(input_shape)

    start_time = start_measuring_time()

    output_tensor = ttnn.empty(input_shape, input_dtype, layout, device=device, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_output(torch_output_tensor, output_tensor), e2e_perf]
