# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 32)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 32)
        + gen_shapes([1, 1], [256, 256], [1, 1], 32),
        "input_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "xfail": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 4)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 4)
        + gen_shapes([1, 1], [256, 256], [1, 1], 4),
        "input_dtype": [ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    # Regression suite for issue #51976: fp32 lgamma had a coverage gap and a
    # plateau in z in [0.5, 0.75), the reflected range for x in (0.25, 0.75).
    # Sampling only from that window (instead of the wide 0.0001-100 range
    # above) avoids diluting the bug across a mostly-unaffected input space.
    "boundary_regression": {
        "input_shape": gen_shapes([1, 1, 32, 32], [1, 1, 256, 256], [1, 1, 32, 32], 8),
        "input_dtype": [ttnn.float32],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "low": [0.24],
        "high": [0.76],
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
    low=0.0001,
    high=100,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=low, high=high, dtype=torch.float32), input_dtype
    )(input_shape)
    golden_function = ttnn.get_golden_function(ttnn.lgamma)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.lgamma(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # lgamma is strictly decreasing on (0, 1.4616...), so a window fully inside
    # that range (like [0.24, 0.76]) must map to a strictly non-increasing
    # sequence when sorted by input, and must stay close to the golden value.
    # PCC alone can miss a narrow plateau or a coverage gap since it only
    # checks correlation, not absolute closeness, over the whole tensor.
    if high <= 1.4616:
        flat_input = torch_input_tensor.flatten()
        flat_golden = torch_output_tensor.flatten()
        flat_output = output_tensor.flatten()

        order = torch.argsort(flat_input)
        sorted_output = flat_output[order]
        diffs = sorted_output[1:] - sorted_output[:-1]
        is_monotonic = bool(torch.all(diffs <= 1e-4))

        # kernel's normal error here is ~5.3e-5, so tighter tolerance would fail for no reason
        is_close = bool(torch.allclose(flat_golden, flat_output, atol=1e-4, rtol=1e-4))

        if not is_monotonic or not is_close:
            message = f"boundary regression check failed: is_monotonic={is_monotonic}, is_close={is_close}"
            return [(False, message), e2e_perf]

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
