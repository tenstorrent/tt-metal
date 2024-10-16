# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.utils import gen_shapes, gen_low_high_scalars
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

parameters = {
    "nightly": {
        "input_specs": [
            {"shape": [1, 1, 38, 38], "min": 1e-12},
            {"shape": [1, 1], "min": 1e-12},
            {"shape": [1, 24, 64, 1], "min": 1e-12},
            {"shape": [1, 32, 64, 1], "min": 1e-12},
            {"shape": [16, 6, 64, 1], "min": 1e-12},
            {"shape": [16, 8, 64, 1], "min": 1e-12},
            {"shape": [4, 12, 64, 1], "min": 1e-12},
            {"shape": [4, 16, 64, 1], "min": 1e-12},
            {"shape": [64, 3, 64, 1], "min": 1e-12},
            {"shape": [64, 4, 64, 1], "min": 1e-12},
        ],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def run(
    input_specs,
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
    )(input_specs["shape"])

    golden_function = ttnn.get_golden_function(ttnn.clamp)
    torch_output_tensor = golden_function(torch_input_tensor_a, input_specs["min"])

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.clamp(input_tensor_a, input_specs["min"], memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
