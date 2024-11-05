# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


parameters = {
    "nightly": {
        "input_specs": [
            {"shape": [0, 1], "other": 1.0},
            {"shape": [1, 12, 7, 7], "other": []},
            {"shape": [1, 16, 1, 6], "other": []},
            {"shape": [1, 23, 40, 1], "other": [128]},
            {"shape": [1, 23, 40], "other": [1, 1, 40]},
            {"shape": [1, 23, 40], "other": [1, 23, 1]},
            {"shape": [1, 512], "other": [1, 1]},
            {"shape": [2, 512], "other": [2, 1]},
            {"shape": [3, 320, 320], "other": [3, 1, 1]},
            {"shape": [3, 480, 640], "other": [3, 1, 1]},
            {"shape": [96, 80], "other": [80]},
            {"shape": [], "other": []},
        ],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def run(
    input_specs,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    input_shape = input_specs["shape"]
    if len(input_shape) == 0:
        torch_input_tensor_a = torch.empty([])
    else:
        torch_input_tensor_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(input_shape)

    other = input_specs["other"]
    if isinstance(other, (int, float)):
        torch_other_tensor = torch.tensor(other, dtype=torch.float32)
    elif len(other) == 0:
        torch_other_tensor = torch.empty([])
    else:
        torch_other_tensor = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(other)

    golden_function = ttnn.get_golden_function(ttnn.divide)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_other_tensor)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_other_tensor,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()

    output_tensor = ttnn.divide(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
