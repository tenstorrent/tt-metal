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

TIMEOUT = 30

random.seed(0)


parameters = {
    "nightly": {
        "input_specs": [
            {"shape": [0, 1], "other": [0, 1]},
            {"shape": [0], "other": [0]},
            {"shape": [1, 1, 1, 42], "other": -3.0},
            {"shape": [1, 1, 1, 42], "other": -3.75},
            {"shape": [1, 1, 1, 42], "other": 0.5},
            {"shape": [1, 1, 1, 42], "other": 1},
            {"shape": [1, 1, 1, 42], "other": 2.25},
            {"shape": [1, 1, 1, 42], "other": [1, 1, 1, 42]},
            {"shape": [1, 1, 32, 1], "other": -3.0},
            {"shape": [1, 1, 32, 1], "other": -3.75},
            {"shape": [1, 1, 32, 1], "other": 0.5},
            {"shape": [1, 1, 32, 1], "other": 1},
            {"shape": [1, 1, 32, 1], "other": 2.25},
            {"shape": [1, 1, 32, 1], "other": [1, 1, 32, 1]},
            {"shape": [1, 1024, 1, 1], "other": [1, 1024, 1, 1]},
            {"shape": [1, 10], "other": [10, 1]},
            {"shape": [1, 128, 1, 1], "other": [1, 128, 1, 1]},
            {"shape": [1, 15], "other": [15, 1]},
            {"shape": [1, 17], "other": [17, 1]},
            {"shape": [1, 1], "other": [1, 1]},
            {"shape": [1, 2048, 1, 1], "other": [1, 2048, 1, 1]},
            {"shape": [1, 256, 1, 1], "other": [1, 256, 1, 1]},
            {"shape": [1, 2], "other": [2, 1]},
            {"shape": [1, 32], "other": 1},
            {"shape": [1, 45], "other": 1},
            {"shape": [1, 512, 1, 1], "other": [1, 512, 1, 1]},
            {"shape": [1, 59], "other": 1},
            {"shape": [1, 5], "other": 1},
            {"shape": [1, 60], "other": 1},
            {"shape": [1, 64, 1, 1], "other": [1, 64, 1, 1]},
            # {"shape": [1, s0 + 1], "other": [s0 + 1, 1]},
            # {"shape": [1, s0], "other": 1 },
            # {"shape": [1, s10 + 1], "other": 1 },
            {"shape": [1066], "other": 0.5},
            {"shape": [1066], "other": [1066]},
            {"shape": [120, 1], "other": [120, 1]},
            {"shape": [120], "other": 0.5},
            {"shape": [128, 1], "other": [128, 1]},
            {"shape": [128], "other": 0.5},
            {"shape": [128], "other": [128]},
            {"shape": [16, 1, 49], "other": [16, 49, 1]},
            {"shape": [16, 1, 64], "other": [16, 64, 1]},
            {"shape": [160], "other": 0.5},
            {"shape": [160], "other": [160]},
            {"shape": [1], "other": 1},
            {"shape": [24, 1], "other": [1, 24]},
            {"shape": [240, 1], "other": [240, 1]},
            {"shape": [240], "other": 0.5},
            {"shape": [3, 320, 320], "other": [3, 1, 1]},
            {"shape": [3, 480, 640], "other": [3, 1, 1]},
            {"shape": [30, 1], "other": [30, 1]},
            {"shape": [300, 1], "other": [300, 1]},
            {"shape": [300], "other": 0.5},
            {"shape": [300], "other": [300]},
            {"shape": [30], "other": 0.5},
            {"shape": [320, 1], "other": [320, 1]},
            {"shape": [320], "other": 0.5},
            {"shape": [320], "other": [320]},
            {"shape": [3234, 1], "other": [3234, 1]},
            {"shape": [3234, 2], "other": [3234, 2]},
            {"shape": [3234], "other": [3234]},
            {"shape": [4, 1, 49], "other": [4, 49, 1]},
            {"shape": [4, 1, 64], "other": [4, 64, 1]},
            {"shape": [40], "other": 0.5},
            {"shape": [40], "other": [40]},
            {"shape": [480, 1], "other": [480, 1]},
            {"shape": [480], "other": 0.5},
            {"shape": [60, 1], "other": [60, 1]},
            {"shape": [60], "other": 0.5},
            {"shape": [64, 1, 49], "other": [64, 49, 1]},
            {"shape": [64, 1, 64], "other": [64, 64, 1]},
            {"shape": [640], "other": 0.5},
            {"shape": [640], "other": [640]},
            {"shape": [800, 1], "other": [800, 1]},
            {"shape": [800], "other": 0.5},
            {"shape": [80], "other": 0.5},
            {"shape": [80], "other": [80]},
            {"shape": [8732, 1], "other": [8732, 1]},
            {"shape": [8732, 2], "other": [8732, 2]},
            {"shape": [8732], "other": [8732]},
            {"shape": [96, 80], "other": [80]},
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
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    other = input_specs["other"]
    if isinstance(other, (int, float)):
        torch_other_tensor = torch.tensor(other, dtype=torch.float32)
    else:
        torch_other_tensor = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(other)

    torch_output_tensor = torch.sub(torch_input_tensor_a, torch_other_tensor)

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

    output_tensor = ttnn.subtract(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
