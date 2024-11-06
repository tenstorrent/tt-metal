# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes, gen_low_high_scalars
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

parameters = {
    "nightly": {
        "input_specs": [
            {"shape": [0, 1], "max": 4.135166556742356},
            {"shape": [0, 2], "min": 0, "max": 1066},
            {"shape": [0, 2], "min": 0, "max": 800},
            {"shape": [1, 1, 1, 42], "min": 0, "max": 82},
            {"shape": [1, 1, 32, 1], "min": 0, "max": 49},
            {"shape": [1066], "min": 0.0},
            {"shape": [1066], "max": 639},
            {"shape": [12, 1, 1], "max": 4.605170185988092},
            {"shape": [120], "min": 0.0},
            {"shape": [120], "max": 59},
            {"shape": [128], "min": 0.0},
            {"shape": [128], "max": 127},
            {"shape": [128], "max": 15},
            {"shape": [128], "max": 31},
            {"shape": [128], "max": 63},
            {"shape": [16, 1, 1], "max": 4.605170185988092},
            {"shape": [160], "min": 0.0},
            {"shape": [160], "max": 79},
            {"shape": [24, 1, 1], "max": 4.605170185988092},
            {"shape": [240], "min": 0.0},
            {"shape": [240], "max": 119},
            {"shape": [3, 1, 1], "max": 4.605170185988092},
            {"shape": [300], "min": 0.0},
            {"shape": [300], "max": 479},
            {"shape": [300], "max": 639},
            {"shape": [30], "min": 0.0},
            {"shape": [30], "max": 14},
            {"shape": [32, 1, 1], "max": 4.605170185988092},
            {"shape": [320], "min": 0.0},
            {"shape": [320], "max": 159},
            {"shape": [320], "max": 319},
            {"shape": [320], "max": 479},
            {"shape": [320], "max": 639},
            {"shape": [3234, 1], "max": 4.135166556742356},
            {"shape": [3234, 2], "min": 0, "max": 320},
            {"shape": [4, 1, 1], "max": 4.605170185988092},
            {"shape": [4, 2], "min": 0, "max": 1},
            {"shape": [40], "min": 0.0},
            {"shape": [40], "max": 19},
            {"shape": [480], "min": 0.0},
            {"shape": [480], "max": 239},
            {"shape": [6, 1, 1], "max": 4.605170185988092},
            {"shape": [6, 2], "min": 0, "max": 1},
            {"shape": [60], "min": 0.0},
            {"shape": [60], "max": 29},
            {"shape": [640], "min": 0.0},
            {"shape": [640], "max": 319},
            {"shape": [8, 1, 1], "max": 4.605170185988092},
            {"shape": [800], "min": 0.0},
            {"shape": [800], "max": 479},
            {"shape": [80], "min": 0.0},
            {"shape": [80], "max": 39},
            {"shape": [8732, 1], "max": 4.135166556742356},
            {"shape": [8732, 2], "min": 0, "max": 300},
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

    min_val = input_specs.get("min", None)
    max_val = input_specs.get("max", None)

    golden_function = ttnn.get_golden_function(ttnn.clamp)
    torch_output_tensor = golden_function(torch_input_tensor_a, min_val, max_val)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.clamp(input_tensor_a, min=min_val, max=max_val, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
