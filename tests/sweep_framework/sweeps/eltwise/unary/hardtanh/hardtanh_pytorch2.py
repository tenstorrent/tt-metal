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
            {"shape": [1, 1024, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1152, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1152, 8, 8], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 116, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1248, 9, 9], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 128, 1, 1], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 128, 2, 2], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 128, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 128, 3, 3], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 128, 5, 5], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 128, 56, 56], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1280, 10, 10], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1280, 12, 12], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1280, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1280, 8, 8], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1280, 9, 9], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 134, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1392, 10, 10], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 14, 56, 56], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 150, 150], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 190, 190], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 30, 30], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 33, 33], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 56, 56], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 60, 60], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 65, 65], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 75, 75], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 144, 95, 95], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 16, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 160, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 1632, 12, 12], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 168, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 192, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 192, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 192, 38, 38], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 192, 48, 48], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 192, 75, 75], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 192, 95, 95], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 196, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 20, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 24, 56, 56], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 240, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 240, 15, 15], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 240, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 240, 30, 30], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 256, 10, 10], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 256, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 256, 2, 2], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 256, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 256, 3, 3], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 256, 5, 5], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 272, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 28, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 288, 17, 17], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 288, 19, 19], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 288, 33, 33], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 288, 38, 38], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 32, 112, 112], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 32, 120, 120], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 32, 130, 130], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 32, 150, 150], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 32, 190, 190], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 320, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 334, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 336, 24, 24], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 336, 48, 48], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 34, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 384, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 40, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 40, 56, 56], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 46, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 462, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 480, 10, 10], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 480, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 480, 15, 15], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 512, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 512, 5, 5], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 512, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 528, 17, 17], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 576, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 576, 19, 19], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 576, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 58, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 64, 1, 1], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 64, 112, 112], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 64, 2, 2], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 64, 56, 56], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 640, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 672, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 672, 15, 15], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 672, 20, 20], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 672, 24, 24], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 672, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 672, 8, 8], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 68, 14, 14], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 68, 56, 56], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 720, 17, 17], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 720, 9, 9], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 78, 28, 28], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 816, 10, 10], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 816, 19, 19], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 96, 112, 112], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 96, 120, 120], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 96, 130, 130], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 96, 56, 56], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 96, 60, 60], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 96, 65, 65], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 960, 12, 12], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 960, 24, 24], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 960, 7, 7], "min_val": 0.0, "max_val": 6.0},
            {"shape": [1, 98, 28, 28], "min_val": 0.0, "max_val": 6.0},
        ],
        "input_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def run(
    input_specs,
    input_dtype,
    input_layout,
    input_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((input_specs["shape"]))

    min_val = input_specs.get("min_val")
    max_val = input_specs.get("max_val")

    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
    torch_output_tensor = golden_function(torch_input_tensor_a, min=min_val, max=max_val)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.hardtanh(input_tensor_a, min=min_val, max=max_val, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
