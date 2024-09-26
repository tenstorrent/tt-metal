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
    "hardswish_1": {
        "input_shape": [
            [1, 1024],
            [1, 120, 14, 14],
            [1, 1280],
            [1, 144, 14, 14],
            [1, 16, 112, 112],
            [1, 16, 160, 160],
            [1, 184, 14, 14],
            [1, 184, 20, 20],
            [1, 200, 14, 14],
            [1, 200, 20, 20],
            [1, 240, 14, 14],
            [1, 240, 20, 20],
            [1, 240, 28, 28],
            [1, 240, 40, 40],
            [1, 288, 14, 14],
            [1, 288, 7, 7],
            [1, 480, 10, 10],
            [1, 480, 14, 14],
            [1, 480, 20, 20],
            [1, 576, 7, 7],
            [1, 672, 10, 10],
            [1, 672, 14, 14],
            [1, 672, 20, 20],
            [1, 672, 7, 7],
            [1, 96, 14, 14],
            [1, 96, 28, 28],
            [1, 960, 7, 7],
        ],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def run(
    input_shape,
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
    )(input_shape)

    torch_output_tensor = torch.nn.functional.hardswish(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.hardswish(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
