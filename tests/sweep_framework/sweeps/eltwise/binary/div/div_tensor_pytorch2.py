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


parameters = {
    "nightly": {
        "input_specs": [
            {"shape": [0, 1], "other": 1.0},
            {"shape": [1, 1, 16384, 256], "other": 5.656854249492381},
            {"shape": [1, 1, 19200, 300], "other": 8.0},
            {"shape": [1, 1, 256], "other": 0.5},
            {"shape": [1, 1024, 640], "other": 1.0},
            {"shape": [1, 12, 10, 10], "other": 8.0},
            {"shape": [1, 12, 12, 12], "other": 8.0},
            {"shape": [1, 12, 14, 14], "other": 8.0},
            {"shape": [1, 12, 16, 64], "other": 8.0},
            {"shape": [1, 12, 197, 197], "other": 8.0},
            {"shape": [1, 12, 201, 201], "other": 8.0},
            {"shape": [1, 12, 25, 25], "other": 8.0},
            {"shape": [1, 12, 7, 7], "other": []},
            {"shape": [1, 12, 9, 9], "other": 8.0},
            {"shape": [1, 128, 1536], "other": 3},
            {"shape": [1, 1280, 16, 16], "other": 1.0},
            {"shape": [1, 1280, 8, 8], "other": 1},
            {"shape": [1, 1280, 8, 8], "other": 1.0},
            {"shape": [1, 16, 1, 6], "other": []},
            # {"shape": [1, 16, 1, s10 + 1], "other": []},
            {"shape": [1, 16, 197, 197], "other": 8.0},
            {"shape": [1, 16, 256, 256], "other": 8.0},
            {"shape": [1, 16, 5, 5], "other": []},
            {"shape": [1, 16, 9, 9], "other": 11.313708498984761},
            {"shape": [1, 16, 9, 9], "other": 8.0},
            {"shape": [1, 1], "other": 16},
            {"shape": [1, 1], "other": 2.0794415416798357},
            {"shape": [1, 2, 4096, 256], "other": 5.656854249492381},
            {"shape": [1, 2, 4800, 300], "other": 8.0},
            {"shape": [1, 23, 40, 1], "other": [128]},
            {"shape": [1, 23, 40], "other": [1, 1, 40]},
            {"shape": [1, 23, 40], "other": [1, 23, 1]},
            {"shape": [1, 24, 64, 32], "other": [1, 24, 64, 32]},
            {"shape": [1, 256, 1280], "other": 1.0},
            {"shape": [1, 256, 384], "other": 3},
            {"shape": [1, 3, 1445, 1445], "other": 8.0},
            {"shape": [1, 32, 24576], "other": 3},
            {"shape": [1, 32, 64, 32], "other": [1, 32, 64, 32]},
            {"shape": [1, 320, 64, 64], "other": 1.0},
            {"shape": [1, 4096, 320], "other": 1.0},
            {"shape": [1, 5, 1024, 256], "other": 5.656854249492381},
            {"shape": [1, 5, 1200, 300], "other": 8.0},
            {"shape": [1, 50257], "other": 0.9},
            {"shape": [1, 512, 38, 38], "other": [1, 512, 38, 38]},
            {"shape": [1, 512], "other": [1, 1]},
            {"shape": [1, 512], "other": [1, 512]},
            {"shape": [1, 64, 1280], "other": 1.0},
            {"shape": [1, 64, 6144], "other": 3},
            {"shape": [1, 64, 9, 9], "other": 8.0},
            {"shape": [1, 640, 32, 32], "other": 1.0},
            {"shape": [1, 8, 2048, 256], "other": 5.656854249492381},
            {"shape": [1, 8, 256, 2048], "other": 5.656854249492381},
            {"shape": [1, 8, 256, 256], "other": 5.656854249492381},
            {"shape": [1, 8, 300, 300], "other": 8.0},
            # {"shape": [1, s0, 256], "other": 0.5 },
            {"shape": [10, 10], "other": 2.772588722239781},
            {"shape": [10, 10], "other": 8},
            {"shape": [10], "other": 10},
            {"shape": [10], "other": 9.375},
            {"shape": [128], "other": 128},
            {"shape": [15, 15], "other": 2.772588722239781},
            {"shape": [15, 15], "other": 8},
            {"shape": [16, 6, 64, 32], "other": [16, 6, 64, 32]},
            {"shape": [16, 8, 64, 32], "other": [16, 8, 64, 32]},
            {"shape": [160], "other": 160},
            {"shape": [17, 17], "other": 16},
            {"shape": [17, 17], "other": 2.0794415416798357},
            {"shape": [19], "other": 18.75},
            {"shape": [1], "other": 1},
            {"shape": [1], "other": 1.0},
            {"shape": [2, 2], "other": 16},
            {"shape": [2, 2], "other": 2.0794415416798357},
            {"shape": [2, 512], "other": [2, 1]},
            {"shape": [20], "other": 20},
            {"shape": [2], "other": 2},
            {"shape": [3, 320, 320], "other": [3, 1, 1]},
            {"shape": [3, 480, 640], "other": [3, 1, 1]},
            {"shape": [3234, 1], "other": 10.0},
            {"shape": [3234, 1], "other": 5.0},
            {"shape": [38], "other": 37.5},
            {"shape": [3], "other": 3},
            {"shape": [3], "other": 3.0},
            {"shape": [4, 12, 64, 32], "other": [4, 12, 64, 32]},
            {"shape": [4, 16, 64, 32], "other": [4, 16, 64, 32]},
            {"shape": [5], "other": 4.6875},
            {"shape": [5], "other": 5},
            {"shape": [64, 3, 64, 32], "other": [64, 3, 64, 32]},
            {"shape": [64, 4, 64, 32], "other": [64, 4, 64, 32]},
            {"shape": [8, 100, 32], "other": 5.656854249492381},
            {"shape": [8, 920, 32], "other": 5.656854249492381},
            {"shape": [8732, 1], "other": 10.0},
            {"shape": [8732, 1], "other": 5.0},
            {"shape": [96, 80], "other": [80]},
            {"shape": [], "other": []},
            # {"shape": [s0 + 1, s0 + 1], "other": 16 },
            # {"shape": [s0 + 1, s0 + 1], "other": 2.0794415416798357  },
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

    torch_output_tensor = torch.div(torch_input_tensor_a, torch_other_tensor)

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

    output_tensor = ttnn.div(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
