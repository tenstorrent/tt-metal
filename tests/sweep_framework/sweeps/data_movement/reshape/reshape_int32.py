# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
from itertools import combinations

import torch
import random
import ttnn
from functools import lru_cache
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 360
random.seed(0)


# Does not have memory_config parameter
parameters = {
    "nightly": {
        "shapes": [
            ([1, 1, 16, 16], [1, 1, 8, 32]),  # tile padded to tile padded small
            ([1, 64, 1, 16], [1, 8, 32, 32]),  # tile padded to tile aligned from outer dim small
            ([1, 16, 1, 32], [1, 1, 32, 16]),  # tile padded to outer dim small
            ([1, 1, 32, 32], [1, 1, 16, 64]),  # tile aligned to tile padded small
            ([1, 1, 32, 32], [1, 32, 1, 32]),  # tile aligned to tile padded outer dim small
            ([1, 128, 128, 2048], [128, 1, 2048, 128]),  # tile aligned outer dim to outer dim
            ([1, 128, 2048, 120], [1, 120, 2048, 128]),  # tile padded to tile aligned from outer dim
            ([1, 2048, 128, 128], [1, 128, 128, 2048]),  # tile aligned inner dim to outer dim
        ],
        "input_a_dtype": [ttnn.int32],
        "input_a_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],  # ttnn.ROW_MAJOR_LAYOUT
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def run(
    shapes,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)
    input_shape = shapes[0]
    output_shape = shapes[1]

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    rehape_shape = gen_reshape_shape(tuple(input_shape))
    num_tries = 50
    i = 0

    # If we store result to L1 required volume should not be larger than 1Mb
    while input_a_memory_config == ttnn.L1_MEMORY_CONFIG and max_volume(rehape_shape) > 1024 * 1024 and i < num_tries:
        rehape_shape = gen_reshape_shape(input_shape)
        i += 1

    torch_output_tensor = torch.reshape(torch_input_tensor_a, rehape_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    # print(f"input_shape {input_shape} rehape_shape {rehape_shape} input_a_dtype {input_a_dtype}")

    start_time = start_measuring_time()
    result = ttnn.reshape(input_tensor_a, shape=rehape_shape)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    # print(pcc)
    return [pcc, e2e_perf]
