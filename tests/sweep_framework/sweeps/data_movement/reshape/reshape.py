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


@lru_cache(maxsize=5000)
def get_factors(i, s):
    factors = []
    for j in range(s, i + 1, s):
        if i % j == 0:
            factors.append(j)
    return factors


# @lru_cache(maxsize=5000)
def gen_reshape_shape(input_shape, step=1):
    volume = 1
    for x in input_shape:
        volume *= x

    shapes = []
    out_dims = len(input_shape)

    if out_dims == 4:
        for w in get_factors(volume, step):
            v = volume // w
            for h in get_factors(v, step):
                v2 = v // h
                for c in get_factors(v2, 1):
                    b = v2 // c
                    shapes.append({"reshape_dims": [b, c, h, w]})
    elif out_dims == 3:
        for h in get_factors(volume, step):
            v2 = volume // h
            for c in get_factors(v2, 1):
                b = v2 // c
                shapes.append({"reshape_dims": [b, c, h]})
    elif out_dims == 2:
        for c in get_factors(volume, 1):
            b = volume // c
            shapes.append({"reshape_dims": [b, c]})

    return random.choice(shapes)["reshape_dims"]


# Does not have memory_config parameter
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 6, 256, 256], [1, 1, 1, 1], 16)
        + gen_shapes([1, 1, 1], [6, 256, 256], [1, 1, 1], 16)
        + gen_shapes([1, 1], [256, 256], [1, 1], 16),
        "input_a_dtype": [ttnn.bfloat16, ttnn.float32],
        "input_a_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],  # ttnn.ROW_MAJOR_LAYOUT
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def align_to_32(x):
    if x % 32 == 0:
        return x

    return ((x // 32) + 1) * 32


def max_volume(rehape_shape):
    vol = align_to_32(rehape_shape[-1]) * align_to_32(rehape_shape[-2])

    if len(rehape_shape) >= 3:
        vol *= rehape_shape[-3]

    if len(rehape_shape) == 4:
        vol *= rehape_shape[-4]

    return vol


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    rehape_shape = gen_reshape_shape(input_shape)
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
