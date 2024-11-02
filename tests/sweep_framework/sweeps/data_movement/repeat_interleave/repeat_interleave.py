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
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 6, 256, 256], [1, 1, 1, 1], 8)
        + gen_shapes([1, 1, 1], [6, 256, 256], [1, 1, 1], 8)
        + gen_shapes([1, 1], [256, 256], [1, 1], 8),
        "repeats": [1, 2, 4, 8],
        "dim": [0, 1, 2, 3],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
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


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_shape = test_vector["input_shape"]

    if test_vector["dim"] >= len(input_shape):
        return True, "dim must be < len(input_shape)"

    if (
        test_vector["input_a_memory_config"] == ttnn.L1_MEMORY_CONFIG
        or test_vector["output_memory_config"] == ttnn.L1_MEMORY_CONFIG
    ):
        if max_volume(input_shape) * test_vector["repeats"] > 1024 * 1024:
            return True, "Too large output tensor size for L1 memory config"

    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b/bfloat4_b requires TILE_LAYOUT!"

    return False, None


def run(
    input_shape,
    repeats,
    dim,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    # Fix shape for row mayor
    if input_a_layout == ttnn.ROW_MAJOR_LAYOUT and input_shape[-1] % 2 == 1:
        input_shape[-1] += 1

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    # print(f"input_shape {input_shape} repeats {repeats} dim {dim} input_a_dtype {input_a_dtype} input_a_layout {input_a_layout}")

    golden_function = ttnn.get_golden_function(ttnn.repeat_interleave)
    torch_output_tensor = golden_function(torch_input_tensor_a, repeats=repeats, dim=dim)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.repeat_interleave(input_tensor_a, repeats=repeats, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    # print(pcc)
    return [pcc, e2e_perf]


# # Run sweeps locally
# from tests.sweep_framework.framework.permutations import *

# start_time = start_measuring_time()
# for suite in parameters.keys():
#     device_id = 0
#     device = ttnn.open_device(device_id=device_id)
#     suite_vectors = list(permutations(parameters[suite]))
#     print(len(suite_vectors))
#     for vector in suite_vectors:
#         invalidate_res = invalidate_vector(vector)
#         if invalidate_res[0]:
#             print(f"Invalidated: {invalidate_res[1]}")
#             continue
#         try:
#             passed, _ = run(**vector, device=device)
#             if passed[0] != True:
#                 print(passed)
#         except Exception as e:
#             print(e)

#     ttnn.close_device(device)

# e2e_perf = stop_measuring_time(start_time)
# print(f"time {e2e_perf / 1000000000}s")
