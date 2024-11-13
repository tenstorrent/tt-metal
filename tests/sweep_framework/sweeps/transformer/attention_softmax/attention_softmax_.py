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
        "input_shape": gen_shapes([1, 1, 1, 8], [6, 1, 256, 256], [1, 1, 1, 8], 4)
        + gen_shapes([1, 1, 8], [6, 256, 256], [1, 1, 8], 4)
        + gen_shapes([1, 8], [256, 256], [1, 8], 4),
        "num_heads": [1, 2, 4, 8],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "mask_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "mask_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "mask_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # input_shape = test_vector["input_shape"]

    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b/bfloat4_b requires TILE_LAYOUT!"

    return False, None


def run(
    input_shape,
    num_heads,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    mask_dtype,
    mask_layout,
    mask_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    hidden_size = input_shape[-1]
    head_size = hidden_size // num_heads

    # Fix shape for row mayor
    if input_a_layout == ttnn.ROW_MAJOR_LAYOUT and input_shape[-1] % 2 == 1:
        input_shape[-1] += 1

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    torch_mask_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), mask_dtype
    )(input_shape)
    torch_mask_tensor = (torch_mask_tensor > 0).to(torch.float32)

    # print(f"input_shape {input_shape} input_a_dtype {input_a_dtype} input_a_layout {input_a_layout}")

    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax_)
    tmp_input = torch.clone(torch_input_tensor_a)
    torch_output_tensor = golden_function(tmp_input, head_size=head_size, attention_mask=torch_mask_tensor)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    mask_tensor = ttnn.from_torch(
        torch_mask_tensor,
        dtype=mask_dtype,
        layout=mask_layout,
        device=device,
        memory_config=mask_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.transformer.attention_softmax_(input_tensor_a, head_size=head_size, attention_mask=mask_tensor)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    # print(pcc)
    return [pcc, e2e_perf]
