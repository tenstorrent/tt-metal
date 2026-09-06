# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# logaddexp2 is finite for any finite pair, but the nightly sweep in logaddexp2.py draws
# from [-60, 100] and never reaches the 2**x overflow boundary at |x| > 127, where the
# composed log2(2**a + 2**b) form returned +/-inf. This suite draws from [-1000, 1000]
# so that most positions sit past that boundary in one direction or the other.
#
# bfloat8_b is left out on purpose: its shared-exponent quantization collapses the
# |a - b| < 20 band where the log2(1 + 2**-|a - b|) correction matters, and the fused
# SFPU kernel is routed for float32 and bfloat16 operands only.
parameters = {
    "overflow": {
        "input_shape": gen_shapes([1, 1, 32, 32], [6, 12, 256, 256], [1, 1, 32, 32], 16)
        + gen_shapes([1, 32, 32], [12, 256, 256], [1, 32, 32], 16)
        + gen_shapes([32, 32], [256, 256], [32, 32], 16),
        "input_a_dtype": [ttnn.bfloat16, ttnn.float32],
        "input_b_dtype": [ttnn.bfloat16, ttnn.float32],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_dtype"] != test_vector["input_b_dtype"]:
        return True, "The fused logaddexp2 kernel is routed for matching operand dtypes only"
    return False, None


def run(
    input_shape,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1000, high=1000, dtype=torch.float32), input_a_dtype
    )(input_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-1000, high=1000, dtype=torch.float32), input_b_dtype
    )(input_shape)

    golden_function = ttnn.get_golden_function(ttnn.logaddexp2)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )
    start_time = start_measuring_time()
    result = ttnn.logaddexp2(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
