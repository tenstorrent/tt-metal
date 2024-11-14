# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from functools import partial
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.utility_functions import torch_random


def run_backward_div_tests(
    input_shape,
    approx,
    dtype,
    dlayout,
    in_mem_cfg,
    out_mem_cfg,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    # grad tensor
    x = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype[0])(input_shape[0])
    # input tensor
    y = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype[1])(input_shape[0])

    y.requires_grad = True

    scalar = torch.tensor(1, dtype=torch.bfloat16).uniform_(-100, 100).item()
    print("scalar", scalar)
    try:
        # get ref result
        golden_function = ttnn.get_golden_function(ttnn.bias_gelu_bw)
        ref_value = golden_function(x, y, scalar, value=approx)[0]

        tt_x = ttnn.from_torch(x, dtype=dtype[0], layout=dlayout[0], device=device, memory_config=in_mem_cfg[0])
        tt_y = ttnn.from_torch(y, dtype=dtype[1], layout=dlayout[0], device=device, memory_config=in_mem_cfg[1])

        tt_result = ttnn.bias_gelu_bw(tt_x, tt_y, scalar, approximate=approx, memory_config=out_mem_cfg)[0]
        tt_result = ttnn.to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.999)


test_sweep_args = [
    (
        [(6, 10, 128, 224)],  # AssertionError: 0.99706924575737 , scalar -99.0
        "tanh",
        [ttnn.bfloat8_b, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        14469376,
    ),
    (
        [(4, 2, 96, 192)],  # AssertionError: 0.9744508807102572, scalar -100.0
        "tanh",
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        4378657,
    ),
    (
        [(5, 10, 224, 32)],  # AssertionError: 0.9982306869898846,  scalar -98.5
        "tanh",
        [ttnn.bfloat8_b, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        678741,
    ),
    (
        [(97, 129)],  # Pass,  0.9990033308812074,  scalar -97.5
        "tanh",
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        7580522,
    ),
]


@pytest.mark.parametrize(
    "input_shape, approx, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_backward_div(input_shape, approx, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_backward_div_tests(input_shape, approx, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
