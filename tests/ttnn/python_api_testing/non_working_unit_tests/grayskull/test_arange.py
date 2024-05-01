# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc, divup
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_arange_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    start,
    end,
    step,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.arange(start, end, step)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.arange(start, end, step, device)

        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)
        if divup((end - start), step) % 2 != 0:
            tt_result = tt_result.view(-1)[:-1]

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(1, 64, 128)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        13587334,
        -52,
        -13,
        9,
    ),
    (
        [(1, 32, 128)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        14008474,
        12,
        50,
        3,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, start, end, step",
    (test_sweep_args),
)
def test_arange(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, start, end, step, device):
    run_arange_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, start, end, step, device)
