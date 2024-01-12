# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, set_slow_dispatch_mode
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_matmul_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode, device):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[1]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.matmul(x, y)

        x = ttnn_ops.torch_to_ttnn(x, device, dlayout[0], in_mem_config, dtype[0])
        y = ttnn_ops.torch_to_ttnn(y, device, dlayout[1], in_mem_config, dtype[1])

        tt_result = ttnn.matmul(x, y)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        set_slow_dispatch_mode(prev_dispatch_mode)
        raise e

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(3, 5, 29, 62), (3, 5, 62, 31)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        "1",
    ),
    (
        [(5, 3, 29, 61), (1, 1, 61, 30)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        "1",
    ),
    (
        [(5, 3, 29, 62), (62, 31)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        "1",
    ),
    (
        [(62,), (62, 3)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        "1",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_matmul(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode, device):
    run_matmul_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode, device)
