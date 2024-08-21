# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_matmul_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[1]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.matmul(x, y)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config, dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[1], in_mem_config, dtype[1])

        tt_result = ttnn.matmul(x, y)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)
        logger.info(f"Matmul run {input_shape[0]} x {input_shape[1]} finished")

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(150, 72), (72, 48)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        17799073,
    ),
    (
        [(3, 201, 228), (228, 72)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        3121221,
    ),
    (
        [(6, 6, 230, 138), (138, 234)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        10286194,
    ),
]


def test_matmul(device):
    for i in range(30000):
        for input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed in test_sweep_args:
            run_matmul_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device)
