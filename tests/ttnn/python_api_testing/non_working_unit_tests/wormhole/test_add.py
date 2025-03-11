# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_add_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    xx = torch.Tensor(size=input_shape[0]).uniform_(0, 10).to(torch.int32)
    yy = torch.Tensor(size=input_shape[1]).uniform_(0, 10).to(torch.int32)

    try:
        # get ref result
        ref_value = torch.add(xx, yy)

        x = ttnn_ops.setup_ttnn_tensor(xx, device, dlayout[0], in_mem_config, dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(yy, device, dlayout[1], in_mem_config, dtype[1])

        tt_result = ttnn.add(x, y)  # ttnn.experimental.add(x, y)
        tt_result = ttnn.to_torch(tt_result)
        logger.info(f"Add run {input_shape[0]} finished")

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    print(f"xx {xx}")
    print(f"yy {yy}")

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(32, 1), (32, 1)],
        [ttnn.uint32, ttnn.uint32],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        17799073,
    ),
]


def test_matmul(device):
    for input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed in test_sweep_args:
        run_add_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device)
