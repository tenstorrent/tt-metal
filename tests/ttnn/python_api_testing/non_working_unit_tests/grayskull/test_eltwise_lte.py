# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_eltwise_lte_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    scalar,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    if scalar == 0:
        y = torch.Tensor(size=input_shape[1]).uniform_(-100, 100).to(torch.bfloat16)
    else:
        y = scalar

    try:
        # get ref result
        ref_value = x <= y

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        if scalar == 0:
            y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[1], in_mem_config[1], dtype[1])

        tt_result = ttnn.lte(x, y)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(128, 192), (128, 192)],
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        66.5,
        3930374,
    ),
    (
        [(6, 224, 32), (6, 224, 32)],
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.L1_MEMORY_CONFIG),
        77.5,
        6689270,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed",
    (test_sweep_args),
)
def test_eltwise_lte(input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, device):
    run_eltwise_lte_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, device)
