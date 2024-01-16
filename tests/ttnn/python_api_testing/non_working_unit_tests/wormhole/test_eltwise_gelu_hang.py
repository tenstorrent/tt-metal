# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, set_slow_dispatch_mode
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_eltwise_gelu_tests(
    input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.nn.functional.gelu(x)

        x = ttnn_ops.torch_to_ttnn(x, device, dlayout[0], in_mem_config, dtype[0])

        tt_result = ttnn.gelu(x)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)
        logger.info(f"Op run for input dimension {input_shape[0]} finished")

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
        [(150, 72)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        17799073,
        "",
    ),
    (
        [(3, 201, 228)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        3121221,
        "",
    ),
    (
        [(6, 6, 230, 138)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        10286194,
        "",
    ),
]


def test_eltwise_gelu(device):
    for i in range(30000):
        for input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode in test_sweep_args:
            run_eltwise_gelu_tests(
                input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode, device
            )
