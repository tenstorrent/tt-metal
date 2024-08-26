# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from itertools import product

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def run_backward_i0_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-10, 10)
    y = torch.Tensor(size=input_shape[0]).uniform_(-10, 10)

    try:
        # get ref result
        ref_value = pytorch_ops.i0_bw(x, y)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config[1], dtype[1])

        tt_result = ttnn.i0_bw(x, y, memory_config=output_mem_config)[0]
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(224, 128)],
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        4689090,
    ),
    (
        [(8, 224, 160)],
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        18411293,
    ),
    (
        [(6, 11, 64, 224)],
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        18810621,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_backward_i0(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_backward_i0_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
