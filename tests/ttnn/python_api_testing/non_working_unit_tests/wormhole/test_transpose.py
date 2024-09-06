# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_transpose_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dim0, dim1, device):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        ref_value = torch.transpose(x, dim0, dim1)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        tt_result = ttnn.transpose(x, dim0, dim1)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(4, 4, 128, 128)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        0,
        1,
    ),
    (
        [(32, 32, 32)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        0,
        1,
    ),
    (
        [(32, 32)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        0,
        1,
    ),
    # (
    #     [(64, 64, 64)],
    #     [ttnn.bfloat16],
    #     [ttnn.TILE_LAYOUT],
    #     [ttnn.DRAM_MEMORY_CONFIG],
    #     (ttnn.DRAM_MEMORY_CONFIG),
    #     4171614,
    #     0,
    #     2,
    # ),
    (
        [(32, 32, 32)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        1,
        2,
    ),
    (
        [(4, 4, 128, 128)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        0,
        1,
    ),
    (
        [(2, 2, 128, 128)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        0,
        1,
    ),
    (
        [(2, 2, 126, 126)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        0,
        2,
    ),
    (
        [(2, 2, 126, 126)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        1,
        2,
    ),
    (
        [(2, 2, 126, 126)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        1,
        3,
    ),
    (
        [(2, 2, 128, 128)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        0,
        2,
    ),
    (
        [(2, 2, 128, 128)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        1,
        2,
    ),
    (
        [(4, 4, 128, 128)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        1,
        3,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dim0, dim1",
    (test_sweep_args),
)
def test_tranpose(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dim0, dim1, device):
    run_transpose_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dim0, dim1, device)
