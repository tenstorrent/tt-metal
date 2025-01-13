# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_full_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, fill_value, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.full(input_shape[0], fill_value)

        tt_result = ttnn_ops.full(
            x,
            scalar=fill_value,
            device=device,
            dtype=dtype,
            layout=dlayout,
            input_mem_config=in_mem_config,
            output_mem_config=output_mem_config,
        )

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


def run_ones_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.ones(input_shape[0])

        tt_result = ttnn_ops.ones(
            x,
            device=device,
            dtype=dtype,
            layout=dlayout,
            input_mem_config=in_mem_config,
            output_mem_config=output_mem_config,
        )

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(4, 7, 32, 160)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
        5,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, fill_value",
    (test_sweep_args),
)
def test_eltwise_full(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, fill_value, device):
    run_full_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, fill_value, device)


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, fill_value",
    (test_sweep_args),
)
def test_eltwise_ones(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, fill_value, device):
    run_ones_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device)
