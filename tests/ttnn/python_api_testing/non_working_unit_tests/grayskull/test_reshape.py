# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_reshape_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, reshape_dims, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.reshape(x, reshape_dims)
        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config, dtype[0])

        tt_result = ttnn.reshape(x, reshape_dims)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        (224, 128),
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        (None),
        (ttnn.DRAM_MEMORY_CONFIG),
        (448, 64),
        14748599,
    ),
    (
        (224, 128),
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        (None),
        (ttnn.DRAM_MEMORY_CONFIG),
        (448, 64),
        14748599,
    ),
    (
        (12, 32, 160),
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        (None),
        (ttnn.DRAM_MEMORY_CONFIG),
        (1, 192, 320),
        14748599,
    ),
    (
        (4, 12, 64, 224),
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        (None),
        (ttnn.DRAM_MEMORY_CONFIG),
        (6, 8, 224, 64),
        14748599,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, reshape_dims, data_seed",
    (test_sweep_args),
)
def test_reshape(input_shape, dtype, dlayout, in_mem_config, out_mem_config, reshape_dims, data_seed, device):
    run_reshape_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, reshape_dims, data_seed, device)
