# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_bcast_eq_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, scalar, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)

    if dtype[0] == ttnn.bfloat8_b:
        x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None)
        x = ttnn.to_torch(x)

    try:
        ref_value = x == scalar

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.eq(x, scalar, memory_config=output_mem_config)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    success, pcc_value = comp_pcc(ref_value, tt_result)
    assert success, f"{pcc_value}"


test_sweep_args = [
    (
        [(224, 128)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        8687804,
        -8.625,
    ),
    (
        [(6, 160, 64)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        9456908,
        -83.5,
    ),
    (
        [(2, 12, 32, 128)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        8721464,
        77.5,
    ),
    (
        [(1, 11, 64, 224)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        8411671,
        -62.5,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar",
    (test_sweep_args),
)
def test_bcast_eq(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device):
    run_bcast_eq_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device)
