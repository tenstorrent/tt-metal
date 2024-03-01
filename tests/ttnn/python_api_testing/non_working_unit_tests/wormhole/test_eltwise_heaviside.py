# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_inf
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_eltwise_heaviside_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    scalar = 96.0

    # compute ref value
    ref_value = pytorch_ops.heaviside(x, scalar=scalar)

    tt_result = ttnn_ops.eltwise_heaviside(
        x=x,
        scalar=scalar,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        (4, 7, 32, 96),
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        16305027,
    ),
    (
        (6, 2, 224, 192),
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        3624344,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_heaviside(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    run_eltwise_heaviside_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
