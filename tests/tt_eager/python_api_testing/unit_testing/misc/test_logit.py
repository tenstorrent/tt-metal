# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_eltwise_logit_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, eps, data_seed, device):
    torch.manual_seed(data_seed)
    x = gen_rand(size=input_shape, low=0, high=0.99)

    # compute ref value
    ref_value = pytorch_ops.logit(x=x, eps=eps)

    tt_result = tt_lib_ops.eltwise_logit(
        x=x,
        eps=eps,
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
        (9, 5, 160, 96),
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        0.64453125,
        7340822,
    ),
    (
        (12, 9, 64, 384),
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        0.140625,
        12484268,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, eps, data_seed",
    (test_sweep_args),
)
def test_eltwise_logit(input_shape, dtype, dlayout, in_mem_config, out_mem_config, eps, data_seed, device):
    random.seed(0)
    run_eltwise_logit_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, eps, data_seed, device)
