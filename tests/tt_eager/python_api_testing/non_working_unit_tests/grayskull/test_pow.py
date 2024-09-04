# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_pow as tt_eltwise_pow


def run_eltwise_pow_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, exponent, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = pytorch_ops.power(x_ref, exponent=exponent)

    tt_result = tt_eltwise_pow(
        x=x,
        exponent=exponent,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (10, 7, 288, 96),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        5.125,
        4689090,
    ),
    (
        (8, 3, 320, 416),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        4.03125,
        10638326,
    ),
    (
        (2, 12, 428, 504),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        "SYSTEM_MEMORY",
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        3.046875,
        726000,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, exponent, data_seed",
    (test_sweep_args),
)
def test_eltwise_pow(input_shape, dtype, dlayout, in_mem_config, out_mem_config, exponent, data_seed, device):
    run_eltwise_pow_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, exponent, data_seed, device)
