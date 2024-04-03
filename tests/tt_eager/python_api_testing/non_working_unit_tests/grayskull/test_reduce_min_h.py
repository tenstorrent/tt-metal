# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import reduce_min_h as tt_reduce_min_h
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_reduce_min_h_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)

    # compute ref value
    x_ref = x.detach().clone()
    ref_value = pytorch_ops.reduce_min(x_ref, dims=(-2,))

    tt_result = tt_reduce_min_h(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    # ",16530771,(),completed,"Max ATOL Delta: 29.0, Max RTOL Delta: 0.453125, PCC: 0.8566493132465313, PCC check failed",fail
    (
        (4, 10, 96, 128),
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        16530771,
    ),
    # (
    #     (3, 3, 168, 134),
    #     ttl.tensor.DataType.BFLOAT16,
    #     ttl.tensor.Layout.ROW_MAJOR,
    #     "SYSTEM_MEMORY",
    #     ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    #     8057439,
    # ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_reduce_min_h_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_reduce_min_h_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
