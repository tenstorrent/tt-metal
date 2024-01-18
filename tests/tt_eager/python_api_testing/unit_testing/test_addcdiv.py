# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import numpy as np
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc_skip_inf
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import addcdiv as pt_eltwise_addcdiv
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_addcdiv as tt_eltwise_addcdiv


def run_eltwise_addcdiv_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100)
    z = torch.Tensor(size=input_shape).uniform_(-100, 100)

    # get referent value
    ref_value = pt_eltwise_addcdiv(x, y, z, scalar=scalar)

    # calculate tt output
    logger.info("Running Eltwise addcdiv test")
    tt_result = tt_eltwise_addcdiv(
        x=x,
        y=y,
        z=z,
        scalar=scalar,
        device=device,
        dtype=[dtype, dtype, dtype],
        layout=[dlayout, dlayout, dlayout],
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )
    logger.info("Done")

    # compare tt and golden outputs
    success, pcc_value = comp_pcc_skip_inf(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (4, 6, 160, 224),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        0.78125,
        3514701,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed",
    (test_sweep_args),
)
def test_eltwise_addcdiv_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, device):
    random.seed(0)
    run_eltwise_addcdiv_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, device)
