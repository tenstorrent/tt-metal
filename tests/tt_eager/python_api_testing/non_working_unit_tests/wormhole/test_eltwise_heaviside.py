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

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import heaviside as pt_heaviside
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_heaviside as tt_heaviside


def run_eltwise_heaviside_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
        x_ref = x.detach().clone()

        # get referent value
        ref_value = pt_heaviside(x_ref, scalar=scalar)

        # calculate tt output
        logger.info("Running Eltwise heaviside test")
        tt_result = tt_heaviside(
            x=x,
            scalar=scalar,
            device=device,
            device_id=0,
            dtype=[dtype],
            layout=[dlayout],
            input_mem_config=[in_mem_config],
            output_mem_config=out_mem_config
        )
        logger.info("Done")

        # compare tt and golden outputs
        success, pcc_value = comp_pcc(ref_value, tt_result)
        logger.debug(pcc_value)

        assert success


test_sweep_args=[
    ((1, 6, 256, 160),
     ttl.tensor.DataType.BFLOAT16,
     ttl.tensor.Layout.TILE,
     ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
     ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), 86.5, 16417740),
    ((4, 11, 106, 232),
     ttl.tensor.DataType.BFLOAT16,
     ttl.tensor.Layout.ROW_MAJOR,
     ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
     ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), 83.0, 19096254),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed",
    (
        test_sweep_args
    ),
)
def test_eltwise_heaviside_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, device
):
    random.seed(0)
    run_eltwise_heaviside_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, device)
