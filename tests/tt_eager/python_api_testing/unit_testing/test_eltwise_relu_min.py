# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")

import pytest
import torch

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0

from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import relu_min as pt_relu_min
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_relu_min as tt_relu_min


def run_eltwise_relu_min_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    lower_limit = lower_limit

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
        x_ref = x.detach().clone()

        # get referent value
        ref_value = pt_relu_min(x_ref, lower_limit=lower_limit)

        # calculate tt output
        if in_mem_config == "SYSTEM_MEMORY":
            in_mem_config = None

        logger.info("Running Eltwise relu min test")
        tt_result = tt_relu_min(
            x=x,
            lower_limit=lower_limit,
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
    ((1, 10, 192, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 99.0, 895795),
    ((2, 4, 192, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 99.5, 14073508),
    ((6, 9, 192, 128), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 97.5, 9248746),
    ((3, 2, 192, 32), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 98.0, 18784230),
    ((3, 8, 96, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 98.0, 16934480),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 98.5, 13587334),

]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed",
    (
        test_sweep_args
    ),
)

def test_eltwise_relu_min_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed, device
):
    run_eltwise_relu_min_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed, device)
