# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
from loguru import logger
import numpy as np
import random

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

from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import polyval as pt_polyval
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_polyval as tt_polyval


def run_eltwise_polyval_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, coeffs, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    coeffs = coeffs

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
        x_ref = x.detach().clone()

        # get referent value
        ref_value = pt_polyval(x_ref, coeffs=coeffs)

        # get tt input
        if in_mem_config == "SYSTEM_MEMORY":
            in_mem_config = None

        # calculate tt output
        logger.info("Running Eltwise Polyval test")
        tt_result = tt_polyval(
            x=x,
            coeffs=coeffs,
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
    # Date: 2023-09-22 09:15
    ((4, 12, 147, 108), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [-48.0595703125], 1569665),
    ((1, 6, 215, 252), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [93.90789031982422], 1221114),
    ((1, 10, 192, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [70.4070816040039], 2482923),
    ((5, 2, 128, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [65.89027404785156], 11249810),
    ((2, 4, 192, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [-61.05318069458008], 9234542),
    ((4, 9, 96, 32), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [89.58659362792969], 18411293),
    ((3, 2, 192, 32), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [11.359286308288574], 18784230),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, coeffs, data_seed",
    (
        test_sweep_args
    ),
)

def test_eltwise_polyval_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, coeffs, data_seed, device
):
    random.seed(0)
    run_eltwise_polyval_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, coeffs, data_seed, device)
