# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
from loguru import logger
import numpy as np

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


# This ref implementation is only here for debugging
def ref_eltwise_polyval(x, coeffs):
    # polyval function
    # y = np.polyval(np.poly1d(coeffs), x.numpy())
    # y = torch.from_numpy(y)

    result = 0.0
    for coeff in coeffs:
        result = result * x + coeff

    return result


def ref_eltwise_polyval_2(x, coeffs):
    x = torch.FloatTensor(x)
    coeffs = torch.FloatTensor(coeffs)

    curVal=0
    for curValIndex in range(len(coeffs)-1):
        curVal=(curVal+coeffs[curValIndex])*x[0]
    return(curVal+coeffs[len(coeffs)-1])


# coeffs=torch.FloatTensor([1.,2.,1.,1.])
# x=torch.FloatTensor([3.0])
# getPolyVal(x,coeffs)


def run_eltwise_polyval_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, coeffs, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    coeffs = coeffs

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 1):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            x_ref = x.detach().clone()

            if dlayout == ttl.tensor.Layout.TILE:
                x = tilize_to_list(x)
            else:
                x = x.reshape(-1).tolist()

            if in_mem_config == "SYSTEM_MEMORY":
                ttx = tensor.Tensor(
                    x,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                ).cpu()
            else:
                ttx = tensor.Tensor(
                    x,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                    in_mem_config,
                )

            logger.info("Running Eltwise Polyval test")
            ttz = tensor.polyval(ttx, coeffs, output_mem_config=out_mem_config)

            logger.info("Done")

            if in_mem_config != "SYSTEM_MEMORY":
                assert ttx.memory_config().buffer_type == in_mem_config.buffer_type
                logger.debug(f"ttx is on: {ttx.memory_config().buffer_type}")
            else:
                logger.debug(f"ttx is on: SYSTEM_MEMORY")

            assert ttz.memory_config().buffer_type == out_mem_config.buffer_type
            logger.debug(f"ttz is on: {ttz.memory_config().buffer_type}")

            t2_data = ttz.cpu().to_torch()

            tt_got_back = torch.Tensor(t2_data).reshape((N, C, H, W))

            if dlayout == ttl.tensor.Layout.TILE:
                tt_got_back = untilize(tt_got_back)

            # get referent value
            ref_value = ref_eltwise_polyval(x_ref, coeffs)

            # compare tt and golden outputs
            success, pcc_value = comp_pcc(tt_got_back, ref_value)
            logger.debug(pcc_value)

            assert success


test_sweep_args=[
    ((4, 12, 147, 108), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [-48.0595703125],	1569665),
    ((1, 6, 215, 252), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [93.90789031982422],	1221114),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [-84.57501220703125, -41.85026931762695],	17155532),
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
    run_eltwise_polyval_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, coeffs, data_seed, device)
