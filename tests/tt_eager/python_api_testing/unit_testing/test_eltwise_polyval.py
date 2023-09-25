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


def tensor_to_device(x, device, buffer_type):
    if buffer_type == None:
        return x

    return x.to(device, buffer_type)


# This ref implementation is only here for debugging
def ref_eltwise_polyval(x, coeffs):
    # polyval function
    y = np.polyval(np.poly1d(coeffs), x.numpy())
    y = torch.from_numpy(y)

    return y

    # result = 0.0
    # for coeff in coeffs:
    #     result = result * x + coeff

    # return result


def run_eltwise_polyval_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, coeffs, data_seed, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    coeffs = coeffs

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 1):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)

            # get referent value
            ref_value = ref_eltwise_polyval(x, coeffs)

            # calculate tt outpu
            if in_mem_config == "SYSTEM_MEMORY":
                in_mem_config = None

            t0 = ttl.tensor.Tensor(x, dtype)
            t0 = t0.to(dlayout)
            ttx = tensor_to_device(t0, device, in_mem_config)

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

            tt_got_back = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

            if dlayout == ttl.tensor.Layout.TILE:
                tt_got_back = untilize(tt_got_back)

            # compare tt and golden outputs
            success, pcc_value = comp_pcc(tt_got_back, ref_value)
            logger.debug(pcc_value)

            assert success


test_sweep_args=[
    ((4, 12, 147, 108), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [-48.0595703125], 1569665),
    ((1, 6, 215, 252), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [93.90789031982422], 1221114),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), [-26.537084579467773, -98.70405578613281, -3.4798622131347656, -5.008804798126221, 92.45339965820312, 51.021541595458984, 1.1626243591308594], 17155532),
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
