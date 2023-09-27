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


def tensor_to_device(x, device, buffer_type):
    if buffer_type == None:
        return x

    return x.to(device, buffer_type)


# This ref implementation is only here for debugging
def ref_fill_rm(x, hOnes, wOnes, val_hi, val_lo):
    y = x
    y[:, :, :, :] = val_lo
    y[:, :, 0:hOnes, 0:wOnes] = val_hi

    return y


def run_fill_rm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, h_ones, w_ones, val_hi, val_lo, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 1):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)

            # get referent value ----------------------------------------
            ref_value = ref_fill_rm(x, h_ones, w_ones, val_hi, val_lo)

            # # get tt input calculate tt output ---------------------------------------
            if in_mem_config == "SYSTEM_MEMORY":
                in_mem_config = None

            t0 = ttl.tensor.Tensor(
                x.reshape(-1).tolist(),
                [N, C, H, W],
                dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )

            # Layout must be row mayor
            t0 = t0.to(dlayout)
            ttx = tensor_to_device(t0, device, in_mem_config)

            # calculate tt output ---------------------------------------
            logger.info("Running fill rm test")
            ttz = ttl.tensor.fill_rm(N, C, H, W, h_ones, w_ones, ttx, val_hi, val_lo, output_mem_config=out_mem_config)
            tt_got_back = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
            logger.info("Done")

            assert ttz.memory_config().buffer_type == out_mem_config.buffer_type
            logger.debug(f"ttz is on: {ttz.memory_config().buffer_type}")

            success, pcc_value = comp_pcc(ref_value, tt_got_back)
            logger.debug(pcc_value)

            assert success


test_sweep_args=[
    ((7, 14, 22, 134), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 7, 130, -72.14525896036947, -72.05084300666422, 17869870),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, h_ones, w_ones, val_hi, val_lo, data_seed",
    (
        test_sweep_args
    ),
)

def test_fill_rm_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, h_ones, w_ones, val_hi, val_lo, data_seed, device
):
    run_fill_rm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, h_ones, w_ones, val_hi, val_lo, data_seed, device)
