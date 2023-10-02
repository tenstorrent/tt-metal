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

from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import fill_rm as pt_fill_rm
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import fill_rm as tt_fill_rm


def run_fill_rm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, h_ones, w_ones, val_hi, val_lo, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
        x_ref = x.detach().clone()

        # get referent value ----------------------------------------
        ref_value = pt_fill_rm(
            x_ref,
            hOnes=h_ones,
            wOnes=w_ones,
            val_hi=val_hi,
            val_lo=val_lo
        )

        # # get tt input calculate tt output ---------------------------------------
        if in_mem_config == "SYSTEM_MEMORY":
            in_mem_config = None

        logger.info("Running fill rm test")
        tt_result = tt_fill_rm(
            x,
            hOnes=h_ones,
            wOnes=w_ones,
            val_hi=val_hi,
            val_lo=val_lo,
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
    ((7, 14, 22, 134), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 7, 130, -72.14525896036947, -72.05084300666422, 17869870),
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
