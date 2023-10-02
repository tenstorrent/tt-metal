# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
from loguru import logger
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

from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import ne as pt_ne
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_ne as tt_ne


def run_eltwise_ne_tests(input_shape, in0_dtype, in1_dtype, in0_dlayout, in1_dlayout, in0_in_mem_config, in1_in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    test_dims = (input_shape,)

    input0_mem_config = in0_in_mem_config
    if in0_in_mem_config == "SYSTEM_MEMORY":
        input0_mem_config = None

    input1_mem_config = in1_in_mem_config
    if in1_in_mem_config == "SYSTEM_MEMORY":
        input1_mem_config = None

    for N, C, H, W in test_dims:
        x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
        y = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)

        x_ref = x.detach().clone()
        y_ref = y.detach().clone()

        # get referent value
        ref_value = pt_ne(x_ref, y_ref)

        # calculate tt output
        if in0_in_mem_config == "SYSTEM_MEMORY":
            in0_in_mem_config = None

        if in1_in_mem_config == "SYSTEM_MEMORY":
            in1_in_mem_config = None

        logger.info("Running eltwise_ne test")
        tt_result = tt_ne(
            x=x,
            y=y,
            device=device,
            device_id=0,
            dtype=[in0_dtype, in1_dtype],
            layout=[in0_dlayout, in1_dlayout],
            input_mem_config=[in0_in_mem_config, in1_in_mem_config],
            output_mem_config=out_mem_config
        )
        logger.info("Done")

        # compare tt and golden outputs
        success, pcc_value = comp_pcc(ref_value, tt_result)
        logger.debug(pcc_value)

        assert success

test_sweep_args=[
    # TILE, TILE
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 17155532),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16305027),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.DRAM, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 13587334),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.L1, ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 10177486),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.L1, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 15991940),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.L1, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 12014143),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 19575052),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 7329721),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16934480),
    # ROW_MAJOR, ROW_MAJOR
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 14073508),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 19451336),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.DRAM, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 9234542),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.L1, ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 15118389),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.L1, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16530771),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.L1, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 11991265),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 2763978),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 10882535),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 3870495),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, in0_dtype, in1_dtype, in0_dlayout, in1_dlayout, in0_in_mem_config, in1_in_mem_config, out_mem_config, data_seed",
    (
        test_sweep_args
    ),
)

def test_eltwise_ne_test(
    input_shape, in0_dtype, in1_dtype, in0_dlayout, in1_dlayout, in0_in_mem_config, in1_in_mem_config, out_mem_config, data_seed, device
):
    run_eltwise_ne_tests(input_shape, in0_dtype, in1_dtype, in0_dlayout, in1_dlayout, in0_in_mem_config, in1_in_mem_config, out_mem_config, data_seed, device)
