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

def tensor_to_device(x, device, buffer_type):
    if buffer_type == None:
        return x

    return x.to(device, buffer_type)

# This ref implementation is only here for debugging
def ref_eltwise_ne(x, y):
    # ne function
    z = torch.ne(x, y)

    return z

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
        for nrepeat in range(0, 1):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            y = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)

            # get referent value
            ref_value = ref_eltwise_ne(x, y)

            # get tt inputs
            t0 = ttl.tensor.Tensor(x, in0_dtype)
            t0 = t0.to(in0_dlayout)
            ttx = tensor_to_device(t0, device, input0_mem_config)

            t1 = ttl.tensor.Tensor(y, in1_dtype)
            t1 = t1.to(in1_dlayout)
            tty = tensor_to_device(t1, device, input1_mem_config)

            # calculate tt output
            logger.info("Running eltwise_ne test")
            ttz = ttl.tensor.ne(ttx, tty, output_mem_config=out_mem_config)
            tt_got_back = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
            logger.info("Done")

            # compare tt and golden outputs
            success, pcc_value = comp_pcc(ref_value, tt_got_back)
            logger.debug(pcc_value)

            assert success

test_sweep_args=[
    # TILE, TILE
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 17155532),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16305027),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 13587334),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 10177486),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 15991940),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 12014143),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 19575052),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 7329721),
    ((7, 14, 32, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16934480),
    # ROW_MAJOR, ROW_MAJOR
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 14073508),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 19451336),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 9234542),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 15118389),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16530771),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 11991265),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 2763978),
    ((4, 22, 303, 424), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 10882535),
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
