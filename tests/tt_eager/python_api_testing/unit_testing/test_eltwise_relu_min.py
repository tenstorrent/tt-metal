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


# This ref implementation is only here for debugging
def ref_eltwise_relu_min(x, lower_limit):

    constant_tensor = torch.full(x.shape, lower_limit)

    # relu activation function
    y = torch.max(x, constant_tensor)

    return y


def run_eltwise_relu_min_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, lower_limit, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    lower_limit = lower_limit

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 1):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)

            if in_mem_config == "SYSTEM_MEMORY":
                ttx = tensor.Tensor(
                    tilize_to_list(x),
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                ).cpu()
            else:
                ttx = tensor.Tensor(
                    tilize_to_list(x),
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                    in_mem_config,
                )

            logger.info("Running Eltwise relu min test")
            ttz = tensor.relu_min(ttx, lower_limit, output_mem_config=out_mem_config)

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
            tt_got_back = untilize(tt_got_back)

            # get referent value
            ref_value = ref_eltwise_relu_min(x, lower_limit)

            # compare tt and golden outputs
            success, pcc_value = comp_pcc(tt_got_back, ref_value)
            logger.debug(pcc_value)

            assert success


test_sweep_args=[
    ((1, 10, 192, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 99.0, 895795),
    ((2, 4, 192, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 99.5, 14073508),
    ((6, 9, 192, 128), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 97.5, 9248746),
    ((3, 2, 192, 32), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 98.0, 18784230),
    ((3, 8, 96, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 98.0, 16934480),
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
