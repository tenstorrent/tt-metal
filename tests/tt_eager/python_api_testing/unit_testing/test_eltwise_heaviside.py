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
def ref_eltwise_heaviside(x, scalar):
    result = torch.heaviside(x, torch.tensor(scalar))
    return result

def run_eltwise_heaviside_tests(input_shape, scalar, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)
    test_dims = (input_shape,)

    input_mem_config = in_mem_config
    if in_mem_config == "SYSTEM_MEMORY":
        input_mem_config = None

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 1):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            x_ref = x.detach().clone()

            ttx = ttl.tensor.Tensor(
                x.reshape(-1).tolist(),
                [N, C, H, W],
                dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            ttx = ttx.to(dlayout)

            ttx = tensor_to_device(ttx, device, input_mem_config)


            logger.info("Running eltwise heaviside test")
            ttz = ttl.tensor.heaviside(ttx, scalar, output_mem_config=out_mem_config)
            logger.info("Done")

            # check memory configs
            if in_mem_config != "SYSTEM_MEMORY":
                assert ttx.memory_config().buffer_type == in_mem_config.buffer_type
                logger.debug(f"ttx is on: {ttx.memory_config().buffer_type}")
            else:
                logger.debug(f"ttx is on: SYSTEM_MEMORY")

            assert ttz.memory_config().buffer_type == out_mem_config.buffer_type
            logger.debug(f"ttz is on: {ttz.memory_config().buffer_type}")

            # comapre results
            t2_data = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
            tt_got_back = torch.Tensor(t2_data).reshape((N, C, H, W))
            # Can we go without this????
            # if dlayout == ttl.tensor.Layout.TILE:
            #     tt_got_back = untilize(tt_got_back)

            # get referent value
            ref_value = ref_eltwise_heaviside(x_ref, scalar)

            # compare tt and golden outputs
            success, pcc_value = comp_pcc(tt_got_back, ref_value)
            logger.debug(pcc_value)

            assert success


test_sweep_args=[
    ((6, 2, 216, 186), 82.0, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 13482735),
    ((6, 2, 216, 186), -83.5, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR,  ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 494232),
    ((6, 2, 216, 186), 15.625, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR,  ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 4379583),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, scalar, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (
        test_sweep_args
    ),
)

def test_eltwise_heaviside_test(
    input_shape, scalar, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    run_eltwise_heaviside_tests(input_shape, scalar, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
