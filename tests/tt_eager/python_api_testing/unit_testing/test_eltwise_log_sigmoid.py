# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
import random
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")

import pytest
import torch

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0


def tensor_to_device(x, device, buffer_type):
    if buffer_type == None:
        return x

    return x.to(device, buffer_type)

def run_eltwise_log_sigmoid_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)
    test_dims = (input_shape,)

    input_mem_config = in_mem_config
    if in_mem_config == "SYSTEM_MEMORY":
        input_mem_config = None

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 1):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-10, 10)
            x_ref = x.detach().clone()

            t0 = ttl.tensor.Tensor(x, dtype)
            t0 = t0.to(dlayout)
            ttx = tensor_to_device(t0, device, input_mem_config)

            logger.info("Running eltwise log sigmoid test")
            ttz = ttl.tensor.log_sigmoid(ttx, output_mem_config=out_mem_config)
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
            tt_result = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
            # get ref result
            ref_value = pytorch_ops.log_sigmoid(x_ref)
            # compare tt and golden outputs
            success, pcc_value = comp_pcc(ref_value, tt_result)
            logger.debug(pcc_value)

            assert success


test_sweep_args=[
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE,  ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 17155532),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE,  ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16305027),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 13587334),
    ((6, 4, 156, 214), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 19325774),
    ((6, 4, 156, 214), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 4016313),
    ((6, 4, 156, 214), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 13126809),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (
        test_sweep_args
    ),
)
def test_eltwise_log_sigmoid_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    run_eltwise_log_sigmoid_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
