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

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0


def tensor_to_device(x, device, buffer_type):
    if buffer_type == None:
        return x

    return x.to(device, buffer_type)

def run_eltwise_softmax_in_place_tests(input_shape, dtype, dlayout, in_mem_config, data_seed, device):
    torch.manual_seed(data_seed)
    test_dims = (input_shape,)

    input_mem_config = in_mem_config
    if in_mem_config == "SYSTEM_MEMORY":
        input_mem_config = None

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 1):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            x_ref = x.detach().clone()

            t0 = ttl.tensor.Tensor(
                x.reshape(-1).tolist(),
                [N, C, H, W],
                dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            t0 = t0.to(dlayout)
            ttx = tensor_to_device(t0, device, input_mem_config)

            logger.info("Running eltwise eltwise softmax-in-place test")
            ttz = ttl.operations.primary.softmax_in_place(ttx)
            logger.info("Done")

            # check memory configs
            if in_mem_config != "SYSTEM_MEMORY":
                assert ttx.memory_config().buffer_type == in_mem_config.buffer_type
                logger.debug(f"ttx is on: {ttx.memory_config().buffer_type}")
            else:
                logger.debug(f"ttx is on: SYSTEM_MEMORY")

            # comapre results
            tt_result = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
            # get ref result
            ref_value = pytorch_ops.softmax_in_place(x_ref)
            # compare tt and golden outputs
            success, pcc_value = comp_pcc(ref_value, tt_result)
            logger.debug(pcc_value)

            assert success


test_sweep_args=[
    # only test that passes out of all sweeps run for pytorch_eltwise_softmax_in_place_test.yaml:
    ((1, 9, 32, 32), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), 38346),
    # rest failed:
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 17155532),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), 16305027),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, data_seed",
    (
        test_sweep_args
    ),
)

def test_eltwise_softmax_in_place_test(
    input_shape, dtype, dlayout, in_mem_config, data_seed, device
):
    run_eltwise_softmax_in_place_tests(input_shape, dtype, dlayout, in_mem_config, data_seed, device)
