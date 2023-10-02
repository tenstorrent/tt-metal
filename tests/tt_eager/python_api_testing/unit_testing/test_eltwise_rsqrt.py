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
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_rsqrt as tt_eltwise_rsqrt


def run_eltwise_rsqrt_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    fast_and_appx = False

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(0, 100)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.rsqrt(x_ref)

    tt_result = tt_eltwise_rsqrt(
        x=x,
        fast_and_appx=fast_and_appx,
        device=device,
        device_id=0,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args=[
    ((3, 11, 92, 100), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 2653276),
    ((3, 11, 92, 100), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 417293),
    ((3, 11, 92, 100), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 5147678),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 17155532),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 13587334),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 15991940),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (
        test_sweep_args
    ),
)

def test_eltwise_rsqrt_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    run_eltwise_rsqrt_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
