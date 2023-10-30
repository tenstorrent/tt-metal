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
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_rdiv as tt_eltwise_rdiv
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand

def set_dispatch_mode(set_var):
    if set_var:
        dispatch = os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
        os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
        logger.info("Set slow dispatch mode")
    else:
        dispatch = os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
        os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = ""
        logger.info("Set fast dispatch mode")

def run_eltwise_rdiv_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.eltwise_rdiv(x=x_ref, factor=factor)

    tt_result = tt_eltwise_rdiv(
        x=x,
        factor=factor,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


# eltwise-rdiv,"[[4, 24, 192, 384]]","{'dtype': [<DataType.BFLOAT16: 0>], 'layout': [<Layout.TILE: 1>], 'input_mem_config': [tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::L1)], 'output_mem_config': tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::DRAM), 'factor': 1.9915642058736664}",4781318,"(('TT_METAL_SLOW_DISPATCH_MODE', ''),)",completed,"Max ATOL Delta: inf, Max RTOL Delta: inf, PCC: 6.787768461855196e-08, PCC check failed",fail

test_sweep_args=[
    ((4, 24, 192, 384), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE,  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)),  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)), 1.9915642058736664, 4781318),
    # ((11, 18, 320, 352), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE,  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)), 1.6659720483442477, 19325774),
    # ((12, 14, 448, 352), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE,  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)), 1.7265079618522368, 5371386),
    # ((11, 3, 448, 384), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE,  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)),  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)), 1.132563580694432, 10609144),

]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor",
    (
        test_sweep_args
    ),
)

def test_eltwise_rdiv(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor, device
):
    run_eltwise_rdiv_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor, device)
