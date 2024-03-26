# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import reduce_max_w as tt_reduce_max_w
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_reduce_max_w_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    # compute ref value
    x_ref = x.detach().clone()
    ref_value = pytorch_ops.reduce_max(x_ref, dims=(-1,))

    tt_result = tt_reduce_max_w(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


# ,10177486,"(('TT_METAL_SLOW_DISPATCH_MODE', ''),)",completed,"Max ATOL Delta: 0.5, Max RTOL Delta: 0.4296875, PCC: 0.9999901727548534, Equal check failed",fail


test_sweep_args = [
    (
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        10177486,
    ),
]


def test_reduce_max_w_test(device):
    random.seed(0)
    for (
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
    ) in test_sweep_args:
        run_reduce_max_w_tests(
            input_shape,
            dtype,
            dlayout,
            in_mem_config,
            out_mem_config,
            data_seed,
            device,
        )
