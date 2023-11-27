# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests.common import set_dispatch_mode
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import reduce_sum_w as tt_reduce_sum_w


def run_reduce_sum_w_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device):
    torch.manual_seed(data_seed)
    set_dispatch_mode(dispatch_mode)

    if in_mem_config[0] == "SYSTEM_MEMORY":
        in_mem_config[0] = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.reduce_sum(x_ref, dims=(-1,))

    tt_result = tt_reduce_sum_w(
        x=x,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )
    # compare tt and golden outputs

    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        (4, 4, 92, 30),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        ["SYSTEM_MEMORY"],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17643447,
        False,
    ),
    (
        (1, 7, 4, 26),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        ["SYSTEM_MEMORY"],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17414875,
        False,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_reduce_sum_w_mul(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device):
    random.seed(0)
    run_reduce_sum_w_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device)
