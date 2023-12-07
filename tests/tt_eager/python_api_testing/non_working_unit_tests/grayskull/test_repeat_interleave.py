# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_repeat_interleave_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, repeat, dim, data_seed, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    x_ref = x.detach().clone()

    ref_value = pytorch_ops.repeat_interleave(
        x=x_ref,
        repeat=repeat,
        dim=dim,
    )

    tt_result = tt_lib_ops.repeat_interleave(
        x=x,
        repeat=repeat,
        dim=dim,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    set_slow_dispatch_mode(prev_dispatch_mode)

    assert success


test_sweep_args = [
    (
        (4, 11, 96, 64),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [None],
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        1,
        0,
        15366393,
        "1",
    ),
    (
        (6, 2, 96, 192),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [None],
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        1,
        2,
        9424875,
        "1",
    ),
    (
        (1, 1, 128, 64),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [None],
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        1,
        2,
        2738700,
        "1",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, repeat, dim, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_repeat_interleave(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, repeat, dim, data_seed, dispatch_mode, device
):
    run_repeat_interleave_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, repeat, dim, data_seed, dispatch_mode, device
    )
