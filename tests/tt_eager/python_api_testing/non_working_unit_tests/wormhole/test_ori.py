# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import logical_ori as pt_ori
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_logical_ori as tt_ori
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_complex
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_ori_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, immediate, data_seed, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = pt_ori(x=x_ref, immediate=immediate)

    tt_result = tt_ori(
        x=x,
        immediate=immediate,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert success


test_sweep_args = [
    (
        (5, 11, 252, 22),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        66,
        9394661,
        "1",
    ),
    (
        (5, 11, 252, 22),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        84,
        13482735,
        "1",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, immediate, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_ori_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, immediate, data_seed, dispatch_mode, device
):
    run_ori_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, immediate, data_seed, dispatch_mode, device
    )
