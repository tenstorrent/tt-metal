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
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import transpose_cn as tt_transpose_cn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_transpose_cn_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    # compute ref value
    x_ref = x.detach().clone()
    ref_value = pytorch_ops.transpose(x_ref, dim0=0, dim1=1)

    tt_result = tt_transpose_cn(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_equal(ref_value, tt_result)
    print(f"ref value: {ref_value[0, 0, 1:10, 1:10]}")
    print(f"tt value: {tt_result[0, 0, 1:10, 1:10]}")
    logger.debug(pcc_value)
    logger.debug(success)

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert success


# PCC: 0.9992681541588561, Equal check failed	fail	NA	NA	NA	Details

test_sweep_args = [
    (
        (12, 15, 476, 206),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        "SYSTEM_MEMORY",
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        14150048,
        "1",
    ),
]


# @pytest.mark.parametrize(
#     "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode",
#     (test_sweep_args),
# )
# def test_transpose_cn_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device):
def test_transpose_cn_test(device):
    random.seed(0)

    input_shape = (12, 15, 476, 206)
    dtype = ttl.tensor.DataType.BFLOAT16
    dlayout = ttl.tensor.Layout.ROW_MAJOR
    in_mem_config = "SYSTEM_MEMORY"
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    data_seed = 14150048
    dispatch_mode = "1"

    run_transpose_cn_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device)
