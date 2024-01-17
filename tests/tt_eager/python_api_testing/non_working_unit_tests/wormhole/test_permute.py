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
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import permute as tt_permute
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_permute_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, permute_dims, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    # compute ref value
    x_ref = x.detach().clone()
    ref_value = pytorch_ops.permute(x_ref, permute_dims=permute_dims)

    tt_result = tt_permute(
        x=x,
        permute_dims=permute_dims,
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

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert success


test_sweep_args = [
    (
        (3, 2, 98, 68),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        11417914,
        (2, 3, 0, 1),
        "1",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, permute_dims, dispatch_mode",
    (test_sweep_args),
)
def test_permute_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, permute_dims, dispatch_mode, device
):
    random.seed(0)
    run_permute_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, permute_dims, dispatch_mode, device
    )
