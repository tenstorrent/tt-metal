# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import tt_lib as ttl
import os

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops, tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_sum_0_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(0, 100).to(torch.bfloat16)
    ref_value = pytorch_ops.sum(x, dim=0)

    tt_result = tt_lib_ops.sum(
        x=x,
        dim=0,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17155532,
    ),
    (
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        16305027,
    ),
    (
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        None,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        13587334,
    ),
]


# unit test which reproduce low PCC error
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_sum_0_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_sum_0_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
