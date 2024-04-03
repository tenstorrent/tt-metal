# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import groupnorm_noweights as tt_groupnorm_noweights


def run_groupnorm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-10, 10)

    x_ref = x.detach().clone()

    # compute ref value --------------------------
    ref_value = pytorch_ops.groupnorm_noweights(x_ref)

    # compute tt value ---------------------------
    tt_result = tt_groupnorm_noweights(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs -------------
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    assert success


test_sweep_args = [
    (
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        4673250,
    ),
    (
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        9456908,
    ),
    (
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.TILE,
        "SYSTEM_MEMORY",
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        4689090,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_groupnorm_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_groupnorm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
