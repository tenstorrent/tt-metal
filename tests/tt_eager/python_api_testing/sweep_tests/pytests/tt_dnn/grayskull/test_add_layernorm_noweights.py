# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import (
    add_layernorm_noweights as tt_add_layernorm_noweights,
)


def run_add_layernorm_noweights_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-10.0, 10.0)
    x_ref = x.detach().clone()

    y = torch.Tensor(size=input_shape).uniform_(-10.0, 10.0)
    y_ref = y.detach().clone()

    # get ref result
    ref_value = pytorch_ops.add_layernorm_noweights(x_ref, y_ref)

    tt_result = tt_add_layernorm_noweights(
        x=x,
        y=y,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(
        ref_value,
        tt_result,
    )
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (1, 1, 32, 32),
        (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16),
        (ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE),
        (
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        38346,
    ),
    (
        (1, 1, 32, 32),
        (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16),
        (ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE),
        (
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        38346,
    ),
    (
        (1, 1, 32, 96),
        (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16),
        (ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE),
        (
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        38346,
    ),
]


@pytest.mark.skip("FAIL")
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_add_layernorm_noweights_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_add_layernorm_noweights_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
