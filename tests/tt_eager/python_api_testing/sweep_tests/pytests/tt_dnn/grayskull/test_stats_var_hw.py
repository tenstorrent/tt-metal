# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import var_hw as tt_var_hw


def run_var_hw_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-10, 10)
    x = torch.where(x.abs() > 1e-3, x, 1e-3)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.var_hw(x_ref)

    tt_result = tt_var_hw(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result, pcc=0.91)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (7, 14, 32, 160),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17155532,
    ),
    (
        (7, 14, 32, 160),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        16305027,
    ),
    (
        (5, 3, 54, 98),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        19700662,
    ),
    (
        (5, 3, 54, 98),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        9765530,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_var_hw_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_var_hw_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
