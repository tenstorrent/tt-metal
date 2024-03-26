# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import sum as tt_sum


def run_sum_3_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config[0] == "SYSTEM_MEMORY":
        in_mem_config[0] = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.sum(x_ref, dim=0)

    tt_result = tt_sum(
        x=x,
        dim=0,
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
        (5, 5, 244, 30),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        14293189,
    ),
    (
        (5, 5, 244, 30),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        4888559,
    ),
    (
        (1, 11, 200, 130),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        14096589,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_sum_3(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    run_sum_3_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
