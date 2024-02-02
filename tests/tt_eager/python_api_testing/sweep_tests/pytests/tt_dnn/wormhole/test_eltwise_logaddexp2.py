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
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_logaddexp2 as tt_eltwise_logaddexp2
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_eltwise_logaddexp2_test(
    input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    torch.manual_seed(data_seed)

    x = gen_rand(size=input_shape_1, low=-64, high=64).to(torch.bfloat16)
    y = gen_rand(size=input_shape_2, low=-64, high=64).to(torch.bfloat16)
    x = torch.where(x.abs() > 1e-3, x, 1e-3)
    y = torch.where(y.abs() > 1e-3, y, 1e-3)

    # compute ref value
    x_ref = x.detach().clone()
    y_ref = y.detach().clone()

    ref_value = pytorch_ops.logaddexp2(x_ref, y_ref)

    tt_result = tt_eltwise_logaddexp2(
        x=x,
        y=y,
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
        (2, 5, 37, 194),
        (2, 5, 37, 194),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR],
        [None, None],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        15842480,
    ),
    (
        (2, 5, 37, 194),
        (2, 5, 37, 194),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        10406825,
    ),
    (
        (2, 5, 37, 194),
        (2, 5, 37, 194),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR],
        [None, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        2474385,
    ),
    (
        (4, 7, 32, 96),
        (4, 7, 32, 96),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17155532,
    ),
    (
        (4, 7, 32, 96),
        (4, 7, 32, 96),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        16305027,
    ),
    (
        (4, 7, 32, 96),
        (4, 7, 32, 96),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), None],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        13587334,
    ),
]


@pytest.mark.parametrize(
    "input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_logaddexp2_test(
    input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    random.seed(0)
    run_eltwise_logaddexp2_test(
        input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
    )
