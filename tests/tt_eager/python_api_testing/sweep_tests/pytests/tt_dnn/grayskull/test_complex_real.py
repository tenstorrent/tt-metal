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
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import complex_real as tt_complex_real
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_complex


# NB: memory use of complex ops is quadruple the tensor size
def run_complex_real_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = gen_rand_complex(size=input_shape, low=-100, high=100)

    # compute ref value
    ref_value = pytorch_ops.complex_real(x).to(torch.bfloat16)

    tt_result = tt_complex_real(
        x=x,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        (1, 1, 192, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        38346,
    ),
    (
        (1, 1, 192, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        16561724,
    ),
    (
        (1, 1, 192, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        11178160,
    ),
    (
        (1, 1, 192, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        8184290,
    ),
    (
        (1, 1, 160, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        19325774,
    ),
    (
        (1, 1, 160, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        4016313,
    ),
    (
        (1, 1, 96, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        18049894,
    ),
    (
        (1, 1, 96, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        5982363,
    ),
    (
        (1, 1, 96, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        6974090,
    ),
    (
        (1, 1, 96, 416),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        12602337,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_complex_real(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device):
    random.seed(0)
    run_complex_real_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device)
