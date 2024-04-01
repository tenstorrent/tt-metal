# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops, tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_eltwise_rdiv_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.eltwise_rdiv(x=x_ref, factor=factor)

    tt_result = tt_lib_ops.eltwise_rdiv(
        x=x,
        factor=factor,
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
        (4, 24, 192, 384),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)),
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        4781318,
        1.9915642058736664,
    ),
    (
        (10, 22, 160, 256),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        "SYSTEM_MEMORY",
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        2692686,
        1.3892220599951342,
    ),
    (
        (11, 18, 320, 352),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        19325774,
        0.8325692531786724,
    ),
    (
        (12, 14, 448, 352),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        5371386,
        0.8991587580374605,
    ),
    (
        (11, 22, 448, 104),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        "SYSTEM_MEMORY",
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        1190117,
        0.19607612839476013,
    ),
    (
        (9, 20, 214, 424),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)),
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        7837345,
        8.47037061625196,
    ),
    (
        (11, 10, 280, 312),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        16699356,
        1.6465889871095842,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor",
    (test_sweep_args),
)
def test_eltwise_rdiv(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor, device):
    run_eltwise_rdiv_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor, device)
