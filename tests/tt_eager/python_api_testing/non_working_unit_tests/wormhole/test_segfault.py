# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import complex_mul as tt_complex_mul
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_complex


def run_complex_mul_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = gen_rand_complex(size=input_shape, low=-100, high=100)
    y = gen_rand_complex(size=input_shape, low=-100, high=100)

    # compute ref value
    ref_value = pytorch_ops.complex_mul(x, y)

    tt_result = tt_complex_mul(
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
        (1, 6, 128, 448),
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        7340822,
    ),
    (
        (1, 6, 128, 448),
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        9741841,
    ),
    (
        (1, 6, 128, 448),
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        1517803,
    ),
    (
        (1, 10, 224, 256),
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        6169610,
    ),
    (
        (1, 10, 224, 256),
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        19315642,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_mul(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    run_complex_mul_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
