# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import numpy as np
import random
from itertools import product
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
import tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops as pytorch_ops
import tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops as tt_lib_ops


def run_ternary_bert_tests(
    tt_op, pt_op, input_shapes, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    torch.manual_seed(data_seed)

    x = gen_rand_complex(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    y = gen_rand_complex(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    z = torch.Tensor(size=input_shapes[2]).uniform_(-100, 100).to(torch.bfloat16)

    x = torch.where(x.abs() > 1e-3, x, 1e-3)
    y = torch.where(y.abs() > 1e-3, y, 1e-3)
    z = torch.where(z.abs() > 1e-3, z, 1e-3)

    # get referent value
    ref_value = pt_op(x, y, z)

    # calculate tt output
    logger.info(f"Running {tt_op} test")
    tt_result = tt_op(
        x=x,
        y=y,
        z=z,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )
    logger.info("Done")

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


supported_ternary_data_types = [
    [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
    [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
    [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
    [
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ],
    [
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ],
    [
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ],
    [
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ],
]

test_sweep_args_ternary = []

for dtype_0, dtype_1, dtype_2, mem_cfg_0, mem_cfg_1, mem_cfg_2, out_mem_cfg in product(*supported_ternary_data_types):
    test_sweep_args_ternary.append(
        (
            tt_lib_ops.bert_large_ff1_matmul,
            pytorch_ops.bert_large_ff1_matmul,
            [(9, 1, 384, 1024), (1, 1, 1024, 4096), (1, 1, 1, 4096)],
            [dtype_0, dtype_1, dtype_2],
            [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            [mem_cfg_0, mem_cfg_1, mem_cfg_2],
            out_mem_cfg,
            random.randint(0, 20000000),
        )
    )

    test_sweep_args_ternary.append(
        (
            tt_lib_ops.bert_large_ff2_matmul,
            pytorch_ops.bert_large_ff2_matmul,
            [(9, 1, 384, 4096), (1, 1, 4096, 1024), (1, 1, 1, 1024)],
            [dtype_0, dtype_1, dtype_2],
            [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            [mem_cfg_0, mem_cfg_1, mem_cfg_2],
            out_mem_cfg,
            random.randint(0, 20000000),
        )
    )

    test_sweep_args_ternary.append(
        (
            tt_lib_ops.bert_large_fused_qkv_matmul,
            pytorch_ops.bert_large_fused_qkv_matmul,
            [(9, 1, 384, 1024), (1, 1, 1024, 3072), (1, 1, 1, 3072)],
            [dtype_0, dtype_1, dtype_2],
            [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            [mem_cfg_0, mem_cfg_1, mem_cfg_2],
            out_mem_cfg,
            random.randint(0, 20000000),
        )
    )

    test_sweep_args_ternary.append(
        (
            tt_lib_ops.bert_large_selfout_matmul,
            pytorch_ops.bert_large_selfout_matmul,
            [(9, 1, 384, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)],
            [dtype_0, dtype_1, dtype_2],
            [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            [mem_cfg_0, mem_cfg_1, mem_cfg_2],
            out_mem_cfg,
            random.randint(0, 20000000),
        )
    )


@pytest.mark.parametrize(
    "tt_op, pt_op, input_shapes, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args_ternary),
)
def test_ternary_bert_tests(
    tt_op, pt_op, input_shapes, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    random.seed(0)
    run_ternary_bert_tests(tt_op, pt_op, input_shapes, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
