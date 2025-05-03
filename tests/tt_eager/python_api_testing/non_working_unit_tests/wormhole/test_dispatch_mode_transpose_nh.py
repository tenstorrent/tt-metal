# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import sys
from loguru import logger
import random
import pytest
import torch
import random
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import setup_tt_tensor, tt2torch_tensor
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def tt_transpose_nh(x, device, dtype, layout, input_mem_config, output_mem_config):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transpose(t0, 0, 2, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


def gen_shapes(start_shape, end_shape, interval, num_shapes):
    result = []
    num_dims = len(start_shape)
    dim_ranges = [range(start_shape[i], end_shape[i] + interval[i], interval[i]) for i in range(num_dims)]

    for i in range(num_shapes):
        result.append(
            [
                random.choice(dim_ranges[0]),
                random.choice(dim_ranges[1]),
                random.choice(dim_ranges[2]),
                random.choice(dim_ranges[3]),
            ]
        )
    return result


def run_transpose_nh_tests(dtype, dlayout, in_mem_config, out_mem_config, device):
    if dlayout == ttnn.ROW_MAJOR_LAYOUT:
        shapes = gen_shapes([1, 1, 2, 2], [12, 24, 512, 512], [1, 1, 2, 2], 256)
    else:
        shapes = gen_shapes([1, 1, 32, 32], [12, 24, 512, 512], [1, 1, 32, 32], 256)

    overall_pass = True

    for input_shape in shapes:
        data_seed = random.randint(0, 20000000)
        torch.manual_seed(data_seed)

        logger.info(f"Running with shape: {input_shape} and seed: {data_seed}")
        logger.debug(f"Running test with args:{input_shape}, {dlayout}, {dtype},{in_mem_config}, {out_mem_config}")

        x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)
        # compute ref value

        tt_result = tt_transpose_nh(
            x=x,
            device=device,
            dtype=[dtype],
            layout=[dlayout],
            input_mem_config=[in_mem_config],
            output_mem_config=out_mem_config,
        )
        ref_value = pytorch_ops.transpose(x, dim0=0, dim1=2)

        # compare tt and golden outputs
        success, pcc_value = comp_equal(ref_value, tt_result)
        logger.debug(pcc_value)
        logger.debug(success)

        if not success:
            overall_pass = False

    assert overall_pass


test_sweep_args = [
    (
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ),
    (
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ),
    (
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        None,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ),
]


@pytest.mark.parametrize(
    "dtype, dlayout, in_mem_config, out_mem_config",
    (test_sweep_args),
)
def test_transpose_nh_test(dtype, dlayout, in_mem_config, out_mem_config, device):
    random.seed(0)
    run_transpose_nh_tests(dtype, dlayout, in_mem_config, out_mem_config, device)
