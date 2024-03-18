# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
import pytest
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
    comp_equal,
)


def run_concat(shapes, dim, device, layout, dtype, input_mem_config, output_mem_config):
    if layout == ttl.tensor.Layout.ROW_MAJOR and dtype == ttl.tensor.DataType.BFLOAT8_B:
        pytest.skip("Illegal config")
    if layout == ttl.tensor.Layout.TILE:
        for shape in shapes:
            if shape[-2] % 32 != 0 or shape[-1] % 32 != 0:
                pytest.skip("Illegal config")
    inputs = []
    tt_inputs = []
    for i in range(len(shapes)):
        shape = torch.Size(shapes[i])
        inputs.append(torch.rand(shape).to(torch.bfloat16))
        tt_inputs.append(
            ttl.tensor.Tensor(
                inputs[i],
                dtype,
            )
            .to(layout)
            .to(device, input_mem_config)
        )

    tt_cpu = torch.concat(inputs, dim)

    tt = ttl.tensor.concat(tt_inputs, dim, output_mem_config)

    tt_dev = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    else:
        passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "shapes, dim",
    (
        (((1, 2, 64, 64),), -1),
        (((1, 1, 64, 64), (1, 1, 128, 64)), -2),
        (((1, 1, 32, 128), (1, 1, 32, 64), (1, 1, 32, 256)), -1),
        (((2, 4, 32, 1280), (2, 3, 32, 1280), (2, 5, 32, 1280), (2, 8, 32, 1280)), 1),
        (((1, 1, 32, 32), (1, 1, 32, 32)), 3),
        (((1, 1, 32, 64), (1, 1, 32, 128)), 3),
        (((1, 1, 32, 128), (1, 1, 32, 64)), 3),
        (((1, 1, 64, 128), (1, 1, 64, 256)), 3),
        (((1, 32, 32, 32), (1, 32, 32, 32)), 2),
        (((2, 4, 32, 1280), (2, 4, 32, 1280)), 3),
        # SD Shapes
        (((2, 1280, 4, 4), (2, 1280, 4, 4)), 1),
        (((2, 640, 32, 32), (2, 320, 32, 32)), 1),
        (((2, 1280, 8, 8), (2, 1280, 8, 8)), 1),
        (((2, 640, 16, 16), (2, 640, 16, 16)), 1),
        (((2, 320, 32, 32), (2, 320, 32, 32)), 1),
    ),
)
@pytest.mark.parametrize(
    "layout, dtype",
    (
        (ttl.tensor.Layout.TILE, ttl.tensor.DataType.BFLOAT16),
        (ttl.tensor.Layout.TILE, ttl.tensor.DataType.BFLOAT8_B),
        (ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.DataType.BFLOAT16),
    ),
)
@pytest.mark.parametrize(
    "input_mem_config",
    (
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.DRAM,
        ),
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    ),
)
@pytest.mark.parametrize(
    "output_mem_config",
    (
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.DRAM,
        ),
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    ),
)
def test_concat(shapes, dim, device, layout, dtype, input_mem_config, output_mem_config, function_level_defaults):
    run_concat(shapes, dim, device, layout, dtype, input_mem_config, output_mem_config)


@pytest.mark.parametrize(
    "shapes, dim",
    (
        (((1, 1, 64, 64), (1, 1, 128, 64)), -2),
        (((1, 1, 32, 128), (1, 1, 32, 64), (1, 1, 32, 256)), -1),
        (((2, 4, 32, 1280), (2, 3, 32, 1280), (2, 5, 32, 1280), (2, 8, 32, 1280)), 1),
        (((2, 4, 32, 1280), (2, 4, 32, 1280)), 3),
        # SD Shapes
        (((2, 1280, 4, 4), (2, 1280, 4, 4)), 1),
        (((2, 320, 32, 32), (2, 320, 32, 32)), 1),
        (((2, 1280, 8, 8), (2, 1280, 8, 8)), 1),
    ),
)
@pytest.mark.parametrize(
    "layout, dtype",
    (
        (ttl.tensor.Layout.TILE, ttl.tensor.DataType.BFLOAT16),
        (ttl.tensor.Layout.TILE, ttl.tensor.DataType.BFLOAT8_B),
        (ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.DataType.BFLOAT16),
    ),
)
@pytest.mark.parametrize(
    "input_mem_config",
    (
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.DRAM,
        ),
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    ),
)
@pytest.mark.parametrize(
    "output_mem_config",
    (
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.DRAM,
        ),
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    ),
)
def test_concat_with_program_cache(
    shapes, dim, device, layout, dtype, input_mem_config, output_mem_config, use_program_cache, function_level_defaults
):
    run_concat(shapes, dim, device, layout, dtype, input_mem_config, output_mem_config)
    tmp = ttl.tensor.empty([1, 256, 32, 32], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    run_concat(shapes, dim, device, layout, dtype, input_mem_config, output_mem_config)


@pytest.mark.parametrize(
    "input_shape, shard_shape, output_shard_shape, shard_grid",
    (
        (
            (1, 1, 16, 16),
            (8, 16),
            (8, 32),
            ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (80, 64),
            ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(0, 1))}),
        ),
    ),
)
def test_sharded_concat(input_shape, shard_shape, output_shard_shape, shard_grid, device):
    num_inputs = 2
    inputs = []
    tt_inputs = []
    input_shard_scheme = ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    input_shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, input_shard_orientation, False)
    sharded_mem_config = ttl.tensor.MemoryConfig(input_shard_scheme, ttl.tensor.BufferType.L1, shard_spec)

    output_shard_spec = ttl.tensor.ShardSpec(shard_grid, output_shard_shape, input_shard_orientation, False)
    output_sharded_mem_config = ttl.tensor.MemoryConfig(input_shard_scheme, ttl.tensor.BufferType.L1, output_shard_spec)

    total_elements = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    for i in range(num_inputs):
        shape = torch.Size(input_shape)
        inputs.append(i * 1000 * total_elements + torch.arange(0, shape.numel()).reshape(shape).to(torch.bfloat16))
        tt_inputs.append(
            ttl.tensor.Tensor(
                inputs[i],
                ttl.tensor.DataType.BFLOAT16,
            )
            .to(ttl.tensor.Layout.ROW_MAJOR)
            .to(device, sharded_mem_config)
        )

    dram_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    dim = 3
    tt_output_interleaved = ttl.tensor.concat(tt_inputs, dim, output_sharded_mem_config)

    tt_dev = tt_output_interleaved.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    tt_cpu = torch.concat(inputs, dim)

    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing
