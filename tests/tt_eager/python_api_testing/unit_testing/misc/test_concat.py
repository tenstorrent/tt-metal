# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import ttnn
from models.utility_functions import print_diff_argmax
import pytest
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
    comp_equal,
)


def run_concat(shapes, dim, device, layout, dtype, input_mem_config, output_mem_config):
    if layout == ttnn.ROW_MAJOR_LAYOUT and dtype == ttnn.bfloat8_b:
        pytest.skip("Illegal config")
    if layout == ttnn.TILE_LAYOUT:
        for shape in shapes:
            if shape[-2] % 32 != 0 or shape[-1] % 32 != 0:
                pytest.skip("Illegal config")
    inputs = []
    tt_inputs = []
    for i in range(len(shapes)):
        shape = torch.Size(shapes[i])
        inputs.append(torch.rand(shape).to(torch.bfloat16))
        tt_inputs.append(
            ttnn.Tensor(
                inputs[i],
                dtype,
            )
            .to(layout)
            .to(device, input_mem_config)
        )

    tt_cpu = torch.concat(inputs, dim)

    tt = ttnn.concat(tt_inputs, dim, memory_config=output_mem_config)

    tt_dev = tt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)

    if dtype == ttnn.bfloat8_b:
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
        (ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ),
)
@pytest.mark.parametrize(
    "input_mem_config",
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    ),
)
@pytest.mark.parametrize(
    "output_mem_config",
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
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
        (ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ),
)
@pytest.mark.parametrize(
    "input_mem_config",
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    ),
)
@pytest.mark.parametrize(
    "output_mem_config",
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    ),
)
def test_concat_with_program_cache(
    shapes, dim, device, layout, dtype, input_mem_config, output_mem_config, use_program_cache, function_level_defaults
):
    run_concat(shapes, dim, device, layout, dtype, input_mem_config, output_mem_config)
    tmp = ttnn.empty([1, 256, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    run_concat(shapes, dim, device, layout, dtype, input_mem_config, output_mem_config)


@pytest.mark.parametrize(
    "input_shape, shard_shape, output_shard_shape, shard_grid",
    (
        (
            (1, 1, 16, 16),
            (8, 16),
            (8, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (80, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (80, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 235200, 16),
            (4800, 16),
            (4800, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 6))}),
        ),
    ),
)
def test_sharded_concat(input_shape, shard_shape, output_shard_shape, shard_grid, device):
    num_inputs = 2
    inputs = []
    tt_inputs = []
    input_shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    input_shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, input_shard_orientation, False)
    sharded_mem_config = ttnn.MemoryConfig(input_shard_scheme, ttnn.BufferType.L1, shard_spec)

    output_shard_spec = ttnn.ShardSpec(shard_grid, output_shard_shape, input_shard_orientation, False)
    output_sharded_mem_config = ttnn.MemoryConfig(input_shard_scheme, ttnn.BufferType.L1, output_shard_spec)

    total_elements = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    for i in range(num_inputs):
        shape = torch.Size(input_shape)
        inputs.append(i * 1000 * total_elements + torch.arange(0, shape.numel()).reshape(shape).to(torch.bfloat16))
        tt_inputs.append(
            ttnn.Tensor(
                inputs[i],
                ttnn.bfloat16,
            )
            .to(ttnn.ROW_MAJOR_LAYOUT)
            .to(device, sharded_mem_config)
        )

    dram_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    dim = 3
    tt_output_interleaved = ttnn.concat(tt_inputs, dim, memory_config=output_sharded_mem_config)

    tt_dev = tt_output_interleaved.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    tt_cpu = torch.concat(inputs, dim)
    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing
