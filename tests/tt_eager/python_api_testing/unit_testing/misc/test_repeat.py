# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import ttnn.deprecated as ttl
import ttnn
from models.utility_functions import print_diff_argmax
import pytest
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
    comp_equal,
)


def run_repeat(input_shape, repeats, device, layout, dtype, input_mem_config, output_mem_config):
    if layout == ttnn.experimental.tensor.Layout.ROW_MAJOR and dtype == ttnn.experimental.tensor.DataType.BFLOAT8_B:
        pytest.skip("Illegal config")
    if layout == ttnn.experimental.tensor.Layout.TILE:
        if input_shape[-2] % 32 != 0 or input_shape[-1] % 32 != 0:
            pytest.skip("Illegal config")
    input = torch.rand(input_shape).to(torch.bfloat16)
    tt_input = (
        ttnn.experimental.tensor.Tensor(
            input,
            dtype,
        )
        .to(layout)
        .to(device, input_mem_config)
    )

    tt_cpu = input.repeat(torch.Size(repeats))

    tt = ttnn.repeat(tt_input, ttnn.Shape(repeats), memory_config=output_mem_config)

    tt_dev = tt.cpu().to(ttnn.experimental.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    if dtype == ttnn.experimental.tensor.DataType.BFLOAT8_B:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    else:
        passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "input_shape, repeats",
    (
        ((1, 2, 64, 64), [1, 1, 1, 1]),
        ((1, 1, 64, 64), [1, 1, 1, 2]),
        ((1, 1, 32, 128), [5, 3, 4, 2]),
        ((2, 4, 32, 1280), [3, 1, 1, 5]),
        ((1, 1, 32, 16), [1, 1, 1, 2048]),
    ),
)
@pytest.mark.parametrize(
    "layout, dtype",
    (
        (ttnn.experimental.tensor.Layout.TILE, ttnn.experimental.tensor.DataType.BFLOAT16),
        (ttnn.experimental.tensor.Layout.TILE, ttnn.experimental.tensor.DataType.BFLOAT8_B),
        (ttnn.experimental.tensor.Layout.ROW_MAJOR, ttnn.experimental.tensor.DataType.BFLOAT16),
    ),
)
@pytest.mark.parametrize(
    "input_mem_config",
    (
        ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.DRAM,
        ),
        ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        ),
    ),
)
@pytest.mark.parametrize(
    "output_mem_config",
    (
        ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.DRAM,
        ),
        ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        ),
    ),
)
def test_repeat(
    input_shape, repeats, device, layout, dtype, input_mem_config, output_mem_config, function_level_defaults
):
    run_repeat(input_shape, repeats, device, layout, dtype, input_mem_config, output_mem_config)


@pytest.mark.parametrize(
    "input_shape, repeats",
    (
        ((1, 2, 64, 64), [1, 1, 1, 1]),
        ((1, 1, 64, 64), [1, 1, 1, 2]),
        ((1, 1, 32, 128), [5, 3, 4, 2]),
        ((2, 4, 32, 1280), [3, 1, 1, 5]),
        ((1, 1, 32, 16), [1, 1, 1, 2048]),
    ),
)
@pytest.mark.parametrize(
    "layout, dtype",
    (
        (ttnn.experimental.tensor.Layout.TILE, ttnn.experimental.tensor.DataType.BFLOAT16),
        (ttnn.experimental.tensor.Layout.TILE, ttnn.experimental.tensor.DataType.BFLOAT8_B),
        (ttnn.experimental.tensor.Layout.ROW_MAJOR, ttnn.experimental.tensor.DataType.BFLOAT16),
    ),
)
@pytest.mark.parametrize(
    "input_mem_config",
    (
        ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.DRAM,
        ),
        ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        ),
    ),
)
@pytest.mark.parametrize(
    "output_mem_config",
    (
        ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.DRAM,
        ),
        ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        ),
    ),
)
def test_repeat_with_program_cache(
    input_shape,
    repeats,
    device,
    layout,
    dtype,
    input_mem_config,
    output_mem_config,
    use_program_cache,
    function_level_defaults,
):
    run_repeat(input_shape, repeats, device, layout, dtype, input_mem_config, output_mem_config)
    tmp = ttnn.empty(
        [1, 256, 32, 32], ttnn.experimental.tensor.DataType.BFLOAT16, ttnn.experimental.tensor.Layout.TILE, device
    )
    run_repeat(input_shape, repeats, device, layout, dtype, input_mem_config, output_mem_config)
