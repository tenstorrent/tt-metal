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


def run_repeat(input_shape, repeats, device, layout, dtype, input_mem_config, output_mem_config):
    if layout == ttl.tensor.Layout.ROW_MAJOR and dtype == ttl.tensor.DataType.BFLOAT8_B:
        pytest.skip("Illegal config")
    if layout == ttl.tensor.Layout.TILE:
        if input_shape[-2] % 32 != 0 or input_shape[-1] % 32 != 0:
            pytest.skip("Illegal config")
    input = torch.rand(input_shape).to(torch.bfloat16)
    tt_input = (
        ttl.tensor.Tensor(
            input,
            dtype,
        )
        .to(layout)
        .to(device, input_mem_config)
    )

    tt_cpu = input.repeat(torch.Size(repeats))

    tt = ttl.tensor.repeat(tt_input, ttl.tensor.Shape(repeats), output_mem_config)

    tt_dev = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
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
    tmp = ttl.tensor.empty([1, 256, 32, 32], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    run_repeat(input_shape, repeats, device, layout, dtype, input_mem_config, output_mem_config)
