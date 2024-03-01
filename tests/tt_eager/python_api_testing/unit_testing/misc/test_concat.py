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


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B)))
@pytest.mark.parametrize("nChannels", ((2, 3, 4)))
def test_tile_simple_concat(memcfg, dtype, nChannels, device, function_level_defaults):
    input_shape = torch.Size([nChannels, nChannels, 32, 32])
    x = torch.arange(0, input_shape.numel()).reshape(input_shape).bfloat16()

    y = (1 + torch.arange(0, input_shape.numel()).reshape(input_shape)).bfloat16()

    xtt = (
        ttl.tensor.Tensor(x, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(y, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    dim = 3
    output_shape = list(x.shape)
    output_shape[3] = y.shape[3] + x.shape[3]
    tt_cpu = torch.concat([x, y], dim)
    assert tt_cpu.shape == torch.Size(output_shape)

    tt = ttl.tensor.concat(xtt, dim)
    assert list(tt.shape()) == output_shape
    xtt_data = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    tt_dev = xtt_data.to_torch()

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    else:
        passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


# @pytest.mark.skip(reason="For Stable Diffusion Sizes only")
@pytest.mark.parametrize(
    "shape_a, shape_b, dim",
    (
        ((1, 1, 32, 32), (1, 1, 32, 32), 3),
        ((1, 1, 32, 64), (1, 1, 32, 128), 3),
        ((1, 1, 32, 128), (1, 1, 32, 64), 3),
        ((1, 1, 64, 128), (1, 1, 64, 256), 3),
        ((1, 32, 32, 32), (1, 32, 32, 32), 2),
        ((1, 1, 32, 32), (1, 1, 32, 32), 3),
        ((2, 4, 32, 1280), (2, 4, 32, 1280), 3),
        # SD Shapes
        ((2, 1280, 4, 4), (2, 1280, 4, 4), 1),
        ((2, 640, 32, 32), (2, 320, 32, 32), 1),
        ((2, 1280, 8, 8), (2, 1280, 8, 8), 1),
        ((2, 640, 16, 16), (2, 640, 16, 16), 1),
        ((2, 320, 32, 32), (2, 320, 32, 32), 1),
    ),
)
def test_tile_concat(shape_a, shape_b, dim, device, function_level_defaults):
    shape_a = torch.Size(shape_a)

    x = torch.arange(0, shape_a.numel()).reshape(shape_a).to(torch.bfloat16)

    shape_b = torch.Size(shape_b)
    y = torch.arange(0, shape_b.numel()).reshape(shape_b).to(torch.bfloat16)

    xtt = (
        ttl.tensor.Tensor(
            x,
            ttl.tensor.DataType.BFLOAT16,
        ).to(device),
        ttl.tensor.Tensor(
            y,
            ttl.tensor.DataType.BFLOAT16,
        ).to(device),
    )

    output_shape = list(x.shape)
    output_shape[dim] = y.shape[dim] + x.shape[dim]
    tt_cpu = torch.concat([x, y], dim)
    assert tt_cpu.shape == torch.Size(output_shape)

    tt = ttl.tensor.concat([xtt[0], xtt[1]], dim)
    assert list(tt.shape()) == output_shape
    tt_dev = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

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
    ),
)
def test_multi_input_concat(shapes, dim, device, function_level_defaults):
    inputs = []
    tt_inputs = []
    for i in range(len(shapes)):
        shape = torch.Size(shapes[i])
        inputs.append(i + torch.arange(0, shape.numel()).reshape(shape).to(torch.bfloat16))
        tt_inputs.append(
            ttl.tensor.Tensor(
                inputs[i],
                ttl.tensor.DataType.BFLOAT16,
            ).to(device)
        )

    tt_cpu = torch.concat(inputs, dim)

    tt = ttl.tensor.concat(tt_inputs, dim)

    tt_dev = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing
