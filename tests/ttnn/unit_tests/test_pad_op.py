# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from math import prod
import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [[1, 1, 18, 13]])
@pytest.mark.parametrize("padshape", [[1, 1, TILE_HEIGHT, TILE_WIDTH]])
@pytest.mark.parametrize("use_multicore", [False, True])
def test_pad_op(device, in_dtype, shape, padshape, use_multicore):
    torch_input = torch.randn(shape, dtype=torch.bfloat16).bfloat16()

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tt = ttnn.pad(ttnn_input, padshape, [0, 0, 0, 0], value=0, use_multicore=use_multicore)
    output_tt = ttnn.to_torch(output_tt)
    assert output_tt.shape == torch.Size(padshape)

    shape_diff = list(map(lambda x, y: x - y, padshape, shape))
    output_torch = torch.nn.functional.pad(torch_input, [0, shape_diff[-1], 0, shape_diff[-2]], value=0)
    assert_with_pcc(output_tt, output_torch, 0.9999)


def _unsqueeze(smaller, larger, fill):
    diff = len(larger) - len(smaller)
    return [fill] * diff + smaller


@pytest.mark.parametrize("shape", [[2, 8], [1, 2, 3, 4], [5, 4, 3, 2, 1]])
@pytest.mark.parametrize("padding", [[25, 1], [5, 4], [64], [32, 32], [1, 0, 0, 0], [1, 0, 0]])
def test_pad_tile(shape, padding, device):
    if shape == [5, 4, 3, 2, 1] and padding == [1, 0, 0, 0]:
        pytest.xfail("Can't pad upper dims with rank>4")

    if len(shape) < len(padding):
        shape = _unsqueeze(shape, padding, 1)
    elif len(padding) < len(shape):
        padding = _unsqueeze(padding, shape, 0)

    input = torch.ones(prod(shape), dtype=torch.bfloat16).reshape(shape)
    input_tensor = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    torch_padding = sum([[0, p] for p in reversed(padding)], [])
    torch_output = torch.nn.functional.pad(input, torch_padding, value=5)

    output = ttnn.pad(input_tensor, [(0, p) for p in padding], value=5)

    out_tt = ttnn.to_torch(output)

    assert_with_pcc(out_tt, torch_output, 0.9999)
