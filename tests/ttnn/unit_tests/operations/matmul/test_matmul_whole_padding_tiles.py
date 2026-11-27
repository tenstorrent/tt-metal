# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""matmul and linear on TILE tensors whose padded shape has whole tiles past the tile-rounded logical
shape (https://github.com/tenstorrent/tt-metal/issues/65486). tilize_with_val_padding makes one when
asked for a larger padded shape. Each op must give bit-identical results to the same op on tensors
made from the same logical values with the default padding.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device

# (logical shape, padded shape) of A, whose inner dim is K = 96 or 70: extra tile columns, extra tile
# rows, both with a partial last tile, padding in every tile dim of a rank-4 batch, rank 3, padding in
# an outer dim, and rank 5.
_A_SHAPES = [
    ([1, 1, 64, 96], [1, 1, 64, 128]),
    ([1, 1, 64, 96], [1, 1, 128, 96]),
    ([1, 1, 50, 70], [1, 1, 96, 128]),
    ([2, 3, 64, 96], [2, 3, 96, 128]),
    ([3, 40, 96], [3, 64, 128]),
    ([1, 2, 64, 96], [2, 2, 64, 96]),
    ([2, 2, 1, 64, 96], [2, 2, 1, 64, 128]),
]

_PAD_VALUE = 7.0


def _padded(device, torch_tensor, padded_shape):
    row_major = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    padded = ttnn.tilize_with_val_padding(row_major, ttnn.Shape(padded_shape), _PAD_VALUE)
    assert list(padded.padded_shape) == padded_shape
    return padded


def _plain(device, torch_tensor):
    return ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("logical_shape, padded_shape", _A_SHAPES, ids=[str(p) for _, p in _A_SHAPES])
def test_matmul_a_with_whole_padding_tiles(device, logical_shape, padded_shape):
    torch.manual_seed(0)
    torch_a = torch.randn(logical_shape).bfloat16()
    torch_b = torch.randn([logical_shape[-1], 64]).bfloat16()
    b = _plain(device, torch_b)

    expected = ttnn.to_torch(ttnn.matmul(_plain(device, torch_a), b))
    result = ttnn.to_torch(ttnn.matmul(_padded(device, torch_a, padded_shape), b))
    assert_equal(expected, result)


@pytest.mark.parametrize(
    "b_logical_shape, b_padded_shape",
    [([96, 64], [96, 128]), ([96, 64], [128, 64]), ([70, 50], [128, 96]), ([1, 1, 96, 64], [1, 2, 96, 64])],
    ids=["cols", "rows", "both_partial", "outer"],
)
def test_matmul_b_with_whole_padding_tiles(device, b_logical_shape, b_padded_shape):
    torch.manual_seed(0)
    torch_a = torch.randn([1, 1, 64, b_logical_shape[-2]]).bfloat16()
    torch_b = torch.randn(b_logical_shape).bfloat16()
    a = _plain(device, torch_a)

    expected = ttnn.to_torch(ttnn.matmul(a, _plain(device, torch_b)))
    result = ttnn.to_torch(ttnn.matmul(a, _padded(device, torch_b, b_padded_shape)))
    assert_equal(expected, result)


def test_linear_with_whole_padding_tiles(device):
    torch.manual_seed(0)
    torch_a = torch.randn([1, 1, 64, 96]).bfloat16()
    torch_b = torch.randn([96, 64]).bfloat16()
    torch_bias = torch.randn([1, 64]).bfloat16()

    expected = ttnn.to_torch(
        ttnn.linear(_plain(device, torch_a), _plain(device, torch_b), bias=_plain(device, torch_bias))
    )
    result = ttnn.to_torch(
        ttnn.linear(
            _padded(device, torch_a, [1, 1, 128, 128]),
            _padded(device, torch_b, [128, 128]),
            bias=_padded(device, torch_bias, [64, 128]),
        )
    )
    assert_equal(expected, result)
