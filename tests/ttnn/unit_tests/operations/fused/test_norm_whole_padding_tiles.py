# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Softmax and normalization ops on TILE tensors whose padded shape has whole tiles past the tile-rounded
logical shape (https://github.com/tenstorrent/tt-metal/issues/65486). tilize_with_val_padding makes one
when asked for a larger padded shape. Each op must give bit-identical results to the same op on a
tensor made from the same logical values with the default padding.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device

# (logical shape, padded shape): extra tile columns, extra tile rows, both with a partial last
# tile, padding in every tile dim of a rank-4 batch, rank 3, padding in an outer dim, and rank 5.
_SHAPES = [
    ([1, 1, 64, 96], [1, 1, 64, 128]),
    ([1, 1, 64, 96], [1, 1, 128, 96]),
    ([1, 1, 50, 70], [1, 1, 96, 128]),
    ([2, 3, 64, 64], [2, 3, 96, 96]),
    ([3, 40, 72], [3, 64, 128]),
    ([1, 2, 64, 64], [2, 2, 64, 64]),
    ([2, 2, 1, 64, 96], [2, 2, 1, 64, 128]),
]

_PAD_VALUE = 7.0


def _padded(device, torch_tensor, padded_shape):
    row_major = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    padded = ttnn.tilize_with_val_padding(row_major, ttnn.Shape(padded_shape), _PAD_VALUE)
    assert list(padded.padded_shape) == padded_shape
    return padded


_OPS = {
    "softmax_w": lambda t: ttnn.softmax(t, dim=-1),
    "softmax_h": lambda t: ttnn.softmax(t, dim=-2),
    "scale_mask_softmax": lambda t: ttnn.scale_mask_softmax(t),
    "layer_norm": lambda t: ttnn.layer_norm(t),
    "rms_norm": lambda t: ttnn.rms_norm(t),
}


@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("logical_shape, padded_shape", _SHAPES, ids=[str(p) for _, p in _SHAPES])
def test_whole_padding_tiles(device, op, logical_shape, padded_shape):
    if op == "scale_mask_softmax" and len(logical_shape) > 4:
        pytest.skip("scale_mask_softmax supports up to rank 4")
    torch.manual_seed(0)
    torch_input = torch.randn(logical_shape).bfloat16()
    plain = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_OPS[op](plain))
    result = ttnn.to_torch(_OPS[op](_padded(device, torch_input, padded_shape)))
    assert_equal(expected, result)


@pytest.mark.parametrize("norm", [ttnn.layer_norm, ttnn.rms_norm], ids=["layer_norm", "rms_norm"])
def test_norm_residual_and_weight_with_whole_padding_tiles(device, norm):
    torch.manual_seed(0)
    logical_shape, padded_shape = [1, 1, 64, 96], [1, 1, 128, 128]
    torch_input = torch.randn(logical_shape).bfloat16()
    torch_residual = torch.randn(logical_shape).bfloat16()
    torch_weight = torch.randn([1, 1, 1, 96]).bfloat16()

    def run(make):
        return ttnn.to_torch(
            norm(
                make(torch_input, padded_shape),
                residual_input_tensor=make(torch_residual, padded_shape),
                weight=make(torch_weight, [1, 1, 64, 128]),
            )
        )

    expected = run(lambda t, _: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device))
    result = run(lambda t, shape: _padded(device, t, shape))
    assert_equal(expected, result)


def test_scale_mask_softmax_mask_with_whole_padding_tiles(device):
    torch.manual_seed(0)
    torch_input = torch.randn([2, 1, 64, 96]).bfloat16()
    torch_mask = torch.where(torch.rand([2, 1, 1, 96]) < 0.3, -1e4, 0.0).bfloat16()

    def run(make):
        return ttnn.to_torch(
            ttnn.scale_mask_softmax(
                make(torch_input, [2, 1, 128, 128]), scale=0.5, mask=make(torch_mask, [2, 1, 32, 128])
            )
        )

    expected = run(lambda t, _: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device))
    result = run(lambda t, shape: _padded(device, t, shape))
    assert_equal(expected, result)
