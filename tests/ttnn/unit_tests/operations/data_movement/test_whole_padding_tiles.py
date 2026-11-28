# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ops on TILE tensors whose padded shape has whole tiles past the tile-rounded logical shape
(https://github.com/tenstorrent/tt-metal/issues/65486). tilize_with_val_padding makes one when
asked for a larger padded shape. Each op must give bit-identical results to the same op on a
tensor made from the same logical values with the default padding. The eltwise ops are covered in
tests/ttnn/unit_tests/operations/eltwise/test_eltwise_whole_padding_tiles.py.
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


def _copy_into(t):
    # Same padded shape as t, different values, so only a copy that writes every page matches.
    output = ttnn.neg(t)
    ttnn.copy(t, output)
    return output


_OPS = {
    "to_l1": lambda t: ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG),
    "to_l1_and_back": lambda t: ttnn.to_memory_config(
        ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG), ttnn.DRAM_MEMORY_CONFIG
    ),
    "copy": _copy_into,
    "reshape_same": lambda t: ttnn.reshape(t, t.shape),
    "reshape_flatten": lambda t: ttnn.reshape(t, [1, 1, -1, list(t.shape)[-1]]),
    "reshape_split_last": lambda t: ttnn.reshape(t, list(t.shape)[:-1] + [2, list(t.shape)[-1] // 2]),
}


@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("logical_shape, padded_shape", _SHAPES, ids=[str(p) for _, p in _SHAPES])
def test_whole_padding_tiles(device, op, logical_shape, padded_shape):
    torch.manual_seed(0)
    torch_input = torch.randn(logical_shape).bfloat16()
    row_major = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    padded = ttnn.tilize_with_val_padding(row_major, ttnn.Shape(padded_shape), _PAD_VALUE)
    assert list(padded.padded_shape) == padded_shape
    plain = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_OPS[op](plain))
    result = ttnn.to_torch(_OPS[op](padded))
    assert_equal(expected, result)


def test_copy_rejects_output_with_other_padded_shape(device, expect_error):
    """Interleaved tiles are copied in order, so a preallocated output has to share the input's padded shape."""
    torch_input = torch.randn([1, 1, 64, 96]).bfloat16()
    row_major = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    padded = ttnn.tilize_with_val_padding(row_major, ttnn.Shape([1, 1, 64, 128]), _PAD_VALUE)
    plain = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must equal the input tensor's padded shape"):
        ttnn.copy(padded, plain)


def _whole_padding_tiles(device, torch_input, padded_shape, dtype=ttnn.bfloat16):
    row_major = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    padded = ttnn.tilize_with_val_padding(row_major, ttnn.Shape(padded_shape), _PAD_VALUE, dtype=dtype)
    assert list(padded.padded_shape) == padded_shape
    return padded


def test_reshape_block_float4(device):
    """bfloat4_b cannot be row major, so rebuilding its tile grid goes through bfloat16."""
    torch.manual_seed(0)
    torch_input = torch.randn([1, 1, 64, 96]).bfloat16()
    padded = _whole_padding_tiles(device, torch_input, [1, 1, 64, 128], ttnn.bfloat4_b)

    result = ttnn.reshape(padded, [1, 1, 32, 192])
    assert result.dtype == ttnn.bfloat4_b
    # Device and host quantize to bfloat4_b differently, so the reference is the tensor's own values.
    assert_equal(ttnn.to_torch(padded).reshape([1, 1, 32, 192]), ttnn.to_torch(result))


def test_reshape_keeps_requested_padding(device):
    """An explicit output padded shape that keeps the padding tiles stays a view of the same pages."""
    torch.manual_seed(0)
    torch_input = torch.randn([1, 2, 64, 96]).bfloat16()
    padded = _whole_padding_tiles(device, torch_input, [1, 2, 64, 128])

    result = ttnn.reshape(padded, ttnn.Shape([2, 1, 64, 96]), ttnn.Shape([2, 1, 64, 128]))
    assert list(result.padded_shape) == [2, 1, 64, 128]
    assert_equal(torch_input.reshape([2, 1, 64, 96]), ttnn.to_torch(result))


@pytest.mark.parametrize("logical_shape, new_shape", [([], [1]), ([], [1, 1]), ([2], [1, 2]), ([1, 2], [2])], ids=str)
def test_reshape_rank_0_and_1(device, logical_shape, new_shape):
    """TILE lays out ranks 0 and 1 as one 32x32 tile, which is their default padding, not whole padding tiles."""
    torch_input = torch.randn(logical_shape).bfloat16()
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.reshape(tt_input, new_shape)
    assert list(result.shape) == new_shape
    assert_equal(torch_input.reshape(new_shape), ttnn.to_torch(result))
