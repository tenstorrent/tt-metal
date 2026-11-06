# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.untilize_and_split_heads.

(B, S, H*D) TILE -> (B, H, ceil(S/32), 32*D) ROW_MAJOR, each page one tile row of a head.
"""

import pytest
import torch

import ttnn

ROWS_PER_PAGE = 32  # the tile height


def _reference(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, S, H*D) -> (B, H, ceil(S/32), 32*D). Rows past S are zero here, but the op leaves them
    unspecified, so callers compare rows [:S] only (see _assert_rows_match)."""
    B, S, HD = x.shape
    D = HD // num_heads
    pages = -(-S // ROWS_PER_PAGE)
    out = torch.zeros(B, num_heads, pages * ROWS_PER_PAGE, D, dtype=x.dtype)
    out[:, :, :S] = x.reshape(B, S, num_heads, D).permute(0, 2, 1, 3)
    return out.reshape(B, num_heads, pages, ROWS_PER_PAGE * D)


def _assert_rows_match(out_t: ttnn.Tensor, x: torch.Tensor, num_heads: int):
    B, S, HD = x.shape
    D = HD // num_heads
    out = ttnn.to_torch(out_t)
    want = _reference(x, num_heads)
    assert list(out.shape) == list(want.shape)
    rows = out.reshape(B, num_heads, -1, D)[:, :, :S]
    torch.testing.assert_close(rows, want.reshape(B, num_heads, -1, D)[:, :, :S], rtol=0, atol=0)


def _tiled(x: torch.Tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs) -> ttnn.Tensor:
    return ttnn.from_torch(
        x.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, **kwargs
    )


@pytest.mark.parametrize("in_memory", ["dram", "l1"])
@pytest.mark.parametrize("out_memory", ["dram", "l1"])
@pytest.mark.parametrize("D", [32, 64])
@pytest.mark.parametrize("H", [1, 8])
@pytest.mark.parametrize("B,S", [(1, 1), (1, 32), (2, 33), (2, 332)])
def test_untilize_and_split_heads(device, B, S, H, D, out_memory, in_memory):
    torch.manual_seed(0)
    x = torch.randn(B, S, H * D).to(torch.bfloat16)
    in_config = ttnn.L1_MEMORY_CONFIG if in_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
    out_config = ttnn.L1_MEMORY_CONFIG if out_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG

    out_t = ttnn.experimental.untilize_and_split_heads(_tiled(x, device, in_config), H, memory_config=out_config)
    assert out_t.layout == ttnn.ROW_MAJOR_LAYOUT
    assert out_t.memory_config().buffer_type == out_config.buffer_type
    _assert_rows_match(out_t, x, H)


@pytest.mark.parametrize("out_memory", ["dram", "l1"])
# 3, 4 and 9 tiles per page. Blackhole takes the fast untilize at 2+ tiles;
# elsewhere 9 tiles exceed the 8-tile DEST limit and untilize in sub-blocks.
@pytest.mark.parametrize("D", [96, 128, 288])
@pytest.mark.parametrize("H", [1, 2])
def test_untilize_and_split_heads_wide_heads(device, H, D, out_memory):
    torch.manual_seed(0)
    x = torch.randn(2, 33, H * D).to(torch.bfloat16)
    out_config = ttnn.L1_MEMORY_CONFIG if out_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
    _assert_rows_match(ttnn.experimental.untilize_and_split_heads(_tiled(x, device), H, memory_config=out_config), x, H)


def test_untilize_and_split_heads_program_cache(device):
    """Calls on fresh buffers reuse one program; a different S or head split compiles its own.

    Within a round both inputs stay alive, so consecutive calls alternate between
    two addresses (asserted) and every cache hit has to rebind them.
    """
    torch.manual_seed(0)
    x_a = torch.randn(2, 40, 64).to(torch.bfloat16)
    x_b = torch.randn(2, 40, 64).to(torch.bfloat16)
    entries_before = device.num_program_cache_entries()
    for _ in range(3):
        in_a, in_b = _tiled(x_a, device), _tiled(x_b, device)
        out_a = ttnn.experimental.untilize_and_split_heads(in_a, 2)
        out_b = ttnn.experimental.untilize_and_split_heads(in_b, 2)
        assert in_a.buffer_address() != in_b.buffer_address()
        assert out_a.buffer_address() != out_b.buffer_address()
        _assert_rows_match(out_a, x_a, 2)
        _assert_rows_match(out_b, x_b, 2)
        for t in (out_a, in_a, out_b, in_b):
            ttnn.deallocate(t)
    assert device.num_program_cache_entries() - entries_before == 1

    # The same H*D as one head instead of two.
    _assert_rows_match(ttnn.experimental.untilize_and_split_heads(_tiled(x_a, device), 1), x_a, 1)
    assert device.num_program_cache_entries() - entries_before == 2

    # Three tile rows per head instead of two.
    x_c = torch.randn(2, 72, 64).to(torch.bfloat16)
    _assert_rows_match(ttnn.experimental.untilize_and_split_heads(_tiled(x_c, device), 2), x_c, 2)
    assert device.num_program_cache_entries() - entries_before == 3


def test_untilize_and_split_heads_default_memory_config(device):
    """Without memory_config the output lands where the input lives."""
    x = _tiled(torch.randn(1, 40, 64), device, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.experimental.untilize_and_split_heads(x, 2)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1


def test_untilize_and_split_heads_rejects_row_major(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 64, 64).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(RuntimeError, "must be TILE"):
        ttnn.experimental.untilize_and_split_heads(x, 2)


def test_untilize_and_split_heads_rejects_float32(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 64, 64), dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "BFLOAT16"):
        ttnn.experimental.untilize_and_split_heads(x, 2)


def test_untilize_and_split_heads_rejects_non_32x32_tiles(device, expect_error):
    x = _tiled(torch.randn(1, 64, 64), device, tile=ttnn.Tile((16, 32)))
    with expect_error(RuntimeError, "32x32"):
        ttnn.experimental.untilize_and_split_heads(x, 2)


def test_untilize_and_split_heads_rejects_transposed_tiles(device, expect_error):
    x = _tiled(torch.randn(1, 64, 64), device, tile=ttnn.Tile((32, 32), transpose_tile=True))
    with expect_error(RuntimeError, "got 32x32 transposed"):
        ttnn.experimental.untilize_and_split_heads(x, 2)


def test_untilize_and_split_heads_rejects_rank4_input(device, expect_error):
    x = _tiled(torch.randn(1, 2, 64, 64), device)
    with expect_error(RuntimeError, r"must be \(B, S, H\*D\)"):
        ttnn.experimental.untilize_and_split_heads(x, 2)


def test_untilize_and_split_heads_rejects_zero_heads(device, expect_error):
    x = _tiled(torch.randn(1, 64, 64), device)
    with expect_error(RuntimeError, "num_heads must be > 0"):
        ttnn.experimental.untilize_and_split_heads(x, 0)


@pytest.mark.parametrize("padded_shape", [[2, 96, 64], [2, 64, 96]], ids=["rows", "channels"])
def test_untilize_and_split_heads_rejects_extra_tile_padding(device, expect_error, padded_shape):
    """Tile ids come from the logical shape, so padding past the last tile, in S or in H*D, would read other tiles."""
    row_major = ttnn.from_torch(torch.randn(2, 40, 64).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    padded = ttnn.tilize_with_val_padding(row_major, padded_shape, 0.0)
    assert list(padded.shape) == [2, 40, 64] and list(padded.padded_shape) == padded_shape
    with expect_error(RuntimeError, "only padding S up to a whole tile"):
        ttnn.experimental.untilize_and_split_heads(padded, 2)


def test_untilize_and_split_heads_rejects_head_count_that_overflows_uint32(device, expect_error):
    # 2**27 heads * 32 wraps a uint32 to 0; the check must still reject, not divide by zero.
    x = _tiled(torch.randn(1, 64, 64), device)
    with expect_error(RuntimeError, "whole number of tile widths"):
        ttnn.experimental.untilize_and_split_heads(x, 2**27)


def test_untilize_and_split_heads_rejects_heads_narrower_than_a_tile(device, expect_error):
    # H*D = 128 over 8 heads is D = 16: a head would share its tiles with the next.
    x = _tiled(torch.randn(1, 64, 128), device)
    with expect_error(RuntimeError, "whole number of tile widths"):
        ttnn.experimental.untilize_and_split_heads(x, 8)
