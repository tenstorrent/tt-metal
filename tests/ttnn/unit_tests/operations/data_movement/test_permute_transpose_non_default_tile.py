# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The permute and transpose kernels address 32x32 tiles only, so a TILE input with another tile shape used to give
wrong values without an error (#63679). Both now reject it; a no-op and a 32x32 tile still work."""

import pytest
import torch
import ttnn

TILES = [[16, 32], [32, 16], [8, 32]]


def _input(device, tile):
    torch.manual_seed(0)
    x = (torch.randint(-64, 65, (2, 2, 64, 128)) / 8).to(torch.bfloat16)
    return x, ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile(tile), device=device)


@pytest.mark.parametrize("tile", TILES, ids=lambda t: f"{t[0]}x{t[1]}")
@pytest.mark.parametrize("dims", [(1, 0, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (3, 1, 2, 0), (2, 3, 0, 1)])
def test_permute_rejects_non_default_tile(device, dims, tile, expect_error):
    _, x = _input(device, tile)
    with expect_error(RuntimeError, f"ttnn.permute supports only 32x32 tiles, but the input has a {tile[0]}x{tile[1]}"):
        ttnn.permute(x, dims)


@pytest.mark.parametrize("tile", TILES, ids=lambda t: f"{t[0]}x{t[1]}")
@pytest.mark.parametrize("dims", [(-2, -1), (0, 1), (1, 2), (0, 3)])
def test_transpose_rejects_non_default_tile(device, dims, tile, expect_error):
    _, x = _input(device, tile)
    with expect_error(RuntimeError, f"supports only 32x32 tiles, but the input has a {tile[0]}x{tile[1]} tile"):
        ttnn.transpose(x, *dims)


@pytest.mark.parametrize("tile", TILES, ids=lambda t: f"{t[0]}x{t[1]}")
def test_permute_and_transpose_nop_keep_non_default_tile(device, tile):
    x_host, x = _input(device, tile)
    assert torch.equal(ttnn.to_torch(ttnn.permute(x, (0, 1, 2, 3))), x_host)
    assert torch.equal(ttnn.to_torch(ttnn.transpose(x, 1, 1)), x_host)


@pytest.mark.parametrize("tile", TILES, ids=lambda t: f"{t[0]}x{t[1]}")
def test_singleton_axis_swap_keeps_non_default_tile(device, tile):
    # Swapping two size-1 dims moves no data, so neither op runs a kernel.
    x_host = (torch.randint(-64, 65, (1, 1, 64, 128)) / 8).to(torch.bfloat16)
    x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile(tile), device=device)
    assert torch.equal(ttnn.to_torch(ttnn.transpose(x, 0, 1)), x_host)
    assert torch.equal(ttnn.to_torch(ttnn.permute(x, (1, 0, 2, 3))), x_host)


@pytest.mark.parametrize("dims", [(1, 0, 2, 3), (0, 1, 3, 2), (3, 1, 2, 0)])
def test_permute_default_tile(device, dims):
    x_host, x = _input(device, [32, 32])
    assert torch.equal(ttnn.to_torch(ttnn.permute(x, dims)), x_host.permute(*dims))
