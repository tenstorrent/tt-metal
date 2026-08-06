# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host property tests for Hunyuan VAE tile blending and stitching."""

import pytest
import torch

from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_decoder import TTVAEDecodeAdapter


def _make_odd_edge_grid(seed):
    generator = torch.Generator().manual_seed(seed)
    heights = (11, 11, 5)
    widths = (13, 13, 7, 3)
    decoded = []
    coords = []
    for row, height in enumerate(heights):
        for col, width in enumerate(widths):
            decoded.append(torch.randn(1, 3, 5, height, width, generator=generator))
            coords.append((row * 8, col * 10, height, width))
    return decoded, coords, len(widths)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("blend_h,blend_w", [(0, 0), (1, 1), (4, 5), (32, 32)])
def test_vectorized_tile_stitch_matches_legacy(seed, blend_h, blend_w):
    """Match every stitched pixel, including tile boundaries and odd edge tiles."""
    decoded, coords, ncol = _make_odd_edge_grid(seed)
    legacy = TTVAEDecodeAdapter._stitch_tiles(
        [tile.clone() for tile in decoded],
        coords,
        ncol,
        blend_h,
        blend_w,
        row_limit_h=8,
        row_limit_w=10,
        legacy=True,
    )
    vectorized = TTVAEDecodeAdapter._stitch_tiles(
        [tile.clone() for tile in decoded],
        coords,
        ncol,
        blend_h,
        blend_w,
        row_limit_h=8,
        row_limit_w=10,
        legacy=False,
    )

    torch.testing.assert_close(vectorized, legacy, rtol=0, atol=0)


def test_blend_boundary_weights_are_unchanged():
    """The first overlap pixel is all prior tile; the last retains 1/N prior weight."""
    extent = 5
    prior = torch.full((1, 1, 1, 3, extent), 2.0)
    current = torch.full_like(prior, 12.0)

    horizontal = TTVAEDecodeAdapter._blend_h_vectorized(prior, current, extent)
    expected = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    torch.testing.assert_close(horizontal[0, 0, 0, 0], expected, rtol=0, atol=0)

    vertical = TTVAEDecodeAdapter._blend_v_vectorized(prior.transpose(-1, -2), current.transpose(-1, -2), extent)
    torch.testing.assert_close(vertical[0, 0, 0, :, 0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("ndev,n_total", [(1, 5), (2, 5), (4, 9), (8, 3)])
def test_device_major_rounds_restore_original_tile_order(ndev, n_total):
    rounds = (n_total + ndev - 1) // ndev
    padded = rounds * ndev
    round_major = torch.arange(padded)
    device_major = round_major.reshape(rounds, ndev).transpose(0, 1).reshape(padded)

    restored = TTVAEDecodeAdapter._restore_round_major(device_major, ndev, rounds, n_total)

    torch.testing.assert_close(restored, torch.arange(n_total), rtol=0, atol=0)
