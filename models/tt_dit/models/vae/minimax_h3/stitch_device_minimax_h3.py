# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The video VAE's tile blend and unpatchify, on device.

Why this exists: the host path reads *overlapping* pixel tiles back and blends them on host. The
assembled canvas is far smaller than the tiles that produce it, so most of that transfer is overlap
that gets averaged away.

Why it is not simply a weighted accumulation: the reference `stitch_tiles` is **sequential and
asymmetric**. For an interior tile the corner region is `b*L + (1-b)*(a*A + (1-a)*T)` where `L` is the
*unblended* left tile and the diagonal tile never appears. A separable ramp formulation gives an
O(1) error over roughly a ninth of every frame, which surfaces as visible seams. So this mirrors the
reference order exactly, tile by tile, rather than reformulating it.

The blend runs in **float32** on device even though the decoder emits bfloat16, because the host path
it replaces blends in float32 (`.float()` before `stitch_tiles`). Keeping the same precision is what
lets the existing PCC and roundtrip-PSNR gates carry over unchanged.
"""

from __future__ import annotations

import torch

import ttnn


class DeviceTileStitcher:
    """Blends a grid of decoded tiles into one clip, on device, caching its ramp weights.

    The weights depend only on `(extent, axis, shape)`, and a decode reuses the same geometry for
    every chunk, so they are built once and reused for the whole video.

    Everything here stays ROW_MAJOR. This grid's derived overlaps are [96, 80, 80] by height and
    [80, 80, 80, 80, 64, 64] by width, so the blend slices begin at 176 and 80 -- neither a multiple
    of 32 -- and `ttnn.slice` drops to untilize -> row-major -> retilize for exactly that case. The
    trims then hand `ttnn.concat` extents of 80 and 176, which is tile padding on the concat dim and
    triggers the same fallback again. `binary_ng` takes ROW_MAJOR operands and keeps the layout on
    output, so the arithmetic is unaffected and the blend stays float32.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice) -> None:
        self.mesh_device = mesh_device
        self._ramps: dict[tuple, tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def _ramp_pair(self, shape: tuple[int, ...], extent: int, dim: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """`(weight_a, weight_b)` broadcast to `shape`, where `weight_a = 1 - i/extent` along `dim`.

        Materialized at full tile shape rather than relying on ttnn broadcast semantics: a few MB,
        built once, and it removes any question of which operand broadcasts.
        """
        key = (shape, extent, dim)
        if key not in self._ramps:
            positions = torch.arange(extent, dtype=torch.float32)
            view = [1] * len(shape)
            view[dim] = extent
            slab = list(shape)
            slab[dim] = extent
            weight_a = (1 - positions / extent).view(view).expand(slab).contiguous()
            weight_b = (positions / extent).view(view).expand(slab).contiguous()
            self._ramps[key] = tuple(
                ttnn.from_torch(w, dtype=ttnn.float32, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
                for w in (weight_a, weight_b)
            )
        return self._ramps[key]

    @staticmethod
    def _slice(tensor: ttnn.Tensor, dim: int, start: int, stop: int) -> ttnn.Tensor:
        rank = len(tensor.shape)
        dim = dim % rank
        starts = [0] * rank
        stops = list(tensor.shape)
        starts[dim], stops[dim] = start, stop
        return ttnn.slice(tensor, starts, stops)

    def blend(self, a: ttnn.Tensor, b: ttnn.Tensor, blend_extent: int, dim: int) -> ttnn.Tensor:
        """Cross-fade `a`'s tail into `b`'s head along `dim`; result is shaped like `b`.

        Mirrors the host `blend` in `vae_minimax_h3.py`, including its early return when the extent
        covers all of `b`.
        """
        rank = len(b.shape)
        axis = dim % rank
        blend_extent = min(a.shape[axis], b.shape[axis], blend_extent)

        tail_a = self._slice(a, axis, a.shape[axis] - blend_extent, a.shape[axis])
        head_b = self._slice(b, axis, 0, blend_extent)
        weight_a, weight_b = self._ramp_pair(tuple(head_b.shape), blend_extent, axis)
        blended = ttnn.add(ttnn.multiply(tail_a, weight_a), ttnn.multiply(head_b, weight_b))

        if blend_extent == b.shape[axis]:
            return blended
        rest = self._slice(b, axis, blend_extent, b.shape[axis])
        return ttnn.concat([blended, rest], dim=axis)

    def stitch(
        self,
        tiles: list[list[ttnn.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> ttnn.Tensor:
        """Device mirror of `stitch_tiles`, in the same order and with the same trims.

        Note `tiles[i - 1][j]` and `row[j - 1]` are the **original** tiles, not previously blended
        ones. That asymmetry is the reference's and is reproduced here.
        """
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = self._slice(tile, -2, 0, tile.shape[-2] - height_overlaps[i])
                if j < len(row) - 1:
                    tile = self._slice(tile, -1, 0, tile.shape[-1] - width_overlaps[j])
                result_row.append(tile)
            result_rows.append(ttnn.concat(result_row, dim=-1))
        return ttnn.concat(result_rows, dim=-2)


def unpatchify_device(
    tokens: ttnn.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    out_channels: int = 3,
    patch_size: int = 16,
    patch_size_t: int = 4,
) -> ttnn.Tensor:
    """Device mirror of `decoder_minimax_h3.unpatchify`: tokens to `(1, C, T*pt, H*p, W*p)`.

    `ttnn.permute` handles the 8-dimensional `(B,T,H,W,C,pt,p,p) -> (B,C,T,pt,H,p,W,p)` permutation
    directly, which is what makes keeping the tiles on device possible at all.
    """
    batch = tokens.shape[0]
    tokens = ttnn.slice(tokens, [0, 0, 0], [batch, num_frames * height * width, tokens.shape[-1]])
    tokens = ttnn.reshape(
        tokens, (batch, num_frames, height, width, out_channels, patch_size_t, patch_size, patch_size)
    )
    tokens = ttnn.permute(tokens, (0, 4, 1, 5, 2, 6, 3, 7))
    return ttnn.reshape(
        tokens, (batch, out_channels, num_frames * patch_size_t, height * patch_size, width * patch_size)
    )
