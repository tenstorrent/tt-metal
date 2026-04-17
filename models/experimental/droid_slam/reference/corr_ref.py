# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure-torch replacement for upstream DROID-SLAM's CUDA correlation
block.

Upstream's `droid_slam/modules/corr.py::CorrBlock` relies on two
compiled CUDA kernels via `droid_backends`:

  * `corr_index_forward` — bilinear lookup of a correlation volume at
    a (2r+1)x(2r+1) window around per-pixel coordinates.
  * (implicitly) the all-pairs correlation `matmul`, which is already
    expressible in torch.

On this Tenstorrent-only machine we have no CUDA and cannot build
those extensions, so this module ports the same maths to pure torch:
`torch.matmul` for the all-pairs inner product, `F.avg_pool2d` for the
pyramid, and `F.grid_sample` for the windowed bilinear lookup. The
API mirrors upstream so higher-level code can import this instead of
`modules.corr`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coords_grid(ht: int, wd: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Per-pixel (x, y) identity coordinate grid of shape (ht, wd, 2).

    Matches `geom.projective_ops.coords_grid` from upstream but without
    the lietorch dependency.
    """
    y, x = torch.meshgrid(
        torch.arange(ht, device=device, dtype=dtype),
        torch.arange(wd, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1)


def _corr_index_sample(volume: torch.Tensor, coords: torch.Tensor, radius: int) -> torch.Tensor:
    """Pure-torch replacement for `droid_backends.corr_index_forward`.

    Args:
        volume: shape ``(BN, H1, W1, H2, W2)``. For each source
            position ``(y1, x1)`` in ``(H1, W1)`` we read a 2-D
            correlation map of size ``(H2, W2)``.
        coords: shape ``(BN, 2, H1, W1)``. Query center ``(cx, cy)``
            for each source position, in ``volume``'s (H2, W2) pixel
            coordinates.
        radius: integer window half-extent; output has
            ``(2*radius + 1) ** 2`` channels.

    Returns:
        ``(BN, (2r+1)**2, H1, W1)`` float tensor — per-source-position
        bilinearly sampled window around ``coords``.
    """
    bn, h1, w1, h2, w2 = volume.shape
    # Treat every (bn, y1, x1) slice as its own tiny (H2, W2) image so
    # grid_sample can process them all in one call.
    vol = volume.reshape(bn * h1 * w1, 1, h2, w2)

    device = coords.device
    dtype = volume.dtype
    # (cx, cy) at each source position.
    c = coords.permute(0, 2, 3, 1).reshape(bn * h1 * w1, 1, 1, 2).to(dtype)

    r = radius
    dy, dx = torch.meshgrid(
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    offsets = torch.stack([dx, dy], dim=-1).reshape(1, 2 * r + 1, 2 * r + 1, 2)

    grid = c + offsets  # (BN*H1*W1, 2r+1, 2r+1, 2) in level-i pixel coords

    # Normalize to [-1, 1] for grid_sample. align_corners=True so that a
    # query at pixel index k maps to 2*k/(size-1) - 1, matching upstream's
    # integer-centered lookup semantics.
    w_denom = max(w2 - 1, 1)
    h_denom = max(h2 - 1, 1)
    gx = 2.0 * grid[..., 0] / w_denom - 1.0
    gy = 2.0 * grid[..., 1] / h_denom - 1.0
    grid_norm = torch.stack([gx, gy], dim=-1)

    sampled = F.grid_sample(
        vol, grid_norm, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    # (BN*H1*W1, 1, 2r+1, 2r+1) → (BN, H1, W1, (2r+1)^2) → (BN, C, H1, W1)
    sampled = sampled.reshape(bn, h1, w1, (2 * r + 1) ** 2)
    return sampled.permute(0, 3, 1, 2).contiguous()


class CorrBlock:
    """Pure-torch equivalent of `modules.corr.CorrBlock`.

    Constructs an all-pairs correlation volume between matching source
    feature maps, downsamples into a 4-level pyramid, and exposes a
    `__call__(coords)` that returns the per-level bilinear lookup
    windows concatenated along the channel axis.
    """

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 3):
        self.num_levels = num_levels
        self.radius = radius
        corr = CorrBlock.corr(fmap1, fmap2)
        b, n, h1, w1, h2, w2 = corr.shape
        self.B, self.N, self.H1, self.W1 = b, n, h1, w1

        # Flatten per-source to 1-channel 2-D images for avg-pool.
        flat = corr.reshape(b * n * h1 * w1, 1, h2, w2)
        self.corr_pyramid = []
        cur = flat
        for i in range(num_levels):
            h2i = h2 // (2 ** i)
            w2i = w2 // (2 ** i)
            self.corr_pyramid.append(cur.view(b * n, h1, w1, h2i, w2i))
            if i + 1 < num_levels:
                cur = F.avg_pool2d(cur, 2, stride=2)

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        """All-pairs normalized correlation.

        fmap1, fmap2: (B, N, C, H, W). Returns (B, N, H, W, H, W).
        The ``/ 4.0`` factors on each operand match upstream's implicit
        normalization (which divides the final dot product by 16).
        """
        b, n, c, ht, wd = fmap1.shape
        f1 = fmap1.reshape(b * n, c, ht * wd) / 4.0
        f2 = fmap2.reshape(b * n, c, ht * wd) / 4.0
        corr = torch.matmul(f1.transpose(1, 2), f2)  # (B*N, H*W, H*W)
        return corr.view(b, n, ht, wd, ht, wd)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Lookup per-level windows at `coords` and concat.

        coords: (B, N, H1, W1, 2). Returns (B, N, num_levels*(2r+1)^2, H1, W1).
        """
        b, n, h1, w1, _ = coords.shape
        c = coords.permute(0, 1, 4, 2, 3).contiguous().view(b * n, 2, h1, w1)

        out = []
        for i in range(self.num_levels):
            samp = _corr_index_sample(self.corr_pyramid[i], c / (2 ** i), self.radius)
            out.append(samp.view(b, n, -1, h1, w1))
        return torch.cat(out, dim=2)
