# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Coarse-to-fine SDF grid evaluation (upstream HierarchicalVolumeDecoding), device-agnostic.

The dense octree-256 grid costs 257^3 = 16.97 M decoder queries. Here the grid is evaluated at 65^3, then at each finer
level (129^3, 257^3) only around cells the coarser level marks as near the surface: a sign change against a
6-neighbour or |logit| < `band`, dilated by one coarse cell (all but the last level) and by one or two fine cells, as
upstream. Upstream leaves the unvisited points NaN; here they take the coarser level's value, trilinearly upsampled
(the grids nest: coarse index i is fine index 2i), so the result is a full grid marching cubes can take unchanged
and its sign far from the surface stays the coarse one.

`evaluate(points [N, 3] fp32) -> logits [N]` is the decoder (device or CPU).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def _levels(resolution: int, min_resolution: int) -> list[int]:
    res = []
    if resolution < min_resolution:
        res.append(resolution)
    while resolution >= min_resolution:
        res.append(resolution)
        resolution //= 2
    return res[::-1]


def _axis_points(resolution: int, bound: float) -> torch.Tensor:
    return torch.linspace(-bound, bound, resolution + 1, dtype=torch.float32)


def near_surface(grid: torch.Tensor, level: float = 0.0) -> torch.Tensor:
    """Points whose sign differs from any 6-neighbour (replicate padding), as extract_near_surface_volume_fn."""
    s = torch.sign(grid + level)
    p = F.pad(s[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    c = p[1:-1, 1:-1, 1:-1]
    m = torch.zeros_like(c, dtype=torch.bool)
    for sl in (
        (slice(0, -2), slice(1, -1), slice(1, -1)),
        (slice(2, None), slice(1, -1), slice(1, -1)),
        (slice(1, -1), slice(0, -2), slice(1, -1)),
        (slice(1, -1), slice(2, None), slice(1, -1)),
        (slice(1, -1), slice(1, -1), slice(0, -2)),
        (slice(1, -1), slice(1, -1), slice(2, None)),
    ):
        m |= p[sl] != c
    return m


def _dilate(mask: torch.Tensor, steps: int) -> torch.Tensor:
    m = mask[None, None].float()
    for _ in range(steps):
        m = F.max_pool3d(m, 3, stride=1, padding=1)
    return m[0, 0] > 0


def hierarchical_grid(
    evaluate: Callable[[torch.Tensor], torch.Tensor],
    resolution: int = 256,
    *,
    min_resolution: int = 63,
    bound: float = 1.01,
    level: float = 0.0,
    band: float = 0.95,
    stats: dict | None = None,
) -> torch.Tensor:
    """Full [(res+1)^3] logit grid, decoder queried only near the surface below the coarsest level."""
    levels = _levels(resolution, min_resolution)
    lin = _axis_points(levels[0], bound)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid = evaluate(torch.stack([x, y, z], -1).reshape(-1, 3)).reshape(len(lin), len(lin), len(lin))
    counts = [grid.numel()]
    for i, r in enumerate(levels[1:]):
        last = r == levels[-1]
        mark = near_surface(grid, level) | (grid.abs() < band)
        if not last:
            mark = _dilate(mark, 1)
        nxt = torch.zeros((r + 1,) * 3, dtype=torch.bool)
        nxt[::2, ::2, ::2] = mark
        nxt = _dilate(nxt, 2 if last else 1)
        idx = nxt.nonzero()
        lin = _axis_points(r, bound)
        pts = torch.stack([lin[idx[:, 0]], lin[idx[:, 1]], lin[idx[:, 2]]], -1)
        up = F.interpolate(grid[None, None], size=(r + 1,) * 3, mode="trilinear", align_corners=True)[0, 0]
        up[idx[:, 0], idx[:, 1], idx[:, 2]] = evaluate(pts).float()
        grid = up
        counts.append(idx.shape[0])
    if stats is not None:
        stats.update(levels=[l + 1 for l in levels], points_per_level=counts, points=sum(counts))
    return grid
