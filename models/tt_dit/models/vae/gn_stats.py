# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn

TILE = 32


def gn_affine_from_stats(
    stats: ttnn.Tensor,
    pages: int,
    channels: int,
    groups: int,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    eps: float,
    h_mesh_axis: int | None,
    w_mesh_axis: int | None,
    ccl_manager,
    count: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """GroupNorm affine from a conv's epilogue statistics (ttnn.Conv2dFusedOps epilogue_stats).

    stats holds `pages` blocks of 2 * channels / 32 tiles per device, row 0 of tile c the sum of x and row 0 of tile
    N + c the sum of x * x over that block's output pixels for channels 32c .. 32c + 31. The sums are reduced over the
    blocks and the two mesh axes; count is the pixels of the whole image times the channels per group. Returns
    (scale, shift), [1, 1, 1, channels] bfloat16 TILE, with GroupNorm(x) * gamma + beta == x * scale + shift.
    """
    n = channels // TILE
    rep = channels // groups
    x = ttnn.to_layout(stats, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (pages, 2 * n, TILE, TILE))
    x = ttnn.slice(x, (0, 0, 0, 0), (pages, 2 * n, 1, TILE))
    x = ttnn.sum(x, dim=0, keepdim=True)
    x = ttnn.reshape(x, (1, 1, 2, channels))
    for axis in (h_mesh_axis, w_mesh_axis):
        if axis is not None:
            g = ccl_manager.all_gather(x, dim=1, mesh_axis=axis, use_hyperparams=False)
            x = ttnn.sum(g, dim=1, keepdim=True)
    x = ttnn.reshape(x, (1, 2, groups, rep))
    x = ttnn.sum(x, dim=3, keepdim=True)
    s1 = ttnn.slice(x, (0, 0, 0, 0), (1, 1, groups, 1))
    s2 = ttnn.slice(x, (0, 1, 0, 0), (1, 2, groups, 1))
    mean = ttnn.multiply(s1, 1.0 / count)
    var = ttnn.subtract(ttnn.multiply(s2, 1.0 / count), ttnn.multiply(mean, mean))
    rstd = ttnn.rsqrt(ttnn.add(var, eps))
    rstd_c = ttnn.reshape(ttnn.repeat(rstd, ttnn.Shape((1, 1, 1, rep))), (1, 1, 1, channels))
    mean_c = ttnn.reshape(ttnn.repeat(mean, ttnn.Shape((1, 1, 1, rep))), (1, 1, 1, channels))
    gamma_rm = ttnn.to_layout(ttnn.typecast(gamma, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
    beta_rm = ttnn.to_layout(ttnn.typecast(beta, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
    scale = ttnn.multiply(ttnn.reshape(gamma_rm, (1, 1, 1, channels)), rstd_c)
    shift = ttnn.subtract(ttnn.reshape(beta_rm, (1, 1, 1, channels)), ttnn.multiply(mean_c, scale))
    return tuple(ttnn.typecast(ttnn.to_layout(t, ttnn.TILE_LAYOUT), ttnn.bfloat16) for t in (scale, shift))
