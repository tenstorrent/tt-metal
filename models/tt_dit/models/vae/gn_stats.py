# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""GroupNorm statistics -> the next conv's affine.

The conv epilogue (`ttnn.conv2d(..., epilogue_stats=T)`) writes, per device, an fp32 DRAM interleaved TILE
tensor T of shape [1, 1, cores * 2N * 32, 32] (N = Cout / 32, cores in the conv's sharded grid order):
  tile core * 2N + c       row 0: column sums of x   for channels 32c .. 32c + 31 (that core's output pixels)
  tile core * 2N + N + c   row 0: column sums of x*x for the same channels
`gn_affine_from_stats` reduces those sums over the cores and over the H and W mesh axes and returns, on every device,
the per-channel affine that turns the conv's output into silu-ready GroupNorm output:
  GroupNorm(x)[c] * gamma[c] + beta[c] == x[c] * scale[c] + shift[c]
  mean_g = S1_g / count, var_g = S2_g / count - mean_g^2 (fp32), rstd_g = rsqrt(var_g + eps)
  scale[c] = gamma[c] * rstd_g(c), shift[c] = beta[c] - mean_g(c) * scale[c]
count = pixels over the whole image (all devices) x channels per group. The nearest upsample repeats every value 4x, so
it leaves mean / var unchanged and a level's first norm can reuse the previous level's page (scale count accordingly).

`gn_affine_reference` is the same computation in torch on host pages (the unit test's oracle and a debugging aid).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import ttnn

TILE = 32


def _page_sums_torch(pages: torch.Tensor, cores: int, channels: int) -> torch.Tensor:
    """[cores * 2N * 32, 32] host page -> [2, C] fp32 sums over the cores (row 0 of each tile)."""
    n = channels // TILE
    t = pages.reshape(cores, 2 * n, TILE, TILE)[:, :, 0, :].to(torch.float32)
    return t.sum(0).reshape(2, n * TILE)


def gn_affine_reference(
    device_pages: list[torch.Tensor],
    cores: int,
    channels: int,
    groups: int,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch oracle: the pages of every device (any order) -> (scale, shift), each [C] fp32."""
    sums = sum(_page_sums_torch(p, cores, channels) for p in device_pages)
    per_group = sums.reshape(2, groups, channels // groups).sum(-1)
    mean = per_group[0] / count
    var = per_group[1] / count - mean * mean
    rstd = torch.rsqrt(var + eps)
    rep = channels // groups
    scale = gamma.to(torch.float32) * rstd.repeat_interleave(rep)
    shift = beta.to(torch.float32) - mean.repeat_interleave(rep) * scale
    return scale, shift


def gn_affine_from_stats(
    stats: ttnn.Tensor,
    cores: int,
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
    """Device version of `gn_affine_reference`.

    stats: the epilogue page (per device) [1, 1, cores * 2N * 32, 32] fp32 TILE (DRAM).
    gamma / beta: [1, 1, 1, C] fp32 or bf16 (replicated).
    Returns (scale, shift), each [1, 1, 1, C] bf16 TILE, replicated on every device.
    All work is on 2 x C values per device plus one small gather per sharded mesh axis.
    """
    import ttnn

    n = channels // TILE
    rep = channels // groups
    x = ttnn.to_layout(stats, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (cores, 2 * n, TILE, TILE))
    x = ttnn.slice(x, (0, 0, 0, 0), (cores, 2 * n, 1, TILE))
    x = ttnn.sum(x, dim=0, keepdim=True)
    x = ttnn.reshape(x, (1, 1, 2, channels))
    for axis in (h_mesh_axis, w_mesh_axis):
        if axis is None:
            continue
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
    out = []
    for t in (scale, shift):
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
        out.append(ttnn.typecast(t, ttnn.bfloat16))
    return out[0], out[1]
