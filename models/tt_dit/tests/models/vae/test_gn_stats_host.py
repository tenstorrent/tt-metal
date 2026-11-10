# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only check of the GroupNorm-statistics page layout and math: build the pages the conv
epilogue will write (per device, per core, column sums of x and x*x in row 0 of each tile) from a random activation
sharded over a (devices_h x devices_w) mesh and the cores of each device, run `gn_affine_reference`, and compare
x * scale + shift with torch's GroupNorm(x) * gamma + beta. No device needed.
  pytest -q test_gn_stats_host.py
"""

from __future__ import annotations

import pytest
import torch

from models.tt_dit.models.vae.gn_stats import TILE, gn_affine_reference


def _pages(x_dev: torch.Tensor, cores: int) -> torch.Tensor:
    """x_dev [rows, C] (one device's pixels, row-major) -> the epilogue page [cores * 2N * 32, 32] (fp32)."""
    _, c = x_dev.shape
    n = c // TILE
    page = torch.zeros(cores, 2 * n, TILE, TILE, dtype=torch.float32)
    for k, chunk in enumerate(x_dev.chunk(cores, dim=0)):  # height-sharded: consecutive pixel rows per core
        s1 = chunk.sum(0).reshape(n, TILE)
        s2 = (chunk * chunk).sum(0).reshape(n, TILE)
        page[k, :n, 0, :] = s1
        page[k, n:, 0, :] = s2
    return page.reshape(cores * 2 * n * TILE, TILE)


@pytest.mark.parametrize("h, w, c, groups, dh, dw, cores", [(64, 64, 128, 32, 8, 4, 8), (32, 48, 256, 32, 2, 3, 4)])
def test_gn_affine_reference(h, w, c, groups, dh, dw, cores):
    torch.manual_seed(0)
    x = torch.randn(h, w, c, dtype=torch.float64) * 3 + 0.5
    gamma = torch.rand(c, dtype=torch.float64) + 0.5
    beta = torch.randn(c, dtype=torch.float64)
    eps = 1e-6
    pages = []
    for i in range(dh):
        for j in range(dw):
            blk = x[i * h // dh : (i + 1) * h // dh, j * w // dw : (j + 1) * w // dw].reshape(-1, c)
            pages.append(_pages(blk.to(torch.float32), cores))
    count = h * w * (c // groups)
    scale, shift = gn_affine_reference(pages, cores, c, groups, gamma.float(), beta.float(), eps, count)
    ref = torch.nn.functional.group_norm(x.permute(2, 0, 1)[None], groups, gamma, beta, eps)[0].permute(1, 2, 0)
    got = x * scale.double() + shift.double()
    err = ((got - ref).norm() / ref.norm()).item()
    assert err < 1e-5, f"rel err {err:.3e}"
