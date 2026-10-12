# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DS41F-0048: the device fp4 fake quant of the indexer's q / k (``tt/v41/attention.py`` ``fp4_act_fake_quant``) vs
model.py's ``fp4_act_quant(x, 32, inplace=True)`` with e8m0 scales, reproduced exactly by tt-blaze's CPU shim
(``golden/cpu_kernel_shim.py``; re-implemented here so the test needs no blaze import)."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41.attention import fp4_act_fake_quant

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _ref(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    xf = x.float().reshape(-1, x.shape[-1] // block, block)
    amax = xf.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))  # e8m0: the smallest power of two >= amax / 6
    v = torch.clamp(xf / s.unsqueeze(-1), -6.0, 6.0)
    a = v.abs()
    lo = torch.clamp(torch.searchsorted(_E2M1, a.contiguous(), right=True) - 1, 0, 7)
    hi = torch.clamp(lo + 1, max=7)
    d_lo, d_hi = a - _E2M1[lo], _E2M1[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0) & (hi != lo))
    mag = torch.where(pick_hi, _E2M1[hi], _E2M1[lo])
    return (torch.where(v < 0, -mag, mag) * s.unsqueeze(-1)).reshape(x.shape).to(x.dtype)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fp4_act_fake_quant(mesh_device):
    torch.manual_seed(0)
    x = torch.randn(1, 8, 64, 128) * torch.exp(torch.randn(1, 8, 64, 4).repeat_interleave(32, -1) * 2)
    x = x.to(torch.bfloat16)
    x[0, 0, 0, :32] = (
        torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4) * 1.0
    )  # amax 6 -> s = 1: exact ties
    want = _ref(x)
    t = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    got = ttnn.to_torch(fp4_act_fake_quant(t)).reshape(x.shape)
    eq = (got == want).float().mean().item()
    rel = ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()
    logger.info(f"[fp4 fake quant] device == model.py on {eq * 100:.3f} % of {x.numel()} values; max rel err {rel:.3e}")
    logger.info(f"[fp4 fake quant] tie row: device {got[0, 0, 0, :8].tolist()} model.py {want[0, 0, 0, :8].tolist()}")
    assert eq >= 0.999, f"only {eq * 100:.3f} % of the values match model.py's fp4 quant"
