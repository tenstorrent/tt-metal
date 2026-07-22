# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debugging tests for rms_norm — DO NOT DELETE.

Documents the incremental bring-up (TILE first, then RM, gamma, partial W/H).
Deterministic inputs make every intermediate hand-calculable, so DEVICE_PRINT
values can be compared against known expectations.

The `device` fixture comes from the module-scoped conftest — never open a
device here.
"""

import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


def _rms_ref(x, gamma, eps):
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(var + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _run(device, shape, layout, dtype, with_gamma, eps=1e-6):
    torch.manual_seed(0)
    W = shape[-1]
    ti = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(ti, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = None
    tg = None
    if with_gamma:
        tg = torch.randn(W, dtype=torch.float32)
        g = ttnn.from_torch(
            tg.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    out = rms_norm(x, gamma=g, epsilon=eps)
    r = ttnn.to_torch(out).to(torch.float32)
    exp = _rms_ref(ti, tg, eps)
    maxdiff = (r - exp).abs().max().item()
    return r, exp, maxdiff


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---- Phase A: TILE, tile-aligned, no gamma ----


def test_tile_single_no_gamma(device):
    r, exp, md = _run(device, (32, 32), ttnn.TILE_LAYOUT, ttnn.bfloat16, with_gamma=False)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"


def test_tile_multi_no_gamma(device):
    r, exp, md = _run(device, (64, 128), ttnn.TILE_LAYOUT, ttnn.bfloat16, with_gamma=False)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"


# ---- Phase B: gamma ----


def test_tile_single_gamma(device):
    r, exp, md = _run(device, (32, 32), ttnn.TILE_LAYOUT, ttnn.bfloat16, with_gamma=True)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"


# ---- Phase D: partial W (TILE) ----


def test_tile_partial_w(device):
    r, exp, md = _run(device, (32, 50), ttnn.TILE_LAYOUT, ttnn.bfloat16, with_gamma=True)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"


# ---- Phase E: RM ----


def test_rm_single_no_gamma(device):
    r, exp, md = _run(device, (32, 32), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, with_gamma=False)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"


def test_rm_gamma(device):
    r, exp, md = _run(device, (64, 128), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, with_gamma=True)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"


def test_rm_partial_w(device):
    r, exp, md = _run(device, (32, 50), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, with_gamma=True)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"


def test_rm_partial_h(device):
    r, exp, md = _run(device, (50, 64), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, with_gamma=True)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"


# ---- multi-core (many tile-rows) ----


def test_tile_multicore(device):
    r, exp, md = _run(device, (2, 4, 64, 128), ttnn.TILE_LAYOUT, ttnn.bfloat16, with_gamma=True)
    assert _pcc(r, exp) >= 0.99, f"pcc={_pcc(r, exp):.5f} maxdiff={md}"
