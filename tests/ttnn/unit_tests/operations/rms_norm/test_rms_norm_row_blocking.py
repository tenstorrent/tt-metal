# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Refinement 7 — Row-blocking (BLOCK_HEIGHT > 1) for TILE Regime A many-row.
#
# Exercises the row-blocked compute path directly:
#   1. _regime_a_block_height actually selects bh>1 for grid-saturated many-row
#      no-gamma TILE shapes (and bh==1 everywhere the blast radius is bounded out:
#      gamma, ROW_MAJOR, uneven splits, single-row-per-core).
#   2. The bh>1 kernel path is numerically correct (all-ones exact + random vs torch).
#   3. The bh==1 anchor is unchanged.

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd


def _ht_wt(shape):
    W = shape[-1]
    Wt = (W + 31) // 32
    Ht = 1
    for d in shape[:-1]:
        Ht *= d
    Ht //= 32
    return Ht, Wt


def _bh_for(shape, has_gamma=False, gamma_is_rm=0, dtype=ttnn.bfloat16, fp32_acc=True, total_cores=64, enabled=True):
    Ht, Wt = _ht_wt(shape)
    cfg = pd._resolve_compute_config(None)
    cfg.fp32_dest_acc_en = fp32_acc
    num_cores = min(Ht, total_cores)
    old = pd._ENABLE_ROW_BLOCKING
    pd._ENABLE_ROW_BLOCKING = enabled
    try:
        return pd._regime_a_block_height(Ht, num_cores, has_gamma, gamma_is_rm, Wt, dtype, fp32_acc, cfg)
    finally:
        pd._ENABLE_ROW_BLOCKING = old


@pytest.fixture
def row_blocking_enabled():
    """Row-blocking is OFF by default in production (net-negative on this memory-bound
    kernel). Tests that exercise the bh>1 code path force it on, then restore."""
    old = pd._ENABLE_ROW_BLOCKING
    pd._ENABLE_ROW_BLOCKING = True
    try:
        yield
    finally:
        pd._ENABLE_ROW_BLOCKING = old


def test_row_blocking_disabled_by_default():
    # Production default: the knob is off, so a shape that WOULD row-block still gets bh=1.
    assert _bh_for((1, 1, 8192, 256), enabled=False) == 1
    assert _bh_for((1, 1, 16384, 256), enabled=False) == 1


# ---------------------------------------------------------------------------
# 1. bh selection (host-side, no device)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,expected_bh",
    [
        ((1, 1, 4096, 256), 2),  # Ht=128, 64 cores -> 2 rows/core, Wt=8 -> bh=2
        ((1, 1, 8192, 256), 4),  # Ht=256 -> 4 rows/core, bh=4 (== DEST limit @ fp32_acc)
        ((1, 1, 16384, 256), 4),  # Ht=512 -> 8 rows/core, bh=4 (largest divisor <= DEST)
        ((1, 1, 4096, 512), 2),  # Ht=128 -> 2 rows/core, Wt=16
        ((1, 1, 2048, 256), 1),  # Ht=64 == cores -> 1 row/core, no blocking
        ((1, 1, 1024, 256), 1),  # Ht=32 < cores -> grid not saturated, no blocking
        ((1, 1, 6144, 256), 3),  # Ht=192 -> 3 rows/core; 3 | 3 and 3 <= DEST(4) -> bh=3
    ],
)
def test_bh_selection_no_gamma_tile(shape, expected_bh):
    assert _bh_for(shape) == expected_bh


def test_bh_disabled_with_gamma():
    # gamma forces bh=1 (blast radius: no-gamma only)
    assert _bh_for((1, 1, 8192, 256), has_gamma=True) == 1


def test_bh_disabled_uneven_split():
    # Ht=130 over 64 cores is not an even split -> bh=1 (no remainder handling)
    assert _bh_for((1, 1, 4160, 256)) == 1


def test_bh_fp32acc_off_doubles_resident_not_dest():
    # bf16 input, fp32_dest_acc_en=False -> DEST limit 8, but intermediates stay bf16.
    # Ht=256 -> 4 rows/core, bh=4 (divisor of 4, <= 8).
    assert _bh_for((1, 1, 8192, 256), fp32_acc=False) == 4


# ---------------------------------------------------------------------------
# 2. numerical correctness of the bh>1 path on device
# ---------------------------------------------------------------------------
def _ref(x, gamma=None, eps=1e-6):
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return out if gamma is None else out * gamma


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 4096, 256),  # bh=2
        (1, 1, 8192, 256),  # bh=4
        (1, 1, 16384, 256),  # bh=4 (8 rows/core)
        (1, 1, 4096, 512),  # bh=2, wider W
        (2, 1, 4096, 256),  # bh selection on a 4D many-row shape
    ],
)
def test_row_blocked_random_vs_torch(device, row_blocking_enabled, shape):
    assert _bh_for(shape) > 1, f"shape {shape} must route to a row-blocked program for this test"
    torch.manual_seed(0)
    x = torch.randn(*shape)
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(ti)).float()
    ref = _ref(x)
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc} too low for {shape}"
    assert (out - ref).abs().max().item() <= 0.1, f"maxerr too high for {shape}"


@pytest.mark.parametrize("shape", [(1, 1, 4096, 256), (1, 1, 8192, 256)])
def test_row_blocked_all_ones_exact(device, row_blocking_enabled, shape):
    # All-ones input -> every row's RMS = 1 -> output == 1 everywhere.
    assert _bh_for(shape) > 1
    x = torch.ones(*shape)
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(ti)).float()
    assert (out - 1.0).abs().max().item() < 0.05, f"all-ones row-blocked not ~1.0 for {shape}"


def test_row_blocked_distinguishable_rows(device, row_blocking_enabled):
    # Each tile-row gets a distinct magnitude so a per-row recip mix-up (wrong
    # TileOffset indexing) would show. Row r scaled by (r+1).
    shape = (1, 1, 8192, 256)  # bh=4
    assert _bh_for(shape) == 4
    H = shape[-2]
    base = torch.randn(1, 1, H, shape[-1])
    scales = (torch.arange(H).float() + 1.0).reshape(1, 1, H, 1)
    x = base * scales
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(ti)).float()
    ref = _ref(x)
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc >= 0.999, f"per-row scaled PCC {pcc} too low (recip indexing bug?)"


# ---------------------------------------------------------------------------
# 3. bh==1 anchor unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 1, 2048, 256), (1, 1, 32, 256), (8, 1, 256, 256)])
def test_bh1_anchor_correct(device, shape):
    assert _bh_for(shape) == 1
    torch.manual_seed(1)
    x = torch.randn(*shape)
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(ti)).float()
    ref = _ref(x)
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc >= 0.999, f"bh=1 anchor PCC {pcc} for {shape}"
