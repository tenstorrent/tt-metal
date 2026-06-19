# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Refinement 1 — Regime B (wide-W cross-core all-gather) correctness.
#
# Before this refinement, Regime B output was too large by exactly
# sqrt(2*num_chunks): the gathered/combined Sum(x^2) underflowed by
# 1/(2*num_chunks). Two compounding bugs lived at the cb_partial_sumsq
# cross-thread handshake (reader read a stale/early front; combine read the
# un-popped PASS-1 local sum). The fix introduces a dedicated single-push
# cb_local_sumsq handoff. These tests pin the corrected behavior on every
# Regime-B-routed shape, including the LOOSE wide-W cases.
#
# DO NOT DELETE — documents the Refinement 1 correctness contract.

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

# Shapes that the host heuristic routes to Regime B (Ht_total < grid OR a full row
# exceeds the L1 resident budget). num_chunks (= ceil(Wt_s / reduce_block)) is the
# parameter the old bug scaled with, so the set spans num_chunks = 1, 2, 3.
REGIME_B_SHAPES = [
    (1, 1, 32, 4096),  # K=64, Wt_s=2,  num_chunks=1
    (1, 1, 32, 1280),  # K=40, Wt_s=1,  num_chunks=1
    (1, 1, 64, 8192),  # K=32, Wt_s=8,  num_chunks=2
    (1, 1, 128, 4096),  # K=16, Wt_s=8,  num_chunks=2
    (1, 1, 256, 2048),  # K=8,  Wt_s=8,  num_chunks=2
    (1, 1, 64, 12288),  # K=32, Wt_s=12, num_chunks=3
    (1, 32, 4096),
    (128, 512),
    (2, 1, 64, 4096),
]

# LOOSE cross-core cases (very wide W, exercise the largest K / Wt_s).
LOOSE_SHAPES = [
    (1, 1, 32, 16384),
    (1, 1, 32, 32768),
    (1, 1, 64, 12288),
]


def _torch_rms_norm(x, gamma=None, eps=1e-6):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(var + eps)
    return out * gamma if gamma is not None else out


def _rel_rms(actual, expected):
    return (actual - expected).pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()


def _pcc(actual, expected):
    return torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1]


@pytest.mark.parametrize("shape", REGIME_B_SHAPES + LOOSE_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_regime_b_all_ones_exact(device, shape):
    """All-ones input: RMS(ones) == 1 everywhere. Any all-gather/combine scale error
    shows up directly as a deviation from 1.0 (the old bug returned sqrt(2*num_chunks))."""
    x = torch.ones(shape, dtype=torch.bfloat16)
    ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(ti)).float()
    # bf16 RMS over a uniform field is exact to a tight band.
    assert torch.allclose(out, torch.ones_like(out), rtol=0.02, atol=0.02), f"mean={out.mean().item()} (expected 1.0)"


@pytest.mark.parametrize("shape", REGIME_B_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_regime_b_distinguishable_shards(device, shape):
    """Per-position-distinguishable input: each W-shard carries a different magnitude,
    so the all-gather must deliver ALL K distinct partials (not K copies of one). A
    monotonic ramp along W makes a dropped/duplicated shard visibly wrong."""
    W = shape[-1]
    # Ramp 0.5 .. 1.5 along W so each shard's Sum(x^2) is distinct.
    ramp = torch.linspace(0.5, 1.5, W).reshape(*([1] * (len(shape) - 1)), W)
    x = ramp.expand(shape).contiguous()
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(ti)).float()
    expected = _torch_rms_norm(x.float())
    assert _pcc(out, expected) > 0.999, f"pcc={_pcc(out, expected).item()}"
    assert _rel_rms(out, expected) < 0.04, f"relRMS={_rel_rms(out, expected).item()}"


@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "gamma"])
@pytest.mark.parametrize("shape", REGIME_B_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_regime_b_random_vs_torch(device, shape, with_gamma):
    """Random standard-normal input vs the torch reference, both regimes of the
    normalize pass (with and without gamma)."""
    torch.manual_seed(0)
    W = shape[-1]
    x = torch.randn(shape)
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = tg = None
    if with_gamma:
        g = torch.randn(W)
        tg = ttnn.from_torch(
            g.reshape(1, 1, 1, W).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
    out = ttnn.to_torch(rms_norm(ti, gamma=tg)).float()
    expected = _torch_rms_norm(x.float(), g.float() if g is not None else None)
    assert _pcc(out, expected) > 0.999, f"pcc={_pcc(out, expected).item()}"
    assert _rel_rms(out, expected) < 0.04, f"relRMS={_rel_rms(out, expected).item()}"
