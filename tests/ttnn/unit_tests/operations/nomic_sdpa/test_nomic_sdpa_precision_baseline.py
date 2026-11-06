# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for nomic_sdpa (verifier, Phase 0).

Measures, against the float64 reference of the golden suite, over 4 shapes x the 3 golden input profiles (real, flat,
peaky) at the default compute config:
  PCC, max / mean abs error, relative RMS error, bf16 ULP error (p50 / p99 / max), and the got / true ratio spread
  (median, p5, p95 over elements with |reference| above 10% of its RMS) — the scale-bug detector: a tight cluster
  around a non-1.0 constant is a structural bug, a broad spread centred on 1.0 is rounding noise.

The bar is the golden suite's PROFILE_TOLERANCES (the incumbent's own error). Run with `-s` to see the table.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.nomic_sdpa.helpers import PROFILE_TOLERANCES, make_qkv, reference
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.nomic_sdpa import nomic_sdpa

HEADS, HEAD_DIM, HIDDEN = 12, 64, 768

SHAPES = [(1, 128), (4, 288), (2, 512), (8, 512)]


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 at |x| (8 significand bits): 2^(floor(log2|x|) - 7); tiny values use the min normal."""
    mag = x.abs().clamp_min(2.0**-126)
    return torch.pow(2.0, torch.floor(torch.log2(mag)) - 7)


def measure(got: torch.Tensor, expected: torch.Tensor) -> dict:
    got = got.double().flatten()
    expected = expected.double().flatten()
    diff = (got - expected).abs()
    rms_ref = expected.pow(2).mean().sqrt()
    ulp = diff / _bf16_ulp(expected)
    significant = expected.abs() > 0.1 * rms_ref
    ratio = got[significant] / expected[significant]
    gc = got - got.mean()
    ec = expected - expected.mean()
    return {
        "pcc": float(gc @ ec / (gc.norm() * ec.norm())),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_rms": float((got - expected).pow(2).mean().sqrt() / rms_ref),
        "ulp_p50": float(ulp.quantile(0.5)) if ulp.numel() < 2**24 else float(ulp[: 2**24 - 1].quantile(0.5)),
        "ulp_p99": float(ulp.kthvalue(max(1, int(0.99 * ulp.numel()))).values),
        "ulp_max": float(ulp.max()),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(ratio.kthvalue(max(1, int(0.05 * ratio.numel()))).values),
        "ratio_p95": float(ratio.kthvalue(max(1, int(0.95 * ratio.numel()))).values),
    }


@pytest.mark.parametrize("profile", ["real", "flat", "peaky"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_nomic_sdpa_precision_baseline(device, shape, profile):
    batch, seqlen = shape
    q, k, v = make_qkv((batch, HEADS, seqlen, HEAD_DIM), profile, seed=0)
    expected = reference(q, k, v, 1.0).reshape(batch, 1, seqlen, HIDDEN)
    config = ttnn.L1_MEMORY_CONFIG
    tq, tk, tv = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
        for t in (q, k, v)
    )
    output = nomic_sdpa(tq, tk, tv, scale=1.0, memory_config=config)
    got = ttnn.to_torch(output).reshape(batch, 1, seqlen, HIDDEN)
    for t in (tq, tk, tv, output):
        ttnn.deallocate(t)

    m = measure(got, expected)
    _, allclose_msg = comp_allclose(expected.float(), got.float())
    print(
        f"\nBASELINE {batch}x{seqlen} {profile:5s} pcc={m['pcc']:.6f} max_abs={m['max_abs']:.4f} "
        f"mean_abs={m['mean_abs']:.5f} rel_rms={m['rel_rms']:.5f} ulp p50/p99/max={m['ulp_p50']:.1f}/"
        f"{m['ulp_p99']:.1f}/{m['ulp_max']:.0f} ratio median={m['ratio_median']:.4f} "
        f"p5/p95={m['ratio_p5']:.4f}/{m['ratio_p95']:.4f} | {allclose_msg}"
    )

    pcc_bar, rms_bar = PROFILE_TOLERANCES[profile]
    assert_with_pcc(expected.float(), got.float(), pcc=pcc_bar)
    assert m["rel_rms"] <= rms_bar, f"relative RMS {m['rel_rms']:.5f} > bar {rms_bar}"
    # Scale-bug detector. The default (HiFi2 + approximate exp) is measured 1.9% small on `real` and 0.7% on `flat` /
    # `peaky` (median got/true 0.981 / 0.993): the approximate exp's one-sided error (exact exp: 0.995 / 0.997 / 1.000)
    # plus HiFi2 (HiFi4: 0.989). That is the precision class the golden bar was measured on (the incumbent uses the
    # same exp), not a structural bug; a structural scale bug (a wrong scaler, a lost tile of the row sum) is >= 3%.
    assert abs(m["ratio_median"] - 1.0) < 0.03, f"got/true median {m['ratio_median']:.4f}: a uniform scale error"


def _qkv_on_device(device, batch, seqlen, profile):
    q, k, v = make_qkv((batch, HEADS, seqlen, HEAD_DIM), profile, seed=0)
    config = ttnn.L1_MEMORY_CONFIG
    tensors = [
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
        for t in (q, k, v)
    ]
    return (q, k, v), tensors


def test_nomic_sdpa_refuses_bf16_dest_approx_exp_at_512(device, expect_error):
    """bf16 DEST + approximate exp misses the bar at S = 512 (real 0.0273, flat 0.0196): validate() must refuse it."""
    _, (tq, tk, tv) = _qkv_on_device(device, 2, 512, "real")
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=True, dst_full_sync_en=False
    )
    with expect_error(SupportRefusal, ""):
        nomic_sdpa(tq, tk, tv, compute_kernel_config=cfg)


@pytest.mark.parametrize("profile", ["real", "flat", "peaky"])
def test_nomic_sdpa_bf16_dest_exact_exp_meets_bar(device, profile):
    """The honored bf16-DEST config (exact exp) meets the bar at the perf focus."""
    batch, seqlen = 8, 512
    (q, k, v), (tq, tk, tv) = _qkv_on_device(device, batch, seqlen, profile)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False, dst_full_sync_en=False
    )
    output = nomic_sdpa(tq, tk, tv, compute_kernel_config=cfg)
    got = ttnn.to_torch(output).reshape(batch, 1, seqlen, HIDDEN)
    m = measure(got, reference(q, k, v, 1.0).reshape(batch, 1, seqlen, HIDDEN))
    print(f"\nBF16DEST_EXACT {batch}x{seqlen} {profile:5s} pcc={m['pcc']:.6f} rel_rms={m['rel_rms']:.5f}")
    pcc_bar, rms_bar = PROFILE_TOLERANCES[profile]
    assert m["pcc"] >= pcc_bar and m["rel_rms"] <= rms_bar, m
