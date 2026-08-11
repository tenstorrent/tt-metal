# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm — the Phase 0 accuracy record.

Measures, per (shape x dtype x layout x gamma) cell:
  * PCC (Pearson correlation) against an fp32 torch reference,
  * max / mean absolute error,
  * relative RMS error (RMS error / RMS of the reference),
  * the got/true RATIO SPREAD — median r and its p5/p95, where
    r = actual / expected over the finite, non-zero-reference elements.

The ratio spread is the scale-bug detector: a tight cluster of `r` around a
NON-1.0 constant is a uniform scale / structural bug (a mis-scaled reduction, a
padded denominator, a broadcast mistake), which PCC is largely blind to. A broad
spread centred on 1.0 is ordinary rounding noise. Both are printed for every
cell, so a future refinement can tell the two apart without re-deriving them.

The numbers this file prints are recorded in `verification_report.md`.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm import rms_norm


EPS = 1e-6

# Baseline gates: loose enough to be a record rather than a tripwire, tight
# enough that a scale bug or a lost fp32 accumulation fails the run.
PCC_GATE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

SHAPES = [
    pytest.param((1, 1, 32, 32), id="small_1x1x32x32"),
    pytest.param((1, 1, 128, 512), id="medium_1x1x128x512"),
    pytest.param((1, 1, 32, 4096), id="wide_1x1x32x4096"),
    pytest.param((2, 1, 1024, 1024), id="large_2x1x1024x1024"),
    pytest.param((1, 1, 40, 200), id="non_aligned_1x1x40x200"),
]


def _reference(x, gamma):
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out


def _metrics(expected, actual):
    err = (actual - expected).abs()
    ref_rms = expected.pow(2).mean().sqrt().item()
    rel_rms = (err.pow(2).mean().sqrt().item() / ref_rms) if ref_rms > 0 else float("nan")

    # got/true ratio over finite, non-negligible reference elements.
    mask = torch.isfinite(actual) & (expected.abs() > 1e-6 * max(ref_rms, 1e-30))
    if mask.any():
        r = (actual[mask] / expected[mask]).double()
        ratio_median = r.median().item()
        ratio_p5 = torch.quantile(r, 0.05).item()
        ratio_p95 = torch.quantile(r, 0.95).item()
    else:  # pragma: no cover - degenerate reference
        ratio_median = ratio_p5 = ratio_p95 = float("nan")

    return {
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel_rms": rel_rms,
        "ratio_median": ratio_median,
        "ratio_p5": ratio_p5,
        "ratio_p95": ratio_p95,
    }


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm_precision_baseline(device, shape, dtype, layout, with_gamma):
    torch.manual_seed(0)
    torch_dtype = TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    torch_gamma = (
        torch.randn((1,) * (len(shape) - 1) + (W,), dtype=torch.float32).to(torch_dtype) if with_gamma else None
    )

    expected = _reference(torch_input, torch_gamma)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=dtype, layout=layout, device=device) if with_gamma else None

    tt_out = rms_norm(tt_input, gamma=tt_gamma, epsilon=EPS)
    actual = ttnn.to_torch(tt_out).float()

    m = _metrics(expected, actual)
    allclose, allclose_msg = comp_allclose(expected, actual, rtol=0.05, atol=0.05)
    _, pcc_msg = assert_with_pcc(expected, actual, PCC_GATE[dtype])

    print(
        f"\n[precision] shape={tuple(shape)} dtype={dtype} layout={layout} gamma={with_gamma}\n"
        f"            {pcc_msg}\n"
        f"            max_abs={m['max_abs']:.6f} mean_abs={m['mean_abs']:.6f} rel_rms={m['rel_rms']:.6f}\n"
        f"            ratio median={m['ratio_median']:.6f} p5={m['ratio_p5']:.6f} p95={m['ratio_p95']:.6f}\n"
        f"            {allclose_msg} (allclose@rtol=atol=0.05: {allclose})"
    )

    # A uniform, non-1.0 got/true ratio is a scale/structural bug, not rounding:
    # PCC stays high while the whole tensor is mis-scaled. Gate on it explicitly.
    assert abs(m["ratio_median"] - 1.0) < 0.02, (
        f"got/true ratio median {m['ratio_median']:.6f} is not ~1.0 — uniform scale error "
        f"(p5={m['ratio_p5']:.6f}, p95={m['ratio_p95']:.6f})"
    )
    assert_with_pcc(expected, actual, PCC_GATE[dtype])
