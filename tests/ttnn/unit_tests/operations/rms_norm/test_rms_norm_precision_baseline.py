# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for ttnn.operations.rms_norm (verifier artifact).

Measures, per (shape, dtype, layout):

  * PCC                — via `assert_with_pcc` (tests.ttnn.utils_for_testing)
  * max / mean abs err  — reported by `comp_allclose` (models.common.utility_functions)
  * relative RMS error  — ||got - true||_2 / ||true||_2
  * got/true RATIO SPREAD — median r and its p5/p95 spread over the finite,
    non-negligible reference elements.

The ratio spread is the scale-bug detector: a *tight* cluster of `r` around a
non-1.0 constant is a uniform scale / structural bug (a CB race, a wrong
scaler, a broadcast mistake) — NOT rounding — and it is exactly the signature
that hides behind a high PCC. A broad spread centred on 1.0 is ordinary
precision noise.

Numbers are printed (run with `-s`) and recorded in
`ttnn/ttnn/operations/rms_norm/verification_report.md`. The asserts are loose
gates (the golden-suite thresholds); the *measurement* is the point.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm import rms_norm


TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# Same gates the golden suite uses (helpers.TOLERANCES): (pcc, relative rms).
TOLERANCES = {
    ttnn.float32: (0.999, 0.02),
    ttnn.bfloat16: (0.995, 0.04),
}

# small / medium / large-ish / wide-hidden (chunked reduce, NW > 1).
SHAPES = [
    (1, 1, 32, 64),
    (1, 1, 64, 128),
    (2, 4, 128, 512),
    (1, 1, 32, 4096),
]


def _torch_rms_norm(x, gamma, epsilon):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _ratio_spread(actual, expected):
    """median / p5 / p95 of got/true over finite, non-negligible reference lanes."""
    scale = expected.abs().max().clamp_min(1e-30)
    mask = torch.isfinite(actual) & torch.isfinite(expected) & (expected.abs() > 1e-3 * scale)
    if mask.sum() < 8:
        return float("nan"), float("nan"), float("nan")
    r = (actual[mask] / expected[mask]).to(torch.float64)
    q = torch.quantile(r, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return q[1].item(), q[0].item(), q[2].item()


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_rms_norm_precision_baseline(device, shape, dtype, layout):
    torch.manual_seed(0)
    epsilon = 1e-6

    torch_x = torch.randn(shape, dtype=TORCH_DTYPE[dtype])
    torch_gamma = torch.randn(shape[-1], dtype=TORCH_DTYPE[dtype])

    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device)
    tt_gamma = ttnn.from_torch(torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=layout, device=device)

    tt_out = rms_norm(tt_x, gamma=tt_gamma, epsilon=epsilon)

    expected = _torch_rms_norm(torch_x, torch_gamma, epsilon)
    actual = ttnn.to_torch(tt_out).to(torch.float32)

    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)).item()
    r_med, r_p5, r_p95 = _ratio_spread(actual, expected)

    _, allclose_msg = comp_allclose(expected, actual, rtol=1e-2, atol=1e-2)

    pcc_gate, rms_gate = TOLERANCES[dtype]
    layout_id = "TILE" if layout == ttnn.TILE_LAYOUT else "RM"
    dtype_id = "bf16" if dtype == ttnn.bfloat16 else "fp32"
    print(
        f"\n[precision] {str(tuple(shape)):>18} {dtype_id:>4} {layout_id:>4} | "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_rms={rel_rms:.3e} | "
        f"ratio med={r_med:.6f} p5={r_p5:.6f} p95={r_p95:.6f} | {allclose_msg}"
    )

    # Scale-bug tripwire. A uniform scale/structural bug shows up as a
    # SYSTEMATIC offset of the ratio median that dominates the random spread;
    # ordinary rounding shows up as spread centred (near) 1.0. Trip only when
    # the offset beats both the spread and an absolute floor, so genuine
    # truncation bias (fp32 Src-register truncation biases products a little
    # low — see verification_report.md) does not false-trip.
    spread = abs(r_p95 - r_p5)
    assert abs(r_med - 1.0) <= max(2.0 * spread, 5e-3), (
        f"got/true ratio median {r_med:.6f} (p5={r_p5:.6f}, p95={r_p95:.6f}) is a "
        f"systematic offset dominating the spread — uniform scale/structural bug, "
        f"not precision"
    )

    assert rel_rms < rms_gate, f"relative RMS {rel_rms:.3e} exceeds {rms_gate}"
    assert_with_pcc(expected, actual, pcc_gate)
