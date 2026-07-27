# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm precision baseline — PCC / abs error / relative RMS / got-true ratio.

DO NOT DELETE.  This is the verifier's numerical reference point for the op at
Phase 0.  A refinement that changes the numerics (a dtype, a `math_fidelity`, a
DEST-accumulation flip, a reordered reduction) must re-run this file and record
the new row in `changelog.md`.

The last column is the **scale-bug detector**: `r = actual / expected` over the
finite, non-negligible reference elements.  A tight cluster of `r` around a
non-1.0 constant is a uniform scale / structural bug (a wrong scaler, a bad
broadcast, a CB race) — NOT rounding, and NOT fixable with fp32 intermediates.
A broad spread centred on 1.0 is ordinary bf16 precision noise.  Both signatures
can sit at PCC >= 0.999, which is exactly why the ratio is measured separately.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.onorm import onorm

from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

# Fixed KDA s6 head geometry (TP=1) — see eval/golden_tests/onorm/feature_spec.py.
HV = 32
V = 128
FLAT = HV * V

# bf16 in / bf16 out, so the golden suite's bf16 row is the bar: PCC >= 0.995.
PCC_TARGET = 0.995
EPS = 1e-5


def _torch_onorm(o, gate, weight, epsilon):
    o_f32 = o.to(torch.float32)
    ms = o_f32.pow(2).mean(dim=-1, keepdim=True)
    normed = o_f32 * torch.rsqrt(ms + epsilon)
    normed = normed * weight.to(torch.float32).reshape(1, 1, 1, V)
    flat = normed.reshape(o.shape[0], o.shape[1], HV * V)
    return flat * torch.sigmoid(gate.to(torch.float32))


def _ratio_spread(expected, actual):
    """Median and spread of r = actual / expected, over meaningful elements.

    Elements whose reference is near zero are excluded: their ratio is dominated
    by the absolute quantization of a value that carries no information, and
    including them turns every healthy tensor into a "broad spread".
    """
    scale = expected.abs().median()
    mask = torch.isfinite(actual) & torch.isfinite(expected) & (expected.abs() > 0.1 * scale)
    r = (actual[mask] / expected[mask]).to(torch.float64)
    p5, p50, p95 = torch.quantile(r, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return p50.item(), p5.item(), p95.item(), r.std().item()


# (batch, tokens): small / medium / larger / batched-larger.
SHAPES = [
    (1, 32),
    (1, 128),
    (1, 640),
    (4, 256),
]


@pytest.mark.parametrize("batch, tokens", SHAPES, ids=[f"B{b}xT{t}" for b, t in SHAPES])
def test_onorm_precision_baseline(device, batch, tokens):
    torch.manual_seed(42)

    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_gate = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = (1.0 + 0.02 * torch.randn(1, 1, 1, V)).to(torch.bfloat16)

    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gate = ttnn.from_torch(t_gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = onorm(o, gate, w, epsilon=EPS)

    expected = _torch_onorm(t_o, t_gate, t_w, EPS)
    actual = ttnn.to_torch(out).to(torch.float32)

    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (diff.pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()).item()
    r_med, r_p5, r_p95, r_std = _ratio_spread(expected, actual)

    _, allclose_msg = comp_allclose(expected, actual, rtol=0.05, atol=0.05)
    _, pcc_msg = comp_pcc(expected, actual, PCC_TARGET)

    print(
        f"\nPRECISION_BASELINE onorm B={batch} T={tokens} {pcc_msg} "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} rel_rms={rel_rms:.6f} "
        f"ratio_median={r_med:.6f} ratio_p5={r_p5:.6f} ratio_p95={r_p95:.6f} "
        f"ratio_std={r_std:.6f} | {allclose_msg}"
    )

    # Scale-bug guard: a uniform multiplicative error survives PCC but not this.
    assert 0.98 <= r_med <= 1.02, (
        f"got/true ratio median {r_med:.6f} is not ~1.0 — that is a uniform scale/structural "
        f"bug (wrong scaler / broadcast / CB race), not bf16 rounding"
    )
    # Noise guard: bf16 in and out, so a few percent of relative RMS is expected.
    assert rel_rms < 0.02, f"relative RMS {rel_rms:.6f} exceeds the bf16 round-trip budget"

    assert_with_pcc(expected, actual, PCC_TARGET)
