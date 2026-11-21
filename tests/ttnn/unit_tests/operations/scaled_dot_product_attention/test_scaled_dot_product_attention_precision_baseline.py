# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of scaled_dot_product_attention (verifier, Phase 0).

The first phase's precision corner (bfloat16 inputs, the float32 accumulator: the default compute config) on
small, medium and larger shapes, causal and not. Measured against the specification's float64 `reference`:
PCC (comp_pcc), max / mean absolute error (comp_allclose + torch), relative RMS (RMS error / std of the
reference, as eval/metrics.check_output), the bfloat16 ULP error, and the got/true ratio spread (median and
p5 / p95 of actual / expected over elements with |expected| >= 1e-3): a tight cluster around a constant other
than 1 would be a scale bug, a spread centered on 1 is rounding.

Gate: the specification's row for (bfloat16, fp32_dest_acc_en=True), PCC >= 0.999 (the numeric list's
`default` entry) and relative RMS <= 0.05 (NUMERIC["legacy_tolerances"]).

A test's own output is not shown under the simulator's workers, so every measurement is written to
generated/precision_baseline/scaled_dot_product_attention.jsonl (one JSON line per case) and recorded with
record_property.
"""

import json
import re
from pathlib import Path

import pytest
import torch

import ttnn
from eval.specs.scaled_dot_product_attention.spec import reference
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

PCC_MIN, REL_RMS_MAX = 0.999, 0.05
RESULTS = Path(__file__).resolve().parents[5] / "generated" / "precision_baseline" / "scaled_dot_product_attention.jsonl"

# (q shape, k/v shape, scale): small, the model's two calls, grouped heads, and the longest first-phase sequence.
SHAPES = [
    ((1, 1, 32, 32), (1, 1, 32, 32), None),
    ((1, 1, 128, 64), (1, 1, 128, 64), 0.125),
    ((1, 2, 64, 128), (1, 1, 64, 128), 0.08838834764831845),
    ((1, 8, 96, 64), (1, 2, 96, 64), 0.25),
    ((1, 2, 512, 64), (1, 1, 512, 64), 0.125),
]


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """The spacing of bfloat16 numbers at |x| (8 bits of significand)."""
    mag = x.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return torch.pow(2.0, torch.floor(torch.log2(mag)) - 7)


def _measure(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.double().flatten(), expected.double().flatten()
    _, pcc = comp_pcc(expected, actual)
    _, allclose = comp_allclose(expected, actual)
    max_abs = float(re.search(r"Max ATOL Delta: ([0-9.eE+-]+|nan|inf)", allclose).group(1))
    err = (actual - expected).abs()
    ulp = err / _bf16_ulp(expected)
    keep = expected.abs() >= 1e-3
    ratio = actual[keep] / expected[keep]
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], dtype=ratio.dtype)).tolist()
    return {
        "pcc": float(pcc),
        "max_abs": max_abs,
        "mean_abs": err.mean().item(),
        "rel_rms": ((actual - expected).pow(2).mean().sqrt() / expected.std()).item(),
        "ulp_p99": torch.quantile(ulp, 0.99).item(),
        "ulp_max": ulp.max().item(),
        "ratio_p5": q[0],
        "ratio_median": q[1],
        "ratio_p95": q[2],
        "finite": bool(torch.isfinite(actual).all()),
    }


@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shapes", SHAPES, ids=lambda s: "x".join(map(str, s[0])) + "+" + "x".join(map(str, s[1])))
def test_precision_baseline(device, shapes, is_causal, record_property):
    q_shape, kv_shape, scale = shapes
    torch.manual_seed(42)
    q, k, v = (torch.randn(s).bfloat16() for s in (q_shape, kv_shape, kv_shape))
    put = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)  # noqa: E731
    out = scaled_dot_product_attention(put(q), put(k), put(v), is_causal=is_causal, scale=scale)
    m = _measure(ttnn.to_torch(out), reference(q, k, v, is_causal=is_causal, scale=scale))

    case = {"shape": list(q_shape), "kv_shape": list(kv_shape), "is_causal": is_causal, **m}
    for key, value in m.items():
        record_property(key, value)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(case) + "\n")

    assert m["finite"]
    assert m["pcc"] >= PCC_MIN and m["rel_rms"] <= REL_RMS_MAX, case
