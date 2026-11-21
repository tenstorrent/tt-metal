# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of add on its first-phase cells (bfloat16 TILE operands, fp32 DEST, default config).

Per shape: PCC (comp_pcc), max abs error (comp_allclose), mean abs error, relative RMS error (RMS error over the
reference's std, as eval/metrics.check_output), ULP distance on the bfloat16 grid against the float64 reference
rounded once (the specification grades bfloat16 at 1 ulp, NUMERIC_RESOLVED), and the got/true ratio spread
(median, p5, p95 over finite elements with a non-negligible reference), the scale-bug detector: a tight cluster
around a constant other than 1.0 is a structural bug, a spread centred on 1.0 is rounding.

Under forked workers a test's output is not shown, so each case also writes its numbers to
$ADD_PRECISION_OUT (default /tmp/add_precision_baseline/<id>.json).
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from eval.metrics import default_tolerance
from ttnn.operations.add import add

OUT_DIR = Path(os.environ.get("ADD_PRECISION_OUT", "/tmp/add_precision_baseline"))

SHAPES = [
    (1, 1, 17, 50),  # small, non-aligned in H and W (one padded tile)
    (1, 2, 8400),  # YOLOv8 dist2bbox: 263 tiles, 2 valid rows per tile
    (1, 1, 1600, 128),  # YOLOv8 bottleneck residual: 200 tiles
    (2, 4, 128, 512),  # larger: 1,024 tiles over the full grid
]


def _ulp_bf16(actual, expected):
    """Per-element ULP distance on the bfloat16 grid (both quantized to bfloat16 first)."""

    def ordered(x):
        bits = x.to(torch.bfloat16).view(torch.int16).to(torch.int64)
        neg = bits < 0
        mag = torch.where(neg, bits + (1 << 15), bits)
        return torch.where(neg, -mag, mag)

    return (ordered(actual) - ordered(expected)).abs()


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_add_precision_baseline(device, shape, record_property):
    torch.manual_seed(0)
    ta = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tb = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(add(a, b)).double()
    ref = ta.double() + tb.double()  # the specification's reference: float64 on the stored inputs

    _, pcc = comp_pcc(ref, got)
    _, allclose_msg = comp_allclose(ref.float(), got.float())
    diff = (got - ref).abs()
    max_abs, mean_abs = diff.max().item(), diff.mean().item()
    rel_rms = (diff.pow(2).mean().sqrt() / ref.std()).item()
    ulp = _ulp_bf16(got, ref)

    mask = torch.isfinite(got) & (ref.abs() > 1e-2)
    r = (got[mask] / ref[mask]).float()
    ratio = {
        "median": r.median().item(),
        "p5": r.quantile(0.05).item(),
        "p95": r.quantile(0.95).item(),
    }
    row = {
        "shape": list(shape),
        "pcc": float(pcc),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rel_rms_err": rel_rms,
        "ulp_max": int(ulp.max().item()),
        "ulp_frac_nonzero": (ulp > 0).double().mean().item(),
        "ratio": ratio,
        "comp_allclose": allclose_msg,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ("x".join(map(str, shape)) + ".json")).write_text(json.dumps(row))
    for k, v in row.items():
        record_property(k, v)

    pcc_min, rms_max = default_tolerance(ttnn.bfloat16, True)
    assert pcc >= pcc_min, row
    assert rel_rms <= rms_max, row
    assert row["ulp_max"] <= 1, row  # bfloat16 graded at 1 ulp (NUMERIC_RESOLVED)
    assert abs(ratio["median"] - 1.0) < 1e-3, row  # no uniform scale error
