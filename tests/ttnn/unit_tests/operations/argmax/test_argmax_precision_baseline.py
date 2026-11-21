# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of argmax (verifier, Phase 0).

argmax is graded `exact` (NUMERIC_RESOLVED: metric ['exact'], ties lowest_index): an index equals the
float64 reference's or it is wrong. PCC, abs error and relative RMS of the indices are measured as the
verifier protocol asks, and are expected to be 1.0 / 0 / 0; they say nothing finer than "exact". The
figures that carry meaning are:

  exact_fraction  share of rows whose index equals torch.argmax on the stored values;
  value_ulp_max   bf16 ULP distance between x[row, got] and the row's maximum (0 for every exact index;
                  also 0 for a wrong index that lands on a tied maximum, which `exact` would still fail);
  ratio median / p5 / p95 of got/true over rows with a non-zero true index (the scale-bug detector:
                  a tight cluster off 1.0 is a structural bug, a spread around 1.0 is noise).

Each shape is one regime of the op: ROWS on one row per scanner, ROWS with several rows per scanner, and
SEGMENTED on both model shapes. The figures are written to $ARGMAX_PRECISION_OUT (default
/tmp/argmax_precision_baseline) as one JSON per shape, since a forked simulator worker shows no output.
"""

import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.argmax import argmax

SHAPES = [
    (1, 1, 32, 64),  # ROWS, one row per scanner
    (4, 8, 32, 256),  # ROWS, 5-6 rows per scanner
    (1, 1, 32, 4096),  # SEGMENTED, spr = 4
    (1, 1, 2, 151936),  # SEGMENTED, spr = 96 (the model's vocabulary row)
]

PCC_MIN = 0.9999  # NUMERIC["legacy_default"]
REL_RMS_MAX = 0.0


def _bf16_ordinal(v):
    """Monotone integer of a bf16 value: the distance between two ordinals is their ULP distance."""
    bits = v.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    neg = (bits & 0x8000) != 0
    mag = bits & 0x7FFF
    return torch.where(neg, -mag, mag)


@pytest.mark.parametrize("shape", SHAPES, ids=["x".join(map(str, s)) for s in SHAPES])
def test_argmax_precision_baseline(device, shape, record_property):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    out = argmax(t, dim=-1)
    got = ttnn.to_torch(out).to(torch.int64).reshape(shape[:-1])
    expected = torch.argmax(x.double(), dim=-1)

    got_d, exp_d = got.double(), expected.double()
    _, pcc = comp_pcc(exp_d, got_d, PCC_MIN)
    _, allclose_msg = comp_allclose(exp_d, got_d)
    err = (got_d - exp_d).abs()
    std = exp_d.std().item() if exp_d.numel() > 1 else 0.0
    rms = ((got_d - exp_d) ** 2).mean().sqrt().item()
    rel_rms = rms / std if std > 0 else (0.0 if rms == 0 else float("inf"))

    rows = x.reshape(-1, shape[-1])
    picked = rows.gather(1, got.reshape(-1, 1).clamp(0, shape[-1] - 1)).squeeze(1)
    value_ulp = (_bf16_ordinal(picked) - _bf16_ordinal(rows.max(dim=1).values)).abs()

    nz = exp_d != 0
    ratio = got_d[nz] / exp_d[nz]
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], dtype=ratio.dtype)).tolist() if ratio.numel() else [1, 1, 1]

    result = {
        "shape": list(shape),
        "rows": int(got.numel()),
        "exact_fraction": float((got == expected).double().mean().item()),
        "pcc": float(pcc),
        "max_abs_err": float(err.max().item()),
        "mean_abs_err": float(err.mean().item()),
        "rel_rms": float(rel_rms),
        "comp_allclose": allclose_msg,
        "value_ulp_max": int(value_ulp.max().item()),
        "ratio_p5": q[0],
        "ratio_median": q[1],
        "ratio_p95": q[2],
    }
    out_dir = Path(os.environ.get("ARGMAX_PRECISION_OUT", "/tmp/argmax_precision_baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{'x'.join(map(str, shape))}.json").write_text(json.dumps(result, indent=1))
    for k, v in result.items():
        record_property(k, v)

    assert result["exact_fraction"] == 1.0, result
    assert result["value_ulp_max"] == 0, result
    assert pcc >= PCC_MIN and rel_rms <= REL_RMS_MAX, result
