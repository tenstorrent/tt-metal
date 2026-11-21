# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""permute precision baseline: PCC, abs errors, relative RMS, ULP distance and the got/true ratio spread.

permute moves values and computes nothing (spec R1; NUMERIC_RESOLVED: `exact` for every dtype), so the expected
baseline is bit-exact in every regime. The gate is the spec's: exact equality. PCC and relative RMS are recorded
the way eval/metrics.py's check_output gates them, and the got/true ratio spread is the scale-bug detector.

Results go to $PERMUTE_PRECISION_OUT (default /tmp/permute_precision_baseline.jsonl) and record_property: a
test's stdout is not shown under the simulator's workers.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.permute import permute
from ttnn.operations.permute.permute_program_artifacts import canonical, select_regime

OUT = os.environ.get("PERMUTE_PRECISION_OUT", "/tmp/permute_precision_baseline.jsonl")

CASES = [
    # (shape, dims, layout): one per regime x layout, small to large, YOLOv8's traced calls first.
    ((1, 400, 144), (0, 2, 1), ttnn.TILE_LAYOUT),  # tile_transpose, YOLOv8 detect head
    ((1, 4, 16, 8400), (0, 1, 3, 2), ttnn.ROW_MAJOR_LAYOUT),  # staged_gather RM, YOLOv8 DFL
    ((2, 32, 64, 128), (0, 2, 1, 3), ttnn.TILE_LAYOUT),  # staged_gather TILE: whole rows move
    ((4, 8, 32, 256), (1, 0, 2, 3), ttnn.ROW_MAJOR_LAYOUT),  # page_gather RM
    ((2, 4, 128, 512), (0, 1, 3, 2), ttnn.TILE_LAYOUT),  # tile_transpose, larger
]


def _ulp_max(got, ref):
    """Largest distance in bfloat16 units in the last place (bit patterns as ordered integers)."""

    def ordered(x):
        b = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(b < 0, -(b & 0x7FFF), b)

    return int((ordered(got) - ordered(ref)).abs().max().item())


@pytest.mark.parametrize("shape,dims,layout", CASES, ids=lambda v: str(v))
def test_precision_baseline(device, shape, dims, layout, record_property):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    got = ttnn.to_torch(permute(t, dims))
    ref = x.permute(dims).contiguous()
    assert got.shape == ref.shape

    _, pcc = comp_pcc(ref.double(), got.double(), pcc=0.9999)
    _, allclose_msg = comp_allclose(ref.double(), got.double())
    err = (got.double() - ref.double()).abs()
    rel_rms = float(torch.sqrt((err**2).mean()) / ref.double().std())
    nz = (ref != 0) & torch.isfinite(ref)
    ratio = got.double()[nz] / ref.double()[nz]
    row = {
        "shape": list(shape),
        "dims": list(dims),
        "layout": str(layout),
        "regime": select_regime(layout, canonical(shape, [d % len(shape) for d in dims])[1]),
        "pcc": float(pcc),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "rel_rms": rel_rms,
        "ulp_max": _ulp_max(got, ref),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(ratio.quantile(0.05)),
        "ratio_p95": float(ratio.quantile(0.95)),
        "allclose": allclose_msg,
    }
    for k, v in row.items():
        record_property(k, v)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")

    # The specification's numeric requirement is `exact` (R1): every element moved bit for bit.
    assert torch.equal(got, ref), row
    assert row["ulp_max"] == 0 and rel_rms == 0.0 and pcc >= 0.9999, row
