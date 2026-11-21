# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""concat: precision baseline (verifier).

concat moves values and computes nothing (spec R1; NUMERIC_RESOLVED grades bfloat16 `exact`), so the expected
baseline is PCC 1, zero absolute error, zero relative RMS, zero ULP and a got/true ratio of exactly 1 everywhere.
The test measures all of them on shapes that cover both regimes (page_copy and the three run_assembly
granularities) and one larger shape of each, and gates on exactness.

Output under forked workers is not shown: set CONCAT_BASELINE_OUT=<file> to get one JSON line per shape.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, comp_ulp
from ttnn.operations.concat import concat
from ttnn.operations.concat.concat_program_artifacts import select_regime

BASELINE = [
    (3, (1, 1, 64, 64), (1, 1, 64, 80)),  # YOLOv8 cv2 + cv3: page_copy
    (1, (1, 64, 144), (1, 16, 144), (1, 4, 144)),  # YOLOv8 three scales: run_assembly row-granular
    (1, (1, 2, 400), (1, 2, 400)),  # YOLOv8 dist2bbox: run_assembly, join inside a tile
    (-1, (4, 8, 32, 47), (4, 8, 32, 100), (4, 8, 32, 50)),  # run_assembly element-granular, 224 tiles
    (-1, (1, 1, 32, 4096), (1, 1, 32, 4096)),  # wide page_copy
    (1, (2, 512, 1024), (2, 512, 1024)),  # tall page_copy, 2048 tiles
]


def _id(entry):
    dim, *shapes = entry
    return "+".join("x".join(map(str, s)) for s in shapes) + f"_dim{dim}"


def _metrics(got, expected):
    g, e = got.double(), expected.double()
    _, pcc = comp_pcc(e, g, pcc=1.0)
    _, allclose_msg = comp_allclose(e, g)
    ulp_ok, _ = comp_ulp(expected, got, 0)
    err = (g - e).abs()
    std = e.std().item()
    rel_rms = ((g - e).pow(2).mean().sqrt().item() / std) if std > 0 else 0.0
    nz = (e != 0) & torch.isfinite(e) & torch.isfinite(g)
    ratio = g[nz] / e[nz]
    return {
        "pcc": float(pcc),
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel_rms": rel_rms,
        "ulp0": bool(ulp_ok),
        "allclose": allclose_msg,
        "ratio_median": ratio.median().item(),
        "ratio_p5": ratio.quantile(0.05).item() if ratio.numel() < 2**24 else float("nan"),
        "ratio_p95": ratio.quantile(0.95).item() if ratio.numel() < 2**24 else float("nan"),
    }


@pytest.mark.parametrize("entry", BASELINE, ids=[_id(e) for e in BASELINE])
def test_precision_baseline(device, entry, record_property):
    dim, *shapes = entry
    torch.manual_seed(1234)
    host = [torch.randn(*s, dtype=torch.bfloat16) for s in shapes]
    dev = [ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16) for t in host]
    got = ttnn.to_torch(concat(dev, dim=dim))
    expected = torch.cat(host, dim=dim)
    assert got.shape == expected.shape

    m = _metrics(got, expected)
    canonical = dim if dim < 0 else dim - len(shapes[0])
    m.update({"case": _id(entry), "regime": select_regime(shapes, canonical)})
    for k, v in m.items():
        record_property(k, v)
    out = os.environ.get("CONCAT_BASELINE_OUT")
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(m) + "\n")

    # A copy: every element bit-identical (R1, NUMERIC exact).
    assert torch.equal(got, expected)
    assert m["pcc"] >= 0.99999 and m["rel_rms"] == 0.0 and m["max_abs"] == 0.0 and m["ulp0"]
    assert m["ratio_median"] == 1.0
