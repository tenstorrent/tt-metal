# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of to_layout (verifier).

to_layout moves values and computes nothing; every cell of the specification is graded `exact` (rule R1,
NUMERIC_RESOLVED). This test measures, per (shape, regime): PCC, max / mean abs error, relative RMS error
(RMS error / std of the reference, as eval/metrics.check_output), max bfloat16 ULP distance, the got/true
ratio spread (scale-bug detector) and bit equality. It gates on bit equality (the spec's requirement), and
on PCC == 1 and relative RMS == 0.

Test output is hidden under the simulator's workers / forking, so each case writes its numbers to
generated/to_layout_precision_baseline/<case>.json (read by the verification report).
"""

import json
import math
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.to_layout import to_layout

TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
OUT_DIR = Path("generated/to_layout_precision_baseline")

SHAPES = [
    (1, 1, 32, 64),  # small, one block
    (1, 1, 400, 512),  # YOLOv8 call, h_non_aligned, column split (bw = 4)
    (1, 1, 1600, 256),  # YOLOv8 call, tile_aligned
    (2, 4, 128, 512),  # larger, batched, full-width blocks (bw = 16)
]
REGIMES = {"tilize": (RM, TILE), "untilize": (TILE, RM), "copy_tile": (TILE, TILE), "copy_rm": (RM, RM)}


def _bf16_ordered(x):
    """bfloat16 bit patterns as integers whose difference is the ULP distance."""
    bits = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _metrics(got, ref):
    g, r = got.double(), ref.double()
    err = g - r
    rms = err.pow(2).mean().sqrt().item()
    std = r.std().item()
    nz = (r != 0) & torch.isfinite(r) & torch.isfinite(g)
    ratio = (g[nz] / r[nz]).flatten()
    _, pcc = comp_pcc(ref.float(), got.float())
    allclose_ok, allclose_msg = comp_allclose(ref.float(), got.float(), rtol=0.0, atol=0.0)
    return {
        "pcc": float(pcc),
        "allclose_exact": bool(allclose_ok),
        "allclose": allclose_msg,
        "max_abs_err": err.abs().max().item(),
        "mean_abs_err": err.abs().mean().item(),
        "rel_rms_err": rms / std if std > 0 else (0.0 if rms == 0 else math.inf),
        "max_ulp": (_bf16_ordered(got) - _bf16_ordered(ref)).abs().max().item(),
        "ratio_median": ratio.median().item(),
        "ratio_p5": ratio.quantile(0.05).item(),
        "ratio_p95": ratio.quantile(0.95).item(),
        "bit_equal": bool(torch.equal(got.view(torch.int16), ref.view(torch.int16))),
    }


@pytest.mark.parametrize("regime", list(REGIMES))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(device, shape, regime):
    torch.manual_seed(1234)
    src, dst = REGIMES[regime]
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=src, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = to_layout(t, dst)
    assert out.layout == dst and out.dtype == ttnn.bfloat16
    got = ttnn.to_torch(out)
    assert tuple(got.shape) == shape

    m = _metrics(got, x)
    m.update({"shape": list(shape), "regime": regime})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{'x'.join(map(str, shape))}_{regime}.json").write_text(json.dumps(m, indent=1))

    assert m["bit_equal"], m
    assert m["pcc"] >= 1.0 - 1e-12 and m["rel_rms_err"] == 0.0 and m["max_ulp"] == 0, m
