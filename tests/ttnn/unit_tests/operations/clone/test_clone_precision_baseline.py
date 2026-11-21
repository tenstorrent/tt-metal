# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of clone (verifier, Phase 0).

clone moves bytes and computes nothing (spec R1), so the expected result is a bit-exact copy: PCC 1,
every abs error 0, relative RMS 0, ULP distance 0, and a got/true ratio of exactly 1 everywhere.
The gate is the specification's own: `NUMERIC["legacy_default"]` = (PCC >= 0.9999, relative RMS <= 0.0),
plus a bit-for-bit equality check, which is what R1 actually asks for.

Each case writes its measurements to `$CLONE_PRECISION_DIR/<id>.json` (default
/tmp/clone_precision_baseline): under the simulator's pytest workers a test's stdout is not shown.
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from eval.metrics import compute_metrics_torch
from eval.specs.clone.spec import NUMERIC
from models.common.metrics import comp_allclose, compute_pcc
from ttnn.operations.clone import clone

PCC_MIN, RMS_MAX = NUMERIC["legacy_default"]
OUT_DIR = Path(os.environ.get("CLONE_PRECISION_DIR", "/tmp/clone_precision_baseline"))

CASES = [
    # (shape, layout, input memory) — small / medium / the largest traced call / a wide non-aligned RM
    ((1, 1, 400, 64), ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),  # traced YOLOv8 call (RM in L1)
    ((2, 1, 100, 47), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # both dims non-aligned, padded tiles
    ((1, 1, 1600, 512), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # largest traced call
    ((1, 32, 4096), ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # wide rows (8 KiB sticks)
]


def _ratio_spread(actual, expected):
    a = actual.to(torch.float64).flatten()
    e = expected.to(torch.float64).flatten()
    mask = torch.isfinite(a) & torch.isfinite(e) & (e != 0)
    r = a[mask] / e[mask]
    return {
        "median": r.median().item(),
        "p5": torch.quantile(r, 0.05).item(),
        "p95": torch.quantile(r, 0.95).item(),
        "std": r.std().item(),
    }


@pytest.mark.parametrize(
    "shape,layout,in_mem",
    CASES,
    ids=["rm_1x1x400x64_l1", "tile_2x1x100x47_dram", "tile_1x1x1600x512_dram", "rm_1x32x4096_dram"],
)
def test_clone_precision_baseline(device, shape, layout, in_mem, request):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)  # NUMERIC["inputs"]: standard normal
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_mem)

    out = clone(tt_in, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    got = ttnn.to_torch(out)

    m = compute_metrics_torch(got, x.to(torch.float64), readback_dtype=got.dtype)
    pcc = compute_pcc(got, x)
    allclose_ok, allclose_msg = comp_allclose(got, x, rtol=0.0, atol=0.0)
    bitwise = torch.equal(got.view(torch.int16), x.view(torch.int16))
    max_ulp = (got.view(torch.int16).to(torch.int32) - x.view(torch.int16).to(torch.int32)).abs().max().item()
    ratio = _ratio_spread(got, x)

    record = {
        "id": request.node.callspec.id,
        "shape": list(shape),
        "pcc": pcc,
        "pcc_eval": m.pcc,
        "max_abs_err": m.max_abs_diff,
        "mean_abs_err": (got.to(torch.float64) - x.to(torch.float64)).abs().mean().item(),
        "rel_rms": m.rms,
        "ulp_p99": m.ulp_p99,
        "max_ulp_bits": max_ulp,
        "bitwise_equal": bitwise,
        "allclose_exact": allclose_ok,
        "allclose_msg": allclose_msg,
        "ratio": ratio,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{record['id']}.json").write_text(json.dumps(record, indent=1))
    for k in ("pcc", "max_abs_err", "mean_abs_err", "rel_rms", "max_ulp_bits"):
        request.node.user_properties.append((k, record[k]))

    assert bitwise, f"clone is not bit-exact: {record}"
    assert pcc >= PCC_MIN and m.rms <= RMS_MAX, record
    assert ratio["median"] == 1.0 and ratio["p5"] == 1.0 and ratio["p95"] == 1.0, record
