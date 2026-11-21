# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of reshape (verifier).

reshape moves values and computes nothing (spec DEFINITION, rule R1): every numeric entry of the specification
resolves to `exact`, and its legacy default is (PCC >= 0.9999, relative RMS <= 0.0). This test measures PCC, max /
mean abs error, relative RMS, ULP p99 and the got/true ratio spread (the scale-bug detector) on shapes that cover
both regimes and both placements, and gates on exactness.

Output under pytest-xdist / --forked is not shown, so each case records its metrics with `record_property` and
writes them to $RESHAPE_PRECISION_OUT (default /tmp/reshape_precision_baseline/<case>.json).
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from eval.metrics import compute_metrics_torch
from models.common.metrics import comp_allclose, compute_pcc
from ttnn.operations.reshape import reshape
from ttnn.operations.reshape.reshape_program_artifacts import select_regime

TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

CASES = [
    # (id, shape, new shape, layout, memory)
    ("small_regroup_l1", (1, 64, 400), (1, 4, 16, 400), TILE, L1),  # traced call, tile_row_regroup
    ("medium_identity_rm", (1, 1, 1600, 144), (1, 1600, 144), RM, DRAM),  # traced call, page_identity
    ("large_regroup_dram", (1, 40, 40, 512), (1, 1, 1600, 512), TILE, DRAM),  # tile_row_regroup, 32 clusters
    ("large_identity_rm_wide", (1, 64, 8400), (1, 4, 16, 8400), RM, DRAM),  # page_identity, 16,800 B pages
]


def _ratio_spread(actual, expected):
    """Median and p5/p95 of actual/expected over finite, non-zero reference elements."""
    a, e = actual.double().flatten(), expected.double().flatten()
    mask = torch.isfinite(e) & torch.isfinite(a) & (e != 0)
    r = a[mask] / e[mask]
    if r.numel() == 0:
        return {"median": float("nan"), "p5": float("nan"), "p95": float("nan")}
    q = torch.quantile(r[: 1 << 24].float(), torch.tensor([0.05, 0.5, 0.95]))
    return {"p5": q[0].item(), "median": q[1].item(), "p95": q[2].item()}


@pytest.mark.parametrize("case_id,shape,new_shape,layout,mem", CASES, ids=[c[0] for c in CASES])
def test_reshape_precision_baseline(device, record_property, case_id, shape, new_shape, layout, mem):
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
    got = ttnn.to_torch(reshape(x, new_shape))
    expected = torch.reshape(t, new_shape)

    m = compute_metrics_torch(got, expected, readback_dtype=torch.bfloat16)
    pcc = compute_pcc(got.float(), expected.float())
    _, allclose_msg = comp_allclose(got.float(), expected.float())
    abs_err = (got.float() - expected.float()).abs()
    result = {
        "case": case_id,
        "shape": list(shape),
        "new_shape": list(new_shape),
        "layout": str(layout),
        "memory": str(mem.buffer_type),
        "regime": select_regime(shape, new_shape, layout),
        "pcc": pcc,
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "relative_rms": m.rms,
        "ulp_p99": m.ulp_p99,
        "ratio": _ratio_spread(got, expected),
        "allclose": allclose_msg,
        "bit_exact": bool(torch.equal(got, expected)),
    }
    for k, v in result.items():
        record_property(k, v)
    out_dir = Path(os.environ.get("RESHAPE_PRECISION_OUT", "/tmp/reshape_precision_baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{case_id}.json").write_text(json.dumps(result, indent=2, default=str))

    # Gate: PCC and relative RMS together (eval/metrics.py check_output), at the specification's limits for an
    # exact op (legacy_default = (0.9999, 0.0); NUMERIC_RESOLVED: exact for every dtype).
    assert pcc >= 0.9999, result
    assert m.rms == 0.0, result
    assert result["bit_exact"], result
