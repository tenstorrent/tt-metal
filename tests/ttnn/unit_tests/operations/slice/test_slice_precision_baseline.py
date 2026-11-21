# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of slice (verifier, Phase 0).

slice moves values and computes nothing (spec R1), so the numeric requirement of every cell is `exact`
(spec NUMERIC_RESOLVED). This test measures what the op actually delivers on supported cells: PCC, max /
mean absolute error, relative RMS (RMS error / std of the reference, as `eval.metrics.check_output`), and
the got/true ratio spread (the scale-bug detector). Gate: bit-exact, which implies PCC = 1 and relative
RMS = 0.

Under the simulator's forked workers a test's output is not shown: each measurement is written to
`slice_precision_baseline.jsonl` in the temp directory and recorded with `record_property`.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.slice import slice as ttnn_slice

RESULTS = Path(tempfile.gettempdir()) / "slice_precision_baseline.jsonl"

# (shape, start, end, layout, memory config): small, medium, the YOLO calls (one per layout and placement),
# and a larger one whose row-major pages exceed one NoC packet (the explicit-sync ring path).
CASES = [
    ("small_tile_dram", (1, 64, 128), (0, 0, 0), (1, 32, 128), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    ("medium_tile_dram", (4, 128, 512), (0, 32, 0), (4, 96, 512), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    ("yolo_tile_l1", (1, 144, 400), (0, 64, 0), (1, 144, 400), ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG),
    ("medium_rm_l1_wna", (4, 128, 47), (0, 64, 0), (4, 100, 47), ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
    ("large_yolo_rm_dram", (1, 144, 8400), (0, 64, 0), (1, 144, 8400), ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
]


def _measure(got, expected):
    got64, exp64 = got.double(), expected.double()
    _, pcc = comp_pcc(exp64, got64)
    _, allclose = comp_allclose(got64, exp64)
    err = (got64 - exp64).abs()
    std = exp64.std().item()
    rel_rms = ((got64 - exp64).pow(2).mean().sqrt().item() / std) if std > 0 else 0.0
    mask = torch.isfinite(exp64) & (exp64 != 0)
    ratio = got64[mask] / exp64[mask]
    q = torch.quantile(ratio.float(), torch.tensor([0.05, 0.5, 0.95])).tolist()
    return {
        "pcc": float(pcc),
        "max_abs_err": err.max().item(),
        "mean_abs_err": err.mean().item(),
        "rel_rms": rel_rms,
        "allclose": allclose,
        "ratio_p5": q[0],
        "ratio_median": q[1],
        "ratio_p95": q[2],
        "ratio_std": ratio.std().item(),
        "bit_exact": bool(torch.equal(got, expected)),
    }


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_slice_precision_baseline(device, case, record_property):
    name, shape, start, end, layout, mem = case
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
    got = ttnn.to_torch(ttnn_slice(x, list(start), list(end)))
    expected = t[tuple(slice(s, e) for s, e in zip(start, end))]
    assert list(got.shape) == list(expected.shape)

    m = _measure(got, expected)
    m.update(case=name, shape=list(shape), start=list(start), end=list(end))
    for k, v in m.items():
        record_property(k, v)
    with RESULTS.open("a") as f:
        f.write(json.dumps(m) + "\n")

    # Gate: PCC and relative RMS together (as check_output), at the `exact` requirement.
    assert m["pcc"] >= 0.9999 and m["rel_rms"] == 0.0, m
    assert m["bit_exact"], m
    # A uniform scale bug keeps PCC high; an exact copy has every ratio at 1.
    assert m["ratio_p5"] == m["ratio_median"] == m["ratio_p95"] == 1.0, m

