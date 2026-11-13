# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""perf_gap_summary.py: every issue gap seen across a survey, per instruction pair."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).parent / "perf_gap_summary.py"
_SPEC = importlib.util.spec_from_file_location("perf_gap_summary", _PATH)
gaps = importlib.util.module_from_spec(_SPEC)
sys.modules["perf_gap_summary"] = gaps
_SPEC.loader.exec_module(gaps)

NAMES = {
    "0x99": "SFPNONLINEAR",
    "0x72": "SFPSTORE",
    "0x70": "SFPLOAD",
    "0x37": "SETRWC",
}


def _kernel(
    root: Path, module: str, op: str, per_tile: float, bubbles, status="complete"
):
    d = root / module / op
    d.mkdir(parents=True)
    tiles = 128
    report = {
        "status": status,
        "window": {"cycles": int(per_tile * tiles)},
        "threads": {
            "math": {
                "bubbles": [
                    dict(after=a, before=b, count=n, cycles=c) for a, b, n, c in bubbles
                ]
            },
            "unpack": {
                "bubbles": [dict(after="0x37", before="0x70", count=999, cycles=999)]
            },
        },
    }
    (d / "perf_MATH_ISOLATE.json").write_text(json.dumps(report))
    (d / "verdict.json").write_text(
        json.dumps(
            {"bottleneck": "math", "reconcile": {"wave_per_tile": per_tile, "ok": True}}
        )
    )
    (d / "diag.json").write_text(
        json.dumps({"status": "ok" if status == "complete" else "failed"})
    )


def test_pairs_are_aggregated_over_kernels_on_the_bottleneck_thread(tmp_path):
    _kernel(tmp_path, "m", "Sigmoid", 267.0, [("0x99", "0x72", 4096, 4096)])
    _kernel(
        tmp_path,
        "m",
        "Exp",
        170.0,
        [("0x37", "0x70", 256, 3584), ("0x99", "0x72", 128, 128)],
    )
    rows = gaps.summarize(tmp_path, NAMES)
    by_pair = {r["pair"]: r for r in rows}
    nl = by_pair["SFPNONLINEAR->SFPSTORE"]
    assert nl["kernels"] == 2
    assert nl["cycles_per_gap"] == 1.0
    assert nl["max_cyc_per_tile"] == 32.0 and nl["worst_kernel"] == "m/Sigmoid"
    rwc = by_pair["SETRWC->SFPLOAD"]
    assert rwc["cycles_per_gap"] == 14.0 and rwc["kernels"] == 1
    assert rwc["sum_cyc_per_tile"] == 28.0, "unpack-thread gaps are not counted"
    assert (
        rows[0]["pair"] == "SFPNONLINEAR->SFPSTORE"
    ), "ranked by total cycles per tile across kernels"


def test_incomplete_captures_are_skipped(tmp_path):
    _kernel(
        tmp_path,
        "m",
        "Cos",
        300.0,
        [("0x99", "0x72", 10, 10)],
        status="capture_clipped",
    )
    assert gaps.summarize(tmp_path, NAMES) == []


def test_markdown_table(tmp_path):
    _kernel(tmp_path, "m", "Sigmoid", 267.0, [("0x99", "0x72", 4096, 4096)])
    md = gaps.to_markdown(gaps.summarize(tmp_path, NAMES))
    assert "| SFPNONLINEAR->SFPSTORE | 1 | 1.0 | 32.0 | m/Sigmoid |" in md
