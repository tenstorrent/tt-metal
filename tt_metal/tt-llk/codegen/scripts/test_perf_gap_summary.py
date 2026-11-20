# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""perf_gap_summary.py: every issue gap seen across a survey, per instruction pair."""

from __future__ import annotations

import json
from pathlib import Path

import perf_gap_summary as gaps
import perf_headroom as ph

NAMES = {
    "0x99": "SFPNONLINEAR",
    "0x9a": "SFPNONLINEAR",
    "0x72": "SFPSTORE",
    "0x70": "SFPLOAD",
    "0x37": "SETRWC",
}
RATES = {"default": {"cycles": 1, "source": "s"}, "opcodes": {}}
TILES = 128


def _thread(window, bubbles):
    gap_cycles = sum(c for *_, c in bubbles)
    return {
        "issue": window - gap_cycles,
        "stall": {},
        "idle": gap_cycles,
        "unknown": 0,
        "instructions_issued": window - gap_cycles,
        "instructions_by_opcode": {},
        "bubbles": [
            dict(after=a, before=b, count=n, cycles=c) for a, b, n, c in bubbles
        ],
    }


def _kernel(root: Path, module: str, op: str, per_tile: float, bubbles):
    """A survey entry as perf_diagnose leaves it: a real verdict.json."""
    d = root / module / op
    d.mkdir(parents=True)
    window = int(per_tile * TILES)
    report = {
        "status": "complete",
        "window": {"cycles": window},
        "threads": {
            "math": _thread(window, bubbles),
            "unpack": _thread(window, [("0x37", "0x70", 99, 99)]),
        },
    }
    row = {"loop_factor": "32", "tile_cnt": "4", "mean(MATH_ISOLATE)": str(per_tile)}
    verdict = ph.headroom_verdict(
        report, row, RATES, run_type="MATH_ISOLATE", opcode_names=NAMES
    )
    (d / "verdict.json").write_text(json.dumps(verdict))


def test_pairs_are_aggregated_over_kernels_on_the_bottleneck_thread(tmp_path):
    _kernel(tmp_path, "m", "Sigmoid", 267.0, [("0x99", "0x72", 4096, 4096)])
    _kernel(
        tmp_path,
        "m",
        "Exp",
        170.0,
        [("0x37", "0x70", 256, 3584), ("0x99", "0x72", 128, 128)],
    )
    rows = gaps.summarize(tmp_path)
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


def test_a_kernel_counts_once_per_pair(tmp_path):
    # Two hex opcodes with one mnemonic: one named pair, one kernel, summed cycles.
    _kernel(
        tmp_path,
        "m",
        "Sigmoid",
        267.0,
        [("0x99", "0x72", 2048, 2048), ("0x9a", "0x72", 1024, 2048)],
    )
    (row,) = gaps.summarize(tmp_path)
    assert row["kernels"] == 1
    assert row["max_cyc_per_tile"] == 32.0
    assert row["sum_cyc_per_tile"] == 32.0
    assert row["cycles_per_gap"] == 1.33


def test_entries_without_a_verdict_are_skipped(tmp_path):
    (tmp_path / "m" / "Cos").mkdir(parents=True)
    (tmp_path / "m" / "Cos" / "diag.json").write_text(json.dumps({"status": "failed"}))
    assert gaps.summarize(tmp_path) == []


def test_markdown_table(tmp_path):
    _kernel(tmp_path, "m", "Sigmoid", 267.0, [("0x99", "0x72", 4096, 4096)])
    md = gaps.to_markdown(gaps.summarize(tmp_path))
    assert "| SFPNONLINEAR->SFPSTORE | 1 | 1.0 | 32.0 | m/Sigmoid |" in md
