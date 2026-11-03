#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Every issue gap a headroom survey saw, per instruction pair, across all kernels.

Reads <survey>/<module>/<op>/perf_<RUN_TYPE>.json and verdict.json (complete captures
only), takes the bottleneck thread's gaps ("A->B": B could not issue the cycle after A)
and aggregates them per pair:

    pair | kernels | cycles per gap | max cyc/tile (worst kernel) | sum cyc/tile

A pair whose gap length is the same in every kernel is a fixed hardware cost of issuing B
after A (structural or result latency) — the evidence to check a latency table against.

    python codegen/scripts/perf_gap_summary.py <survey-dir> [--min-cyc-per-tile 1] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ASSEMBLY = _HERE.parents[1] / "tt_llk_quasar" / "instructions" / "assembly.yaml"


def _name(opcode: str, names: dict[str, str]) -> str:
    return names.get(opcode, opcode)


def summarize(root: Path, names: dict[str, str]) -> list[dict]:
    pairs: dict[str, dict] = defaultdict(
        lambda: {
            "kernels": 0,
            "count": 0,
            "cycles": 0,
            "sum": 0.0,
            "max": 0.0,
            "worst": "",
        }
    )
    for report_path in sorted(root.glob("*/*/perf_*.json")):
        kernel_dir = report_path.parent
        verdict_path = kernel_dir / "verdict.json"
        report = json.loads(report_path.read_text())
        if (
            report.get("status") not in ("complete", "missing_signals")
            or not verdict_path.is_file()
        ):
            continue
        verdict = json.loads(verdict_path.read_text())
        per_tile = verdict["reconcile"]["wave_per_tile"]
        tiles = report["window"]["cycles"] / per_tile if per_tile else 0
        thread = report["threads"].get(verdict["bottleneck"]) or {}
        kernel = f"{kernel_dir.parent.name}/{kernel_dir.name}"
        for gap in thread.get("bubbles") or []:
            pair = f"{_name(gap['after'], names)}->{_name(gap['before'], names)}"
            p = pairs[pair]
            p["kernels"] += 1
            p["count"] += gap["count"]
            p["cycles"] += gap["cycles"]
            cpt = gap["cycles"] / tiles if tiles else 0.0
            p["sum"] += cpt
            if cpt > p["max"]:
                p["max"], p["worst"] = cpt, kernel
    rows = [
        {
            "pair": pair,
            "kernels": p["kernels"],
            "cycles_per_gap": round(p["cycles"] / p["count"], 2) if p["count"] else 0.0,
            "max_cyc_per_tile": round(p["max"], 3),
            "worst_kernel": p["worst"],
            "sum_cyc_per_tile": round(p["sum"], 3),
        }
        for pair, p in pairs.items()
    ]
    return sorted(rows, key=lambda r: (-r["sum_cyc_per_tile"], r["pair"]))


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| pair | kernels | cycles per gap | max cyc/tile | worst kernel | sum cyc/tile |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['pair']} | {r['kernels']} | {r['cycles_per_gap']} | {r['max_cyc_per_tile']} | "
            f"{r['worst_kernel']} | {r['sum_cyc_per_tile']} |"
        )
    return "\n".join(lines) + "\n"


def _load_names() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "perf_headroom", _HERE / "perf_headroom.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_opcode_names(_ASSEMBLY) if _ASSEMBLY.exists() else {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("survey", type=Path)
    p.add_argument(
        "--min-cyc-per-tile",
        type=float,
        default=1.0,
        help="hide pairs below this worst-kernel cost",
    )
    p.add_argument("--csv", type=Path)
    args = p.parse_args(argv)
    rows = [
        r
        for r in summarize(args.survey, _load_names())
        if r["max_cyc_per_tile"] >= args.min_cyc_per_tile
    ]
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["pair"])
            w.writeheader()
            w.writerows(rows)
    print(to_markdown(rows), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
