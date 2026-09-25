#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compare dynamic consumption, execution time, and total consumption (charge) across
multiple named runs (use cases), grouped by grid (core combination), using each run's
program_intervals.csv (produced by parser.py / auto.py).

Usage:
    python3 compare_runs2.py -i <dir_with_run_subdirs> -c <cases.txt>

<cases.txt> format (one "Case N: subdir_name" or "Case N: subdir_name : Legend label" per
line):
    Case 1: compute_idle : Compute disabled
    Case 2: reader_idle2 : Reader Idle
    Case 3: regular : Writer idle
    Case 4: writer_amp : W+R+C active

Each "subdir_name" must be a subdirectory of <dir_with_run_subdirs> containing a
program_intervals.csv file (i.e. the output of a parser.py run with --program-log). The
optional third field is the label shown in the chart legends; if omitted, subdir_name is
used as the legend label instead.

Outputs (written to <dir_with_run_subdirs>/compare_runs_out/):
    dynamic_consumption_avg.png   -- dynamic_charge_avg_mah per grid, per use case
    dynamic_consumption_peak.png  -- dynamic_charge_peak_mah per grid, per use case
    dynamic_current_avg.png       -- dynamic_avg_current_a per grid, per use case
    dynamic_current_peak.png      -- dynamic_peak_current_a per grid, per use case
    execution_time.png            -- algo_time_s_reported per grid, per use case
    total_consumption.png         -- charge_avg_mah per grid, per use case
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RE_CASE_LINE = re.compile(r"^\s*Case\s+\d+\s*:\s*(?P<name>\S+)\s*(?::\s*(?P<label>.+?)\s*)?$")


def parse_cases_file(path: Path) -> List[Tuple[str, str]]:
    """Returns (subdir_name, legend_label) pairs, in the order they appear in the file.

    legend_label defaults to subdir_name when the optional third field is absent.
    """
    cases: List[Tuple[str, str]] = []
    with open(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            m = RE_CASE_LINE.match(line)
            if not m:
                print(
                    f"WARNING: {path}:{line_no}: skipping unrecognized line: {raw_line.rstrip()!r}",
                    file=sys.stderr,
                )
                continue
            name = m.group("name")
            label = m.group("label") or name
            cases.append((name, label))
    if not cases:
        raise ValueError(f"No valid 'Case N: subdir_name' lines found in {path}")
    names = [name for name, _ in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate subdirectory names in {path}: {names}")
    return cases


def load_program_intervals(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f, skipinitialspace=True))


def build_grid_order(rows_by_case: Dict[str, List[Dict[str, str]]]) -> List[str]:
    """Union of all grids seen across cases, ordered by core count then grid name."""
    cores_by_grid: Dict[str, int] = {}
    for rows in rows_by_case.values():
        for row in rows:
            grid = row["grid"]
            if grid not in cores_by_grid:
                cores_by_grid[grid] = int(row["cores"])
    return sorted(cores_by_grid.keys(), key=lambda g: (cores_by_grid[g], g))


def extract_metric_per_grid(rows: List[Dict[str, str]], column: str) -> Dict[str, float]:
    per_grid: Dict[str, float] = {}
    for row in rows:
        grid = row["grid"]
        raw = row.get(column, "")
        try:
            per_grid[grid] = float(raw)
        except (TypeError, ValueError):
            per_grid[grid] = float("nan")
    return per_grid


def grouped_bar_chart(
    grid_order: List[str],
    cases: List[Tuple[str, str]],
    values_by_case: Dict[str, Dict[str, float]],
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    n_cases = len(cases)
    n_grids = len(grid_order)
    x = np.arange(n_grids)
    width = 0.8 / max(n_cases, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_grids * 1.0), 6))
    for i, (name, label) in enumerate(cases):
        vals = [values_by_case[name].get(grid, float("nan")) for grid in grid_order]
        offset = (i - (n_cases - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(grid_order, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# (csv column, y-axis label, output filename, chart title)
METRICS: List[Tuple[str, str, str, str]] = [
    (
        "dynamic_charge_avg_mah",
        "Dynamic avg charge [mAh]",
        "dynamic_consumption_avg.png",
        "Dynamic consumption (avg charge, baseline removed) per grid, per use case",
    ),
    (
        "dynamic_charge_peak_mah",
        "Dynamic peak charge [mAh]",
        "dynamic_consumption_peak.png",
        "Dynamic consumption (peak charge, baseline removed) per grid, per use case",
    ),
    (
        "dynamic_avg_current_a",
        "Dynamic avg current [A]",
        "dynamic_current_avg.png",
        "Dynamic current (avg, baseline removed) per grid, per use case",
    ),
    (
        "dynamic_peak_current_a",
        "Dynamic peak current [A]",
        "dynamic_current_peak.png",
        "Dynamic current (peak, baseline removed) per grid, per use case",
    ),
    (
        "algo_time_s_reported",
        "Execution time [s]",
        "execution_time.png",
        "Execution time per grid, per use case",
    ),
    (
        "charge_avg_mah",
        "Total consumption (avg charge) [mAh]",
        "total_consumption.png",
        "Total consumption (charge) per grid, per use case",
    ),
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing the run subdirectories (each with a program_intervals.csv)",
    )
    parser.add_argument(
        "-c",
        "--cases-file",
        required=True,
        type=Path,
        help="Text file with 'Case N: subdir_name' lines mapping use cases to run subdirectories",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"ERROR: input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    if not args.cases_file.exists():
        print(f"ERROR: cases file not found: {args.cases_file}", file=sys.stderr)
        return 1

    try:
        cases = parse_cases_file(args.cases_file)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    rows_by_case: Dict[str, List[Dict[str, str]]] = {}
    for name, _label in cases:
        csv_path = args.input_dir / name / "program_intervals.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found (referenced by case {name!r})", file=sys.stderr)
            return 1
        rows_by_case[name] = load_program_intervals(csv_path)

    grid_order = build_grid_order(rows_by_case)
    print(f"Use cases (in order): {cases}")
    print(f"Grid combinations found: {grid_order}")

    out_dir = args.input_dir / "compare_runs_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    for column, ylabel, filename, title in METRICS:
        values_by_case = {name: extract_metric_per_grid(rows_by_case[name], column) for name, _label in cases}
        for name, _label in cases:
            missing = [g for g in grid_order if g not in values_by_case[name]]
            if missing:
                print(
                    f"WARNING: case {name!r} has no {column!r} data for grid(s) {missing} "
                    "-- left as a gap in the chart",
                    file=sys.stderr,
                )
        grouped_bar_chart(grid_order, cases, values_by_case, title, ylabel, out_dir / filename, args.dpi)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
