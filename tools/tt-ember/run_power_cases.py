#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""
Top-level driver for the POWER_CASE power-experiment sweep.

For each requested POWER_CASE value (see tt_metal/programming_examples/long_matmul/README.md for
what each case disables or amplifies in the kernels), and for each requested --ops entry (which per-tile
instruction, FPU or SFPU, the compute kernel runs, via LONG_MATMUL_OP -- see compute.cpp), this script:

  1. Resets the hardware (`tt-smi -r`) and waits for it to re-initialize.
  2. Runs auto.py with POWER_CASE=<i> and LONG_MATMUL_OP=<op> exported into the application's
     environment, writing that case's results to <output-root>/<op>/<subdir>.
  3. After every requested case has completed for a given op, writes a cases file
     (`Case N: subdir : label`) describing the POWER_CASE -> subdir mapping for that op, and
     calls compare_cases.py against it to produce the comparison charts in
     <output-root>/<op>/compare_runs_out/.
  4. If more than one op was requested, also writes a cross-op cases file pairing every
     (op, case) combination -- e.g. "regular (matmul)" next to "regular (add)" -- and calls
     compare_cases.py once more to produce <output-root>/compare_runs_out/, so the two
     operations can be read off the same chart, case by case. Any (op, case) pair that is not
     a complete run with this sweep's settings (no run_complete.json, an incomplete run, or one
     made with other settings) is skipped with a note rather than failing the whole comparison.

This automates exactly the manual sequence, repeated once per op in --ops:
    tt-smi -r
    export POWER_CASE=<i> LONG_MATMUL_OP=<op>
    python3 auto.py --telemetry-exe ... --app-exe ... --output-root <output-root>/<op> \\
        --subdir <name-for-that-case> --app-args ...
run once per case, followed by:
    python3 compare_cases.py -i <output-root>/<op> -c <output-root>/<op>/power_cases.txt

Usage:
    python3 run_power_cases.py \\
        --telemetry-exe /path/to/telemetry \\
        --app-exe /path/to/metal_example_long_matmul \\
        --parser-script /path/to/parser.py \\
        --tt-venv-activate /path/to/tt-metal-venv/bin/activate \\
        --tt-metal-root /path/to/tt-metal \\
        --output-root /path/to/out_new \\
        --trim-ms 1.0 \\
        --app-args 1024 2048 2048 160
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

from common import (
    DEFAULT_DEVICE_ID,
    DEFAULT_DPI,
    DEFAULT_SLOT_MS,
    DEFAULT_TELEMETRY_FREQ_HZ,
    LONG_MATMUL_APP,
    POWER_CASES_TRIM_MS,
    RUN_COMPLETE_FILE,
    auto_py_command,
    environ_with,
    exit_reason,
    long_matmul_args,
    reset_hardware,
    run_settings,
    check_long_matmul_device,
    legacy_mode,
    misplaced_options,
    run_status,
    save_figure,
    shell_join,
    whole_run_value,
)
from compare_cases import build_grid_order, grouped_bar_chart

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# POWER_CASE -> (subdir name, legend label). See long_matmul's README for what each case does. Case 3 (reader AND compute
# both idle) is not in the default sweep below -- it's a two-variable case, and the three
# single-variable ablations already isolate each engine's own contribution to dynamic power.
# Case 5 (LONG_MATMUL_DISABLE_WRITER) is the writer's idle case, symmetric with case 2 (compute idle)
# and case 4 (reader idle): all three compare "full real work" against "zero real work" for
# exactly one engine, with the other two held identical (real, 100% write amplification) on
# both sides. Case 0 ("regular", write amplification off) was used as the writer stand-in
# before case 5 existed -- kept available (not in the default set) since it answers a different,
# still useful question: the marginal cost of turning on 100% write amplification specifically,
# as opposed to the writer's full on/off contribution. Add --power-cases 0 1 2 3 4 5 to include
# both of those plus the two-variable case.
POWER_CASE_SPECS: Dict[int, Tuple[str, str]] = {
    0: ("regular", "Regular (baseline)"),
    1: ("writer_amp", "All active"),
    2: ("compute_idle", "Compute idle"),
    3: ("reader_compute_idle", "Reader+Compute idle"),
    4: ("reader_idle2", "Reader idle"),
    5: ("writer_idle", "Writer idle"),
}
SUBDIR_TO_POWER_CASE: Dict[str, int] = {subdir: case for case, (subdir, _label) in POWER_CASE_SPECS.items()}

# Case 1 (all active) and the three single-variable ablations: each idles exactly one engine
# relative to case 1, with the other two engines held real (and write amplification at 100%) on both
# sides, so (case 1 - case X) isolates engine X's own contribution to dynamic power.
DEFAULT_POWER_CASES: List[int] = [5, 1, 2, 4]

# Which per-tile instruction the compute kernel runs (LONG_MATMUL_OP, see long_matmul's
# kernels/compute/compute.cpp). The full
# POWER_CASE sweep runs once per entry here, into its own <output-root>/<op>/ subtree, so the
# operations can be compared like-for-like. Each op/case run is skipped individually once it is
# complete with the same settings (see the run loop below), so re-running after adding new ops
# only measures the new ones. All seven run by default; pass --ops explicitly to run a subset instead (e.g.
# while iterating on one op).
ALL_OPS: List[str] = ["matmul", "add", "silu", "exp", "sigmoid", "gelu", "recip"]
DEFAULT_OPS: List[str] = ALL_OPS


def run_case(
    cmd: List[str],
    power_case: int,
    op: str,
    dry_run: bool,
) -> int:
    print(f"\n{'=' * 60}", flush=True)
    print(f" op={op} POWER_CASE={power_case}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"[RUN] op={op} POWER_CASE={power_case} {shell_join(cmd)}", flush=True)

    if dry_run:
        print("[RUN] (dry-run) skipped", flush=True)
        return 0

    result = subprocess.run(cmd, env=environ_with({"POWER_CASE": str(power_case), "LONG_MATMUL_OP": op}))
    return result.returncode


def write_cases_file(path: Path, cases_in_order: List[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, (subdir, label) in enumerate(cases_in_order, start=1):
            f.write(f"Case {i}: {subdir} : {label}\n")
    print(f"[CASES] Wrote: {path}", flush=True)


def write_cross_op_cases_file(
    path: Path,
    ops: List[str],
    cases_in_order: List[Tuple[str, str]],
    runs: "Runs",
) -> List[Tuple[str, str]]:
    """Pairs every (op, case) combination into one cases list, subdir paths relative to
    output_root (e.g. "matmul/regular"), grouped by case so same-case entries across ops sit
    next to each other as adjacent bars. Skips any pair that is not a complete run with this
    sweep's settings -- e.g. a case that failed, or wasn't run for that op -- so one missing run
    doesn't block comparing everything else. Returns the entries actually written.
    """
    entries: List[Tuple[str, str]] = []
    for subdir, label in cases_in_order:
        for op in ops:
            rel = f"{op}/{subdir}"
            if not runs.complete(op, subdir):
                print(
                    f"[CROSS-OP] note: no complete '{rel}' run yet -- skipping it in the cross-op comparison",
                    flush=True,
                )
                continue
            entries.append((rel, f"{label} ({op})"))

    with path.open("w", encoding="utf-8") as f:
        for i, (subdir, label) in enumerate(entries, start=1):
            f.write(f"Case {i}: {subdir} : {label}\n")
    print(f"[CASES] Wrote: {path}", flush=True)
    return entries


# Per-engine energy-per-FLOP ablation: each engine's own contribution is isolated by subtracting its
# idle-case's dynamic *average* charge from the all-active case's, then converting charge ->
# energy via V_core and dividing by the fixed per-interval FLOP count (split mode gives every
# grid the same total work). Uses dynamic_charge_avg_as rather than the peak variant: peak
# charge is driven by a single sample and is noisier, and averaging over the whole interval is
# less exposed to that. All three engines use a genuine idle case (real work vs none, with the
# other two engines held identical on both sides of the subtraction) -- see the
# POWER_CASE_SPECS comment above. "regular" (the pre-case-5 writer stand-in, marginal cost of
# write amplification rather than the writer's full contribution) is deliberately not used here.
ALL_ACTIVE_CASE = "writer_amp"
ENGINE_ABLATIONS: List[Tuple[str, str, str]] = [
    ("reader", "reader_idle2", "Reader"),
    ("writer", "writer_idle", "Writer"),
    ("compute", "compute_idle", "Compute"),
]

PJ_PER_J = 1e12

# LONG_MATMUL_MODE, the same switch long_matmul and parser.py read (see long_matmul's README).
LEGACY_MODE = legacy_mode()


@dataclass
class Runs:
    """Reads the runs of this sweep's settings. A run made with other settings (another shape or
    mode), or one that did not finish, is treated as missing, so it never reaches a chart."""

    output_root: Path
    app_exe: Path
    app_args: List[str]
    telemetry_freq: int
    trim_ms: float
    slot_ms: int
    device_id: Optional[int]

    def settings(self, op: str, subdir: str) -> Dict[str, object]:
        power_case = SUBDIR_TO_POWER_CASE[subdir]
        env = environ_with({"POWER_CASE": str(power_case), "LONG_MATMUL_OP": op})
        return run_settings(
            self.app_exe, self.app_args, env, self.telemetry_freq, self.trim_ms, self.slot_ms, self.device_id
        )

    def complete(self, op: str, subdir: str) -> bool:
        return run_status(self.output_root / op / subdir, self.settings(op, subdir)) == "done"

    def grid_order(self, pairs: List[Tuple[str, str]]) -> List[str]:
        """Every grid in the complete runs among (op, subdir) pairs, ordered by core count."""
        rows_by_case: Dict[str, List[Dict[str, str]]] = {}
        for op, subdir in pairs:
            if self.complete(op, subdir):
                with (self.output_root / op / subdir / "program_intervals.csv").open(newline="") as f:
                    rows_by_case[f"{op}/{subdir}"] = list(csv.DictReader(f))
        return build_grid_order(rows_by_case)

    def rows(self, op: str, subdir: str, note_prefix: str) -> Optional[Dict[str, Dict[str, str]]]:
        """program_intervals.csv of op/subdir by grid, or None (with a note) if that run is not a
        complete run with this sweep's settings."""
        if not self.complete(op, subdir):
            print(f"{note_prefix} note: no complete '{op}/{subdir}' run with these settings, skipping it", flush=True)
            return None
        with (self.output_root / op / subdir / "program_intervals.csv").open(newline="") as f:
            return {row["grid"]: row for row in csv.DictReader(f)}


def compute_energy_by_engine(runs: Runs, op: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Dynamic energy [J] per engine (reader/writer/compute) for one op's case sweep, per grid:
    (all-active dynamic avg charge - idle-case dynamic avg charge) x V_core. Returns None if the
    all-active case is missing; skips (with a note) any engine whose idle case is missing, so
    callers still get whatever engines have data.
    """
    note_prefix = f"[ENERGY][{op}]"
    active = runs.rows(op, ALL_ACTIVE_CASE, note_prefix)
    if active is None:
        return None

    per_engine: Dict[str, Dict[str, float]] = {}
    for engine, idle_case, _label in ENGINE_ABLATIONS:
        idle = runs.rows(op, idle_case, note_prefix)
        if idle is None:
            continue
        per_grid: Dict[str, float] = {}
        for grid, active_row in active.items():
            if grid not in idle:
                continue
            try:
                vcore = float(active_row["avg_vcore_v"])
            except (KeyError, ValueError):
                vcore = float("nan")
            active_as = whole_run_value(active_row, "dynamic_charge_avg_as", LEGACY_MODE)
            idle_as = whole_run_value(idle[grid], "dynamic_charge_avg_as", LEGACY_MODE)
            energy_j = (active_as - idle_as) * vcore
            if not math.isfinite(energy_j):
                print(f"{note_prefix} note: {engine} at {grid} has no finite value, skipping", flush=True)
                continue
            per_grid[grid] = energy_j
        if per_grid:
            per_engine[engine] = per_grid

    return per_engine or None


def to_pj_per_flop(per_key_j: Dict[str, Dict[str, float]], flops: float) -> Dict[str, Dict[str, float]]:
    """Joules per grid -> pJ per FLOP, with the fixed per-interval FLOP count (split mode gives
    every grid the same total work)."""
    return {key: {grid: j / flops * PJ_PER_J for grid, j in per_grid.items()} for key, per_grid in per_key_j.items()}


def grids_with_data(grid_order: List[str], per_key: Dict[str, Dict[str, float]]) -> List[str]:
    return [g for g in grid_order if any(g in vals for vals in per_key.values())]


def plot_energy_per_flop_by_engine(
    per_engine: Dict[str, Dict[str, float]],
    grid_order: List[str],
    out_path: Path,
    dpi: int,
) -> None:
    grouped_bar_chart(
        grids_with_data(grid_order, per_engine),
        [(engine, label) for engine, _idle_case, label in ENGINE_ABLATIONS if engine in per_engine],
        per_engine,
        title="Energy per FLOP by engine (reader / writer / compute)",
        ylabel="Energy per FLOP [pJ], dynamic avg-charge basis",
        out_path=out_path,
        dpi=dpi,
    )


def compute_energy_per_flop_per_core(
    runs: Runs,
    op: str,
    cases_in_order: List[Tuple[str, str]],
    flops: float,
) -> Dict[str, Dict[str, float]]:
    """pJ/FLOP/core per profile (case), per grid: (energy_j_vi / FLOPS) / cores. Unlike the
    per-engine ablation charts, this uses each case's own total energy (energy_j_vi, which
    includes the idle floor) directly -- no subtraction between cases -- so it's defined for
    every profile, including ones where no real FLOPs are performed (compute_idle, etc.); read
    it there as "energy per core, normalized by the nominal FLOP count", not as a real
    efficiency figure. Dividing by cores means a case whose energy is dominated by the
    (per-board, not per-core) idle floor will show falling pJ/FLOP/core as cores increase, even
    if its raw pJ/FLOP is flat -- that dilution is the point of the metric, not an artifact.
    Missing cases are skipped with a note.
    """
    per_case: Dict[str, Dict[str, float]] = {}
    for subdir, _label in cases_in_order:
        rows = runs.rows(op, subdir, "[ENERGY/FLOP/CORE]")
        if rows is None:
            continue
        per_grid: Dict[str, float] = {}
        for grid, row in rows.items():
            energy_j = whole_run_value(row, "energy_j_vi", LEGACY_MODE)
            try:
                cores = float(row["cores"])
            except (KeyError, ValueError):
                continue
            if cores <= 0 or not math.isfinite(energy_j):
                continue
            per_grid[grid] = energy_j / flops * PJ_PER_J / cores
        if per_grid:
            per_case[subdir] = per_grid
    return per_case


def plot_energy_per_flop_per_core(
    per_case: Dict[str, Dict[str, float]],
    cases_in_order: List[Tuple[str, str]],
    grid_order: List[str],
    out_path: Path,
    dpi: int,
) -> None:
    grouped_bar_chart(
        grids_with_data(grid_order, per_case),
        [(s, l) for s, l in cases_in_order if s in per_case],
        per_case,
        title="Energy per FLOP per core, per profile",
        ylabel="Energy per FLOP per core [pJ]",
        out_path=out_path,
        dpi=dpi,
    )


def clip_negative(values: Dict[str, float], grids: List[str], name: str, unit: str, clipped: List[str]) -> List[float]:
    """values per grid in grid order, with a missing grid as 0 and a negative value clipped to 0
    (and recorded in `clipped`). A stacked bar can't show a negative slice as a share of the
    whole; these are an idle case measured slightly above the all-active one, which happens for
    cheap ops like add, where the difference is within noise."""
    out = []
    for g in grids:
        v = values.get(g, 0.0)
        if v < 0:
            clipped.append(f"{name}@{g} ({v:.4g} {unit})")
            v = 0.0
        out.append(v)
    return out


def report_clipped(clipped: List[str], chart: str) -> None:
    if clipped:
        print(
            f"[ENERGY] note: clipped {len(clipped)} negative value(s) to 0 for the {chart} "
            f"(noise floor, not a real negative energy): {clipped}",
            file=sys.stderr,
        )


def plot_energy_by_engine_stacked(
    per_engine: Dict[str, Dict[str, float]],
    grid_order: List[str],
    out_path: Path,
    dpi: int,
) -> None:
    """One stacked bar per grid: reader/writer/compute dynamic energy [J] stacked to show each
    engine's share of the total. Negative per-engine values are clipped to 0 for the stack and
    reported on stderr (see clip_negative).
    """
    grids = grids_with_data(grid_order, per_engine)
    x = np.arange(len(grids))

    clipped: List[str] = []
    fig, ax = plt.subplots(figsize=(max(10, len(grids) * 1.0), 6))
    bottom = np.zeros(len(grids))
    for engine, _idle_case, label in ENGINE_ABLATIONS:
        if engine not in per_engine:
            continue
        vals = np.array(clip_negative(per_engine[engine], grids, engine, "J", clipped))
        ax.bar(x, vals, width=0.6, bottom=bottom, label=label)
        bottom += vals
    report_clipped(clipped, "stacked total-energy chart")

    for i, total in enumerate(bottom):
        ax.annotate(
            f"{total:.3g} J",
            (x[i], total),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel("Dynamic energy [J]")
    ax.set_title("Total dynamic energy by engine (reader / writer / compute), per grid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out_path, dpi)
    print(f"Wrote: {out_path}", flush=True)


# Hatch patterns distinguishing ops within a grid's cluster of stacked bars (color already
# distinguishes engine within a bar); one per op in ALL_OPS.
OP_HATCHES = ["", "//", "xx", "\\\\", "oo", "..", "++"]


def plot_cross_op_stacked_by_engine(
    per_op_engine: Dict[str, Dict[str, Dict[str, float]]],
    ops: List[str],
    grid_order: List[str],
    out_path: Path,
    dpi: int,
    ylabel: str,
    title: str,
    value_unit: str,
    normalize_pct: bool = False,
) -> None:
    """The cross-op stacked-by-engine charts (raw Joules, pJ/FLOP, and percentage share): for
    every grid, one stacked reader/writer/compute bar *per op*, side by side, so the operations
    can be compared like-for-like at each grid. Color encodes engine (shared legend across ops);
    hatch encodes op.

    normalize_pct=True rescales each (op, grid) bar's own three engine values to sum to 100 --
    every bar is the same height, only the internal split varies -- so the *share* each engine
    holds is comparable across ops and grids directly, without total magnitude (which differs a
    lot between e.g. silu and matmul) getting in the way. This ratio is the same whether computed
    from raw Joules or pJ/FLOP, since FLOPS is a per-grid constant shared by all three engines at
    that grid and cancels out of the ratio -- so callers can pass either.
    """
    grids = [
        g for g in grid_order if any(g in vals for per_engine in per_op_engine.values() for vals in per_engine.values())
    ]
    x = np.arange(len(grids))
    ops_with_data = [op for op in ops if op in per_op_engine]
    width = 0.8 / max(len(ops_with_data), 1)

    clipped: List[str] = []
    fig, ax = plt.subplots(figsize=(max(12, len(grids) * 1.3), 6.5))

    engine_colors = {engine: f"C{i}" for i, (engine, _idle_case, _label) in enumerate(ENGINE_ABLATIONS)}

    for op_idx, op in enumerate(ops_with_data):
        per_engine = per_op_engine[op]
        offset = (op_idx - (len(ops_with_data) - 1) / 2) * width

        raw = {
            engine: clip_negative(per_engine[engine], grids, f"{op}/{engine}", value_unit, clipped)
            for engine, _idle_case, _label in ENGINE_ABLATIONS
            if engine in per_engine
        }

        if normalize_pct:
            totals = [sum(vals[gi] for vals in raw.values()) for gi in range(len(grids))]
            for engine in raw:
                raw[engine] = [
                    (raw[engine][gi] / totals[gi] * 100.0) if totals[gi] > 0 else 0.0 for gi in range(len(grids))
                ]

        bottom = np.zeros(len(grids))
        for engine, _idle_case, _label in ENGINE_ABLATIONS:
            if engine not in raw:
                continue
            vals = np.array(raw[engine])
            ax.bar(
                x + offset,
                vals,
                width,
                bottom=bottom,
                color=engine_colors[engine],
                hatch=OP_HATCHES[op_idx % len(OP_HATCHES)],
                edgecolor="black",
                linewidth=0.5,
            )
            bottom += vals
    report_clipped(clipped, "cross-op stacked chart")

    engine_handles = [plt.Rectangle((0, 0), 1, 1, fc=engine_colors[engine]) for engine, _i, _l in ENGINE_ABLATIONS]
    engine_labels = [label for _e, _i, label in ENGINE_ABLATIONS]
    op_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch=OP_HATCHES[i % len(OP_HATCHES)])
        for i in range(len(ops_with_data))
    ]
    ax.legend(
        engine_handles + op_handles,
        engine_labels + ops_with_data,
        loc="upper left",
        fontsize=9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}, per grid, per op ({' vs '.join(ops_with_data)})")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out_path, dpi)
    print(f"Wrote: {out_path}", flush=True)


def compute_total_energy_per_flop_by_op(
    runs: Runs,
    ops: List[str],
    case: str,
    flops: float,
) -> Dict[str, Dict[str, float]]:
    """pJ/FLOP per op (not broken down by engine) for one profile/case: each op's total energy
    (energy_j_vi, including the idle floor) for that case, divided by the fixed FLOP count, per
    grid, read once per op here so the operations can be compared directly. Ops missing this
    case are skipped with a note.
    """
    per_op: Dict[str, Dict[str, float]] = {}
    for op in ops:
        rows = runs.rows(op, case, "[ENERGY/FLOP]")
        if rows is None:
            continue
        per_grid: Dict[str, float] = {}
        for grid, row in rows.items():
            energy_j = whole_run_value(row, "energy_j_vi", LEGACY_MODE)
            if not math.isfinite(energy_j):
                continue
            per_grid[grid] = energy_j / flops * PJ_PER_J
        if per_grid:
            per_op[op] = per_grid
    return per_op


def plot_total_energy_per_flop_by_op(
    per_op: Dict[str, Dict[str, float]],
    ops: List[str],
    grid_order: List[str],
    out_path: Path,
    dpi: int,
) -> None:
    """For every grid, one plain bar per op, side by side: each op's total energy (energy_j_vi
    for ALL_ACTIVE_CASE, including the idle floor) divided by the fixed FLOP count. This is the
    un-broken-down total that the cross-op stacked chart's bars sum to (plus the shared idle
    floor that chart ablates away).
    """
    ops_with_data = [op for op in ops if op in per_op]
    grouped_bar_chart(
        grids_with_data(grid_order, per_op),
        [(op, op) for op in ops_with_data],
        per_op,
        title=(
            f"Total energy per FLOP, per op ({' vs '.join(ops_with_data)}), all-active case\n"
            "every op divided by matmul's 2*M*N*K FLOPs, so this compares energy at equal data movement"
        ),
        ylabel="Total energy per FLOP [pJ]",
        out_path=out_path,
        dpi=dpi,
        figsize=(max(12, len(grid_order) * 1.3), 6.5),
    )


def run_compare(
    compare_script: Path,
    output_root: Path,
    cases_file: Path,
    dpi: int,
    dry_run: bool,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(compare_script),
        "-i",
        str(output_root),
        "-c",
        str(cases_file),
        "--dpi",
        str(dpi),
    ]
    print(f"\n[COMPARE] {shell_join(cmd)}", flush=True)
    if dry_run:
        print("[COMPARE] (dry-run) skipped", flush=True)
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the POWER_CASE sweep end-to-end (reset + auto.py per case) and compare the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--power-cases",
        type=int,
        nargs="+",
        default=DEFAULT_POWER_CASES,
        choices=sorted(POWER_CASE_SPECS.keys()),
        help=f"Which POWER_CASE values to run, in order. Default: {DEFAULT_POWER_CASES}",
    )
    ap.add_argument(
        "--ops",
        nargs="+",
        default=DEFAULT_OPS,
        choices=ALL_OPS,
        help=f"Which per-tile operation(s) to sweep (LONG_MATMUL_OP). The full --power-cases "
        f"sweep runs once per op, into <output-root>/<op>/. Default: {DEFAULT_OPS}",
    )
    ap.add_argument("--auto-script", type=Path, default=Path(__file__).parent / "auto.py")
    ap.add_argument("--compare-script", type=Path, default=Path(__file__).parent / "compare_cases.py")
    ap.add_argument("--telemetry-exe", required=True, type=Path)
    ap.add_argument("--telemetry-freq", type=int, default=DEFAULT_TELEMETRY_FREQ_HZ)
    ap.add_argument("--app-exe", required=True, type=Path)
    ap.add_argument("--parser-script", required=True, type=Path)
    ap.add_argument(
        "--tt-venv-activate",
        required=True,
        type=Path,
        help="Path to the Python venv activation script used by auto.py.",
    )
    ap.add_argument("--tt-metal-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--slot-ms", type=int, default=DEFAULT_SLOT_MS)
    ap.add_argument("--device-id", type=int, default=DEFAULT_DEVICE_ID)
    ap.add_argument(
        "--trim-ms",
        type=float,
        default=POWER_CASES_TRIM_MS,
        help=f"Default: {POWER_CASES_TRIM_MS}, not auto.py's default: the published POWER_CASE results "
        "were made with it. A run made with another value counts as different settings.",
    )
    ap.add_argument("--tt-smi", default="tt-smi", help="tt-smi executable name or path. Default: tt-smi")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"DPI for comparison figures. Default: {DEFAULT_DPI}")
    ap.add_argument(
        "--cases-file-name",
        default="power_cases.txt",
        help="Filename (under --output-root/<op>) for the generated per-op cases file. Default: power_cases.txt",
    )
    ap.add_argument(
        "--cross-op-cases-file-name",
        default="power_cases_cross_op.txt",
        help="Filename (under --output-root) for the cross-op cases file, used only when "
        "len(--ops) > 1. Default: power_cases_cross_op.txt",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run a case even if it is already complete, or was made with different settings. "
            "Default: skip complete cases and stop at ones made with different settings."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every command that would run (reset, auto.py, compare_cases.py) without executing it.",
    )
    # --app-args must be last (argparse.REMAINDER)
    ap.add_argument(
        "--app-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed to the application (M N K num_iterations [fixed_blocks_per_core]). Put this option last.",
    )

    args = ap.parse_args(argv)
    # --app-args takes everything after it; refuse options of this script written after it.
    misplaced = misplaced_options(ap, args.app_args)
    if misplaced:
        ap.error(f"--app-args must come last; these options follow it: {' '.join(misplaced)}")
    # Same check auto.py makes, done here so a bad sweep fails before the first reset.
    fixed_blocks_per_core = 0
    mnk_iters: Optional[Tuple[int, int, int, int]] = None
    if args.app_exe.name == LONG_MATMUL_APP:
        try:
            m, n, k, iterations, fixed_blocks_per_core = long_matmul_args(args.app_args)
            check_long_matmul_device(args.app_exe, args.device_id)
        except ValueError as e:
            ap.error(str(e))
        mnk_iters = (m, n, k, iterations)

    output_root = args.output_root.expanduser().resolve()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    cases_in_order: List[Tuple[str, str]] = [POWER_CASE_SPECS[i] for i in args.power_cases]
    runs = Runs(
        output_root, args.app_exe, list(args.app_args), args.telemetry_freq, args.trim_ms, args.slot_ms, args.device_id
    )

    print(f"Output root: {output_root}", flush=True)
    print(f"Ops sweep: {args.ops}", flush=True)
    print(f"POWER_CASE sweep: {args.power_cases} -> {[s for s, _ in cases_in_order]}", flush=True)

    # FLOPs per interval, for the energy-per-FLOP charts -- the same on every grid, since split
    # mode divides the same total work across however many cores are active.
    flops: Optional[float] = None
    if fixed_blocks_per_core != 0:
        # Fixed mode: total work grows with the core count, so one FLOP count per run is wrong.
        print(
            "[ENERGY/FLOP] note: fixed_blocks_per_core is set, so the FLOP count depends on the grid; "
            "skipping the energy-per-FLOP charts",
            flush=True,
        )
    elif mnk_iters is not None:
        m, n, k, iterations = mnk_iters
        flops = 2.0 * m * n * k * iterations
    else:
        print(f"[ENERGY/FLOP] note: the app is not {LONG_MATMUL_APP}; skipping the energy-per-FLOP charts", flush=True)

    # Check every run before the first reset: a run made with other settings stops the sweep, and
    # finding one hours in, after the earlier ops, would waste them.
    if not args.force:
        conflicts = [
            f"{op}/{subdir}"
            for op in args.ops
            for subdir, _label in cases_in_order
            if run_status(output_root / op / subdir, runs.settings(op, subdir)) == "different"
        ]
        if conflicts:
            print(
                f"ERROR: these runs were made with different settings (see their {RUN_COMPLETE_FILE}): "
                f"{', '.join(conflicts)}. Use another --output-root, or --force to replace them.",
                file=sys.stderr,
            )
            return 1

    per_op_engine_j: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_pairs = [(op, subdir) for op in args.ops for subdir, _label in cases_in_order]
    for op in args.ops:
        op_output_root = output_root / op
        if not args.dry_run:
            op_output_root.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}", flush=True)
        print(f" op={op} -> {op_output_root}", flush=True)
        print(f"{'#' * 60}", flush=True)

        for power_case, (subdir, _label) in zip(args.power_cases, cases_in_order):
            run_dir = op_output_root / subdir
            # A run is done only if auto.py recorded it as complete with the same settings; a
            # run that failed part way is redone (runs with other settings were refused above).
            if run_status(run_dir, runs.settings(op, subdir)) == "done" and not args.force:
                print(f"\n[SKIP] '{op}/{subdir}' is already complete. Skipping (use --force to re-run).", flush=True)
                continue

            rc = reset_hardware(args.tt_smi, args.dry_run)
            if rc != 0:
                print(
                    f"ERROR: Hardware reset failed before op={op} POWER_CASE={power_case} ({subdir}). Aborting.",
                    file=sys.stderr,
                )
                return rc

            cmd = auto_py_command(
                auto_script=args.auto_script,
                telemetry_exe=args.telemetry_exe,
                telemetry_freq=args.telemetry_freq,
                app_exe=args.app_exe,
                app_args=list(args.app_args),
                parser_script=args.parser_script,
                tt_venv_activate=args.tt_venv_activate,
                tt_metal_root=args.tt_metal_root,
                output_root=op_output_root,
                subdir=subdir,
                slot_ms=args.slot_ms,
                device_id=args.device_id,
                trim_ms=args.trim_ms,
            )
            rc = run_case(cmd, power_case, op, args.dry_run)
            if rc != 0:
                print(
                    f"ERROR: op={op} POWER_CASE={power_case} ({subdir}) failed with return code {exit_reason(rc)}. Aborting.",
                    file=sys.stderr,
                )
                return rc

        cases_file = op_output_root / args.cases_file_name
        if not args.dry_run:
            write_cases_file(cases_file, cases_in_order)

        rc = run_compare(args.compare_script, op_output_root, cases_file, args.dpi, args.dry_run)
        if rc != 0:
            print(f"ERROR: compare_cases.py failed for op={op} with return code {rc}.", file=sys.stderr)
            return rc

        if not args.dry_run:
            grid_order = runs.grid_order([(op, subdir) for subdir, _label in cases_in_order])
            out_dir = op_output_root / "compare_runs_out"

            per_engine_j = compute_energy_by_engine(runs, op)
            if per_engine_j is not None:
                per_op_engine_j[op] = per_engine_j
                plot_energy_by_engine_stacked(
                    per_engine_j, grid_order, out_dir / "energy_by_engine_stacked.png", args.dpi
                )
                if flops is not None:
                    plot_energy_per_flop_by_engine(
                        to_pj_per_flop(per_engine_j, flops),
                        grid_order,
                        out_dir / "energy_per_flop_by_engine.png",
                        args.dpi,
                    )

            if flops is not None:
                per_case_pj_core = compute_energy_per_flop_per_core(runs, op, cases_in_order, flops)
                if per_case_pj_core:
                    plot_energy_per_flop_per_core(
                        per_case_pj_core,
                        cases_in_order,
                        grid_order,
                        out_dir / "energy_per_flop_per_core.png",
                        args.dpi,
                    )

    cross_op_done = False
    if len(args.ops) > 1 and not args.dry_run:
        print(f"\n{'#' * 60}", flush=True)
        print(f" Cross-op comparison ({' vs '.join(args.ops)})", flush=True)
        print(f"{'#' * 60}", flush=True)

        cross_cases_file = output_root / args.cross_op_cases_file_name
        entries = write_cross_op_cases_file(cross_cases_file, args.ops, cases_in_order, runs)
        if len(entries) < 2:
            print(
                f"[CROSS-OP] only {len(entries)} run(s) have results; need at least 2 to compare. Skipping.",
                flush=True,
            )
        else:
            rc = run_compare(args.compare_script, output_root, cross_cases_file, args.dpi, args.dry_run)
            if rc != 0:
                print(
                    f"ERROR: compare_cases.py failed for the cross-op comparison with return code {rc}.",
                    file=sys.stderr,
                )
                return rc
            cross_op_done = True

            grid_order = runs.grid_order(all_pairs)
            out_dir = output_root / "compare_runs_out"
            if not per_op_engine_j:
                print(
                    f"[ENERGY] note: no op has energy-by-engine data under {output_root}; skipping the cross-op charts"
                )
            else:
                plot_cross_op_stacked_by_engine(
                    per_op_engine_j,
                    args.ops,
                    grid_order,
                    out_dir / "energy_by_engine_stacked_cross_op.png",
                    args.dpi,
                    ylabel="Dynamic energy [J]",
                    title="Total dynamic energy by engine",
                    value_unit="J",
                )
                plot_cross_op_stacked_by_engine(
                    per_op_engine_j,
                    args.ops,
                    grid_order,
                    out_dir / "energy_share_by_engine_stacked_cross_op.png",
                    args.dpi,
                    ylabel="Share of dynamic energy [%]",
                    title="Reader/Writer/Compute share of dynamic energy",
                    value_unit="J",
                    normalize_pct=True,
                )
                if flops is not None:
                    # Reader/writer NoC traffic is the same whichever per-tile instruction runs, so
                    # if the reader+writer share stops shrinking as cores rise while compute's
                    # does, the NoC (not the FPU) is becoming the bottleneck.
                    plot_cross_op_stacked_by_engine(
                        {op: to_pj_per_flop(per_engine, flops) for op, per_engine in per_op_engine_j.items()},
                        args.ops,
                        grid_order,
                        out_dir / "energy_per_flop_by_engine_stacked_cross_op.png",
                        args.dpi,
                        ylabel="Energy per FLOP [pJ]",
                        title="Energy per FLOP by engine (matmul's 2*M*N*K FLOPs for every op)",
                        value_unit="pJ/FLOP",
                    )

            if flops is not None:
                per_op = compute_total_energy_per_flop_by_op(runs, args.ops, ALL_ACTIVE_CASE, flops)
                if per_op:
                    plot_total_energy_per_flop_by_op(
                        per_op, args.ops, grid_order, out_dir / "total_energy_per_flop_by_op.png", args.dpi
                    )

    print(f"\n{'=' * 60}", flush=True)
    print(f" Done. Results are under: {output_root}/<op>/ for op in {args.ops}", flush=True)
    for op in args.ops:
        print(f" {op:>8} comparison charts: {output_root / op / 'compare_runs_out'}", flush=True)
    if cross_op_done:
        print(f" cross-op comparison charts: {output_root / 'compare_runs_out'}", flush=True)
    print(f"{'=' * 60}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
