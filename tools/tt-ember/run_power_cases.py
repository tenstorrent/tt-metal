#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""
Top-level driver for the POWER_CASE power-experiment sweep.

For each requested POWER_CASE value (see the "POWER_CASE Power-Experiment Scenarios" section
in README.md, and tt_metal/programming_examples/high_power_matmul/README.md for what each case
actually disables/amplifies in the kernels), and for each requested --ops entry (which per-tile
FPU instruction the compute kernel runs, via HIGH_POWER_OP -- see mm_power.cpp), this script:

  1. Resets the hardware (`tt-smi -r`) and waits for it to re-initialize.
  2. Runs auto.py with POWER_CASE=<i> and HIGH_POWER_OP=<op> exported into the application's
     environment, writing that case's results to <output-root>/<op>/<subdir>.
  3. After every requested case has completed for a given op, writes a cases file
     (`Case N: subdir : label`) describing the POWER_CASE -> subdir mapping for that op, and
     calls compare_runs2.py against it to produce the comparison charts in
     <output-root>/<op>/compare_runs_out/.
  4. If more than one op was requested, also writes a cross-op cases file pairing every
     (op, case) combination -- e.g. "regular (matmul)" next to "regular (add)" -- and calls
     compare_runs2.py once more to produce <output-root>/compare_runs_out/, so the two
     operations can be read off the same chart, case by case. Any (op, case) pair whose
     program_intervals.csv is missing (failed or not yet run) is skipped with a note rather
     than failing the whole comparison.

This automates exactly the manual sequence, repeated once per op in --ops:
    tt-smi -r
    export POWER_CASE=<i> HIGH_POWER_OP=<op>
    python3 auto.py --telemetry-exe ... --app-exe ... --output-root <output-root>/<op> \\
        --subdir <name-for-that-case> --app-args ...
run once per case, followed by:
    python3 compare_runs2.py -i <output-root>/<op> -c <output-root>/<op>/power_cases.txt

Usage:
    python3 run_power_cases.py \\
        --telemetry-exe /path/to/telemetry \\
        --app-exe /path/to/metal_example_high_power_matmul \\
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
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESET_COOLDOWN_S = 10

# POWER_CASE -> (subdir name, legend label). See README.md's "POWER_CASE Power-Experiment
# Scenarios" section for the full rationale behind each scenario. Case 3 (reader AND compute
# both idle) is not in the default sweep below -- it's a two-variable case, and the four
# single-variable ablations already isolate each engine's own contribution to dynamic power.
# Case 5 (genuine USE_WRITER=0) is the writer's idle case, symmetric with case 2 (compute idle)
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

# The four single-variable ablations: each idles exactly one engine relative to case 1 (all
# active), with the other two engines held real (and write amplification at 100%) on both
# sides, so (case 1 - case X) isolates engine X's own contribution to dynamic power.
DEFAULT_POWER_CASES: List[int] = [5, 1, 2, 4]

# Which per-tile FPU instruction the compute kernel runs (HIGH_POWER_OP, see mm_power.cpp's
# USE_ADD / USE_SILU / USE_EXP / USE_SIGMOID / USE_GELU / USE_RECIP / MATMUL_FNN_DN). The full
# POWER_CASE sweep runs once per entry here, into its own <output-root>/<op>/ subtree, so the
# operations can be compared like-for-like. Each op/case directory is skipped individually if
# it already exists (see the run loop below), so re-running after adding new ops only measures
# the new ones. All seven run by default; pass --ops explicitly to run a subset instead (e.g.
# while iterating on one op).
ALL_OPS: List[str] = ["matmul", "add", "silu", "exp", "sigmoid", "gelu", "recip"]
DEFAULT_OPS: List[str] = ALL_OPS


def shell_join(args: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def reset_hardware(tt_smi_cmd: str, dry_run: bool) -> int:
    print(f"\n[RESET] Running hardware reset: {tt_smi_cmd} -r", flush=True)
    if dry_run:
        print("[RESET] (dry-run) skipped", flush=True)
        return 0
    result = subprocess.run([tt_smi_cmd, "-r"])
    if result.returncode != 0:
        print(f"[RESET] ERROR: Hardware reset failed with return code {result.returncode}.", file=sys.stderr)
        return result.returncode
    print(f"[RESET] Hardware reset completed. Waiting {RESET_COOLDOWN_S}s for device to re-initialize...", flush=True)
    for remaining in range(RESET_COOLDOWN_S, 0, -1):
        print(f"[RESET] Resuming in {remaining}s...", flush=True)
        time.sleep(1)
    print("[RESET] Device ready.", flush=True)
    return 0


def run_case(
    auto_script: Path,
    telemetry_exe: Path,
    telemetry_freq: int,
    app_exe: Path,
    app_args: List[str],
    parser_script: Path,
    tt_venv_activate: Path,
    tt_metal_root: Path,
    output_root: Path,
    subdir: str,
    slot_ms: int,
    device_id: Optional[int],
    trim_ms: float,
    power_case: int,
    op: str,
    dry_run: bool,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(auto_script),
        "--telemetry-exe", str(telemetry_exe),
        "--telemetry-freq", str(telemetry_freq),
        "--app-exe", str(app_exe),
        "--parser-script", str(parser_script),
        "--tt-venv-activate", str(tt_venv_activate),
        "--tt-metal-root", str(tt_metal_root),
        "--output-root", str(output_root),
        "--subdir", subdir,
        "--slot-ms", str(slot_ms),
        "--trim-ms", str(trim_ms),
    ]
    if device_id is not None:
        cmd += ["--device-id", str(device_id)]
    # --app-args must be last (argparse.REMAINDER in auto.py)
    cmd += ["--app-args"] + app_args

    env = dict(os.environ)
    env["POWER_CASE"] = str(power_case)
    env["HIGH_POWER_OP"] = op

    print(f"\n{'=' * 60}", flush=True)
    print(f" op={op} POWER_CASE={power_case} -> subdir={subdir}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"[RUN] op={op} POWER_CASE={power_case} {shell_join(cmd)}", flush=True)

    if dry_run:
        print("[RUN] (dry-run) skipped", flush=True)
        return 0

    result = subprocess.run(cmd, env=env)
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
    output_root: Path,
) -> List[Tuple[str, str]]:
    """Pairs every (op, case) combination into one cases list, subdir paths relative to
    output_root (e.g. "matmul/regular"), grouped by case so same-case entries across ops sit
    next to each other as adjacent bars. Skips any pair whose program_intervals.csv doesn't
    exist yet -- e.g. a case that failed, or wasn't run for that op -- so one missing run
    doesn't block comparing everything else. Returns the entries actually written.
    """
    entries: List[Tuple[str, str]] = []
    for subdir, label in cases_in_order:
        for op in ops:
            rel = f"{op}/{subdir}"
            csv_path = output_root / rel / "program_intervals.csv"
            if not csv_path.exists():
                print(
                    f"[CROSS-OP] note: no program_intervals.csv for '{rel}' yet -- "
                    "skipping it in the cross-op comparison",
                    flush=True,
                )
                continue
            entries.append((rel, f"{label} ({op})"))

    with path.open("w", encoding="utf-8") as f:
        for i, (subdir, label) in enumerate(entries, start=1):
            f.write(f"Case {i}: {subdir} : {label}\n")
    print(f"[CASES] Wrote: {path}", flush=True)
    return entries


# Per-engine energy-per-FLOP ablation, adapted from docs/results/n300_power_cases/
# make_pj_per_flop_by_engine.py: each engine's own contribution is isolated by subtracting its
# idle-case's dynamic *average* charge from the all-active case's, then converting charge ->
# energy via V_core and dividing by the fixed per-interval FLOP count (split mode gives every
# grid the same total work). Uses dynamic_charge_avg_as rather than the peak variant -- peak
# charge is a single-sample-driven, noisier signal (see the compute_idle-vs-all-active
# discussion this chart's negative values came from), and averaging over the whole interval is
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


def _load_program_intervals(csv_path: Path) -> Optional[Dict[str, Dict[str, str]]]:
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as f:
        return {row["grid"]: row for row in csv.DictReader(f)}


def compute_energy_by_engine(
    op_output_root: Path,
    note_prefix: str = "[ENERGY]",
) -> Optional[Dict[str, Dict[str, float]]]:
    """Dynamic energy [J] per engine (reader/writer/compute) for one op's case sweep, per grid:
    (all-active dynamic avg charge - idle-case dynamic avg charge) x V_core. This is the
    shared ablation both compute_energy_per_flop_by_engine() and the stacked total-energy chart
    are built from. Returns None if the all-active case is missing entirely; skips (with a
    note) any engine whose idle case is missing, so callers still get whatever engines have
    data.
    """
    active = _load_program_intervals(op_output_root / ALL_ACTIVE_CASE / "program_intervals.csv")
    if active is None:
        print(
            f"{note_prefix} note: no {ALL_ACTIVE_CASE}/program_intervals.csv under "
            f"{op_output_root}, skipping",
            flush=True,
        )
        return None

    per_engine: Dict[str, Dict[str, float]] = {}
    for engine, idle_case, _label in ENGINE_ABLATIONS:
        idle = _load_program_intervals(op_output_root / idle_case / "program_intervals.csv")
        if idle is None:
            print(
                f"{note_prefix} note: no {idle_case}/program_intervals.csv under "
                f"{op_output_root}, skipping '{engine}'",
                flush=True,
            )
            continue
        per_grid: Dict[str, float] = {}
        for grid, active_row in active.items():
            if grid not in idle:
                continue
            try:
                vcore = float(active_row["avg_vcore_v"])
                active_as = float(active_row["dynamic_charge_avg_as"])
                idle_as = float(idle[grid]["dynamic_charge_avg_as"])
            except (KeyError, ValueError):
                continue
            per_grid[grid] = (active_as - idle_as) * vcore
        if per_grid:
            per_engine[engine] = per_grid

    return per_engine or None


def compute_energy_per_flop_by_engine(
    op_output_root: Path,
    flops: float,
) -> Optional[Dict[str, Dict[str, float]]]:
    """pJ/FLOP per engine (reader/writer/compute) for one op's case sweep -- compute_energy_by_engine()'s
    Joule figures divided by the fixed per-interval FLOP count (split mode gives every grid the
    same total work).
    """
    per_engine_j = compute_energy_by_engine(op_output_root, note_prefix="[ENERGY/FLOP]")
    if per_engine_j is None:
        return None
    return {
        engine: {grid: j / flops * 1e12 for grid, j in per_grid.items()}
        for engine, per_grid in per_engine_j.items()
    }


def plot_energy_per_flop_by_engine(
    per_engine: Dict[str, Dict[str, float]],
    grid_order: List[str],
    out_path: Path,
    dpi: int,
) -> None:
    grids = [g for g in grid_order if any(g in vals for vals in per_engine.values())]
    x = np.arange(len(grids))
    width = 0.8 / max(len(ENGINE_ABLATIONS), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(grids) * 1.0), 6))
    for i, (engine, _idle_case, label) in enumerate(ENGINE_ABLATIONS):
        if engine not in per_engine:
            continue
        vals = [per_engine[engine].get(g, float("nan")) for g in grids]
        offset = (i - (len(ENGINE_ABLATIONS) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel("Energy per FLOP [pJ], dynamic avg-charge basis")
    ax.set_title("Energy per FLOP by engine (reader / writer / compute)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}", flush=True)


def compute_energy_per_flop_per_core(
    op_output_root: Path,
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
    Cases whose CSV is missing are skipped with a note.
    """
    per_case: Dict[str, Dict[str, float]] = {}
    for subdir, _label in cases_in_order:
        rows = _load_program_intervals(op_output_root / subdir / "program_intervals.csv")
        if rows is None:
            print(
                f"[ENERGY/FLOP/CORE] note: no {subdir}/program_intervals.csv under "
                f"{op_output_root}, skipping it",
                flush=True,
            )
            continue
        per_grid: Dict[str, float] = {}
        for grid, row in rows.items():
            try:
                energy_j = float(row["energy_j_vi"])
                cores = float(row["cores"])
            except (KeyError, ValueError):
                continue
            if cores <= 0:
                continue
            per_grid[grid] = energy_j / flops * 1e12 / cores
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
    grids = [g for g in grid_order if any(g in vals for vals in per_case.values())]
    x = np.arange(len(grids))
    active_cases = [(s, l) for s, l in cases_in_order if s in per_case]
    width = 0.8 / max(len(active_cases), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(grids) * 1.0), 6))
    for i, (subdir, label) in enumerate(active_cases):
        vals = [per_case[subdir].get(g, float("nan")) for g in grids]
        offset = (i - (len(active_cases) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel("Energy per FLOP per core [pJ]")
    ax.set_title("Energy per FLOP per core, per profile")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}", flush=True)


def plot_energy_by_engine_stacked(
    per_engine: Dict[str, Dict[str, float]],
    grid_order: List[str],
    out_path: Path,
    dpi: int,
) -> None:
    """One stacked bar per grid: reader/writer/compute dynamic energy [J] stacked to show each
    engine's share of the total. Negative per-engine values (see the compute-vs-all-active
    noise-floor discussion for the add op) are clipped to 0 for the stack -- a negative slice
    has no "share of the whole" meaning -- and reported on stderr instead of silently dropped.
    """
    grids = [g for g in grid_order if any(g in vals for vals in per_engine.values())]
    x = np.arange(len(grids))

    clipped: List[str] = []
    stacks: Dict[str, List[float]] = {}
    for engine, _idle_case, _label in ENGINE_ABLATIONS:
        if engine not in per_engine:
            continue
        vals = []
        for g in grids:
            v = per_engine[engine].get(g, 0.0)
            if v < 0:
                clipped.append(f"{engine}@{g} ({v:.4g} J)")
                v = 0.0
            vals.append(v)
        stacks[engine] = vals

    if clipped:
        print(
            f"[ENERGY] note: clipped {len(clipped)} negative engine/grid value(s) to 0 for the "
            f"stacked total-energy chart (noise floor, not a real negative energy): {clipped}",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=(max(10, len(grids) * 1.0), 6))
    bottom = np.zeros(len(grids))
    totals = np.zeros(len(grids))
    for engine, _idle_case, label in ENGINE_ABLATIONS:
        if engine not in stacks:
            continue
        vals = np.array(stacks[engine])
        ax.bar(x, vals, 0.6, bottom=bottom, label=label)
        bottom += vals
        totals += vals

    for i, total in enumerate(totals):
        ax.annotate(f"{total:.3g} J", (x[i], total), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel("Dynamic energy [J]")
    ax.set_title("Total dynamic energy by engine (reader / writer / compute), per grid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}", flush=True)


# Hatch patterns distinguishing ops within a grid's cluster of stacked bars (color already
# distinguishes engine within a bar). Cycles if more ops than patterns are given, though in
# practice --ops only ever has matmul/add.
OP_HATCHES = ["", "//", "xx", "\\\\", "oo"]


def _plot_cross_op_stacked_by_engine(
    per_op_engine: Dict[str, Dict[str, Dict[str, float]]],
    ops: List[str],
    grid_order: List[str],
    out_path: Path,
    dpi: int,
    ylabel: str,
    title: str,
    value_unit: str,
    normalize_pct: bool = False,
) -> bool:
    """Shared plotting body for the cross-op stacked-by-engine charts (raw Joules, pJ/FLOP, and
    percentage-share) -- for every grid, one stacked reader/writer/compute bar *per op*, side by
    side, so the operations can be compared like-for-like at each grid. Color encodes engine
    (shared legend across ops); hatch encodes op.

    normalize_pct=True rescales each (op, grid) bar's own three engine values to sum to 100 --
    every bar is the same height, only the internal split varies -- so the *share* each engine
    holds is comparable across ops and grids directly, without total magnitude (which differs a
    lot between e.g. silu and matmul) getting in the way. This ratio is the same whether computed
    from raw Joules or pJ/FLOP, since FLOPS is a per-grid constant shared by all three engines at
    that grid and cancels out of the ratio -- so callers can pass either.
    """
    if not per_op_engine:
        return False

    grids = [
        g for g in grid_order
        if any(g in vals for per_engine in per_op_engine.values() for vals in per_engine.values())
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

        raw: Dict[str, List[float]] = {}
        for engine, _idle_case, _label in ENGINE_ABLATIONS:
            if engine not in per_engine:
                continue
            vals = []
            for g in grids:
                v = per_engine[engine].get(g, 0.0)
                if v < 0:
                    clipped.append(f"{op}/{engine}@{g} ({v:.4g} {value_unit})")
                    v = 0.0
                vals.append(v)
            raw[engine] = vals

        if normalize_pct:
            engines_present = list(raw.keys())
            totals = [sum(raw[e][gi] for e in engines_present) for gi in range(len(grids))]
            for engine in engines_present:
                raw[engine] = [
                    (raw[engine][gi] / totals[gi] * 100.0) if totals[gi] > 0 else 0.0
                    for gi in range(len(grids))
                ]

        bottom = np.zeros(len(grids))
        for engine, _idle_case, _label in ENGINE_ABLATIONS:
            if engine not in raw:
                continue
            vals = np.array(raw[engine])
            ax.bar(
                x + offset, vals, width, bottom=bottom,
                color=engine_colors[engine], hatch=OP_HATCHES[op_idx % len(OP_HATCHES)],
                edgecolor="black", linewidth=0.5,
            )
            bottom += vals

    if clipped:
        print(
            f"[ENERGY] note: clipped {len(clipped)} negative op/engine/grid value(s) to 0 for "
            f"the cross-op stacked chart (noise floor, not a real negative value): {clipped}",
            flush=True,
        )

    engine_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=engine_colors[engine]) for engine, _i, _l in ENGINE_ABLATIONS
    ]
    engine_labels = [label for _e, _i, label in ENGINE_ABLATIONS]
    op_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch=OP_HATCHES[i % len(OP_HATCHES)])
        for i in range(len(ops_with_data))
    ]
    ax.legend(
        engine_handles + op_handles, engine_labels + ops_with_data,
        loc="upper left", fontsize=9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}, per grid, per op ({' vs '.join(ops_with_data)})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}", flush=True)
    return True


def plot_cross_op_energy_by_engine_stacked(
    ops: List[str],
    output_root: Path,
    grid_order: List[str],
    out_path: Path,
    dpi: int,
) -> bool:
    """Cross-op version of plot_energy_by_engine_stacked, in raw dynamic Joules. Returns False
    (and prints a note) if no op has any per-engine data at all.
    """
    per_op_engine: Dict[str, Dict[str, Dict[str, float]]] = {}
    for op in ops:
        per_engine = compute_energy_by_engine(output_root / op, note_prefix=f"[ENERGY][{op}]")
        if per_engine is not None:
            per_op_engine[op] = per_engine

    if not per_op_engine:
        print(f"[ENERGY] note: no op has energy-by-engine data under {output_root}; skipping the cross-op stacked chart", flush=True)
        return False

    return _plot_cross_op_stacked_by_engine(
        per_op_engine, ops, grid_order, out_path, dpi,
        ylabel="Dynamic energy [J]", title="Total dynamic energy by engine", value_unit="J",
    )


def plot_cross_op_energy_per_flop_by_engine_stacked(
    ops: List[str],
    output_root: Path,
    grid_order: List[str],
    flops: float,
    out_path: Path,
    dpi: int,
) -> bool:
    """pJ/FLOP version of plot_cross_op_energy_by_engine_stacked -- same stacked
    reader/writer/compute-per-op layout, but normalized by the fixed FLOP count instead of raw
    Joules. Reader/writer NoC traffic is identical regardless of which per-tile instruction
    runs, so as core count rises and the reader+writer share of the stack stops shrinking while
    compute's share does, that's the NoC (not the FPU) becoming the bottleneck -- see the
    per-op-latency discussion this chart is meant to make visible directly. Returns False (and
    prints a note) if no op has any per-engine data at all.
    """
    per_op_engine: Dict[str, Dict[str, Dict[str, float]]] = {}
    for op in ops:
        per_engine = compute_energy_per_flop_by_engine(output_root / op, flops)
        if per_engine is not None:
            per_op_engine[op] = per_engine

    if not per_op_engine:
        print(f"[ENERGY/FLOP] note: no op has per-engine pJ/FLOP data under {output_root}; skipping the cross-op pJ/FLOP stacked chart", flush=True)
        return False

    return _plot_cross_op_stacked_by_engine(
        per_op_engine, ops, grid_order, out_path, dpi,
        ylabel="Energy per FLOP [pJ]", title="Energy per FLOP by engine", value_unit="pJ/FLOP",
    )


def plot_cross_op_energy_share_by_engine_stacked(
    ops: List[str],
    output_root: Path,
    grid_order: List[str],
    out_path: Path,
    dpi: int,
) -> bool:
    """Percentage-share version: every op's bar at every grid is rescaled to sum to 100%, so the
    only thing varying is how the reader/writer/compute *split* changes -- e.g. compute's share
    shrinking and reader+writer's share growing as core count rises is the NoC-bottleneck signal
    made directly visible, without total pJ/FLOP magnitude (which differs a lot between ops)
    competing for attention. Returns False (and prints a note) if no op has any per-engine data.
    """
    per_op_engine: Dict[str, Dict[str, Dict[str, float]]] = {}
    for op in ops:
        per_engine = compute_energy_by_engine(output_root / op, note_prefix=f"[ENERGY][{op}]")
        if per_engine is not None:
            per_op_engine[op] = per_engine

    if not per_op_engine:
        print(f"[ENERGY] note: no op has energy-by-engine data under {output_root}; skipping the cross-op share chart", flush=True)
        return False

    return _plot_cross_op_stacked_by_engine(
        per_op_engine, ops, grid_order, out_path, dpi,
        ylabel="Share of dynamic energy [%]", title="Reader/Writer/Compute share of dynamic energy",
        value_unit="J", normalize_pct=True,
    )


def compute_total_energy_per_flop_by_op(
    ops: List[str],
    output_root: Path,
    case: str,
    flops: float,
) -> Dict[str, Dict[str, float]]:
    """pJ/FLOP per op (not broken down by engine) for one profile/case: each op's total energy
    (energy_j_vi, including the idle floor) for that case, divided by the fixed FLOP count, per
    grid. This is the same quantity make_pj_per_flop.py plots for a single board/case, just read
    once per op here so the operations can be compared directly. Ops missing this case's CSV
    are skipped with a note.
    """
    per_op: Dict[str, Dict[str, float]] = {}
    for op in ops:
        rows = _load_program_intervals(output_root / op / case / "program_intervals.csv")
        if rows is None:
            print(
                f"[ENERGY/FLOP] note: no {op}/{case}/program_intervals.csv under "
                f"{output_root}, skipping '{op}' for '{case}' in the total-energy-per-FLOP-by-op chart",
                flush=True,
            )
            continue
        per_grid: Dict[str, float] = {}
        for grid, row in rows.items():
            try:
                energy_j = float(row["energy_j_vi"])
            except (KeyError, ValueError):
                continue
            per_grid[grid] = energy_j / flops * 1e12
        if per_grid:
            per_op[op] = per_grid
    return per_op


def plot_total_energy_per_flop_by_op(
    ops: List[str],
    output_root: Path,
    grid_order: List[str],
    flops: float,
    out_path: Path,
    dpi: int,
) -> bool:
    """One figure, same layout as plot_cross_op_energy_by_engine_stacked: for every grid, one
    plain bar per op, side by side. No stacking, no hatching -- just each op's total energy
    (energy_j_vi for ALL_ACTIVE_CASE, including the idle floor) divided by the fixed FLOP count,
    one color per op. This is the un-stacked, un-broken-down total that
    plot_cross_op_energy_by_engine_stacked's bars sum to (plus the shared idle floor that chart
    ablates away). Returns False (and prints a note) if no op has data.
    """
    per_op = compute_total_energy_per_flop_by_op(ops, output_root, ALL_ACTIVE_CASE, flops)
    if not per_op:
        print(f"[ENERGY/FLOP] note: no op has {ALL_ACTIVE_CASE} data under {output_root}; skipping the total-energy-per-FLOP-by-op chart", flush=True)
        return False

    grids = [g for g in grid_order if any(g in per_op.get(op, {}) for op in ops)]
    x = np.arange(len(grids))
    ops_with_data = [op for op in ops if op in per_op]
    width = 0.8 / max(len(ops_with_data), 1)

    fig, ax = plt.subplots(figsize=(max(12, len(grids) * 1.3), 6.5))
    for i, op in enumerate(ops_with_data):
        vals = [per_op[op].get(g, float("nan")) for g in grids]
        offset = (i - (len(ops_with_data) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=op)

    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel("Total energy per FLOP [pJ]")
    ax.set_title(f"Total energy per FLOP, per op ({' vs '.join(ops_with_data)}), all-active case")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}", flush=True)
    return True


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
        "-i", str(output_root),
        "-c", str(cases_file),
        "--dpi", str(dpi),
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
        help=f"Which per-tile operation(s) to sweep (HIGH_POWER_OP). The full --power-cases "
             f"sweep runs once per op, into <output-root>/<op>/. Default: {DEFAULT_OPS}",
    )
    ap.add_argument("--auto-script", type=Path, default=Path(__file__).parent / "auto.py")
    ap.add_argument("--compare-script", type=Path, default=Path(__file__).parent / "compare_runs2.py")
    ap.add_argument("--telemetry-exe", required=True, type=Path)
    ap.add_argument("--telemetry-freq", type=int, default=50)
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
    ap.add_argument("--slot-ms", type=int, default=1)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--trim-ms", type=float, default=1.0)
    ap.add_argument("--tt-smi", default="tt-smi", help="tt-smi executable name or path. Default: tt-smi")
    ap.add_argument("--dpi", type=int, default=150, help="DPI for comparison figures. Default: 150")
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
        help="Re-run a case even if <output-root>/<subdir> already exists. Default: skip existing cases.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every command that would run (reset, auto.py, compare_runs2.py) without executing it.",
    )
    # --app-args must be last (argparse.REMAINDER)
    ap.add_argument(
        "--app-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed to the application (M N K num_iterations [fixed_tiles_per_core]). Put this option last.",
    )

    args = ap.parse_args(argv)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases_in_order: List[Tuple[str, str]] = [POWER_CASE_SPECS[i] for i in args.power_cases]

    print(f"Output root: {output_root}", flush=True)
    print(f"Ops sweep: {args.ops}", flush=True)
    print(f"POWER_CASE sweep: {args.power_cases} -> {[s for s, _ in cases_in_order]}", flush=True)

    # FLOPs per interval, for the energy-per-FLOP-by-engine chart -- fixed regardless of grid
    # since split mode divides the same total work across however many cores are active.
    # --app-args is M N K num_iterations [fixed_tiles_per_core]; the chart is skipped (with a
    # note) if fewer than 4 positional args were given.
    flops: Optional[float] = None
    if len(args.app_args) >= 4:
        try:
            M, N, K, iters = (int(args.app_args[i]) for i in range(4))
            flops = 2.0 * M * N * K * iters
        except ValueError:
            print(
                "[ENERGY/FLOP] note: could not parse M N K num_iterations from --app-args; "
                "skipping the energy-per-FLOP-by-engine chart",
                flush=True,
            )
    else:
        print(
            "[ENERGY/FLOP] note: --app-args has fewer than 4 values (M N K num_iterations); "
            "skipping the energy-per-FLOP-by-engine chart",
            flush=True,
        )

    for op in args.ops:
        op_output_root = output_root / op
        op_output_root.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}", flush=True)
        print(f" op={op} -> {op_output_root}", flush=True)
        print(f"{'#' * 60}", flush=True)

        for power_case, (subdir, label) in zip(args.power_cases, cases_in_order):
            run_dir = op_output_root / subdir
            if run_dir.exists() and not args.force:
                print(f"\n[SKIP] '{op}/{subdir}' already exists. Skipping (use --force to re-run).", flush=True)
                continue

            rc = reset_hardware(args.tt_smi, args.dry_run)
            if rc != 0:
                print(f"ERROR: Hardware reset failed before op={op} POWER_CASE={power_case} ({subdir}). Aborting.", file=sys.stderr)
                return rc

            rc = run_case(
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
                power_case=power_case,
                op=op,
                dry_run=args.dry_run,
            )
            if rc != 0:
                print(f"ERROR: op={op} POWER_CASE={power_case} ({subdir}) failed with return code {rc}. Aborting.", file=sys.stderr)
                return rc

        cases_file = op_output_root / args.cases_file_name
        write_cases_file(cases_file, cases_in_order)

        rc = run_compare(args.compare_script, op_output_root, cases_file, args.dpi, args.dry_run)
        if rc != 0:
            print(f"ERROR: compare_runs2.py failed for op={op} with return code {rc}.", file=sys.stderr)
            return rc

        if not args.dry_run:
            active = _load_program_intervals(op_output_root / ALL_ACTIVE_CASE / "program_intervals.csv")
            grid_order = list(active.keys()) if active else []

            if flops is not None:
                per_engine_pj = compute_energy_per_flop_by_engine(op_output_root, flops)
                if per_engine_pj is not None:
                    plot_energy_per_flop_by_engine(
                        per_engine_pj,
                        grid_order,
                        op_output_root / "compare_runs_out" / "energy_per_flop_by_engine.png",
                        args.dpi,
                    )

                per_case_pj_core = compute_energy_per_flop_per_core(op_output_root, cases_in_order, flops)
                if per_case_pj_core:
                    plot_energy_per_flop_per_core(
                        per_case_pj_core,
                        cases_in_order,
                        grid_order,
                        op_output_root / "compare_runs_out" / "energy_per_flop_per_core.png",
                        args.dpi,
                    )

            per_engine_j = compute_energy_by_engine(op_output_root)
            if per_engine_j is not None:
                plot_energy_by_engine_stacked(
                    per_engine_j,
                    grid_order,
                    op_output_root / "compare_runs_out" / "energy_by_engine_stacked.png",
                    args.dpi,
                )

    cross_op_done = False
    if len(args.ops) > 1 and not args.dry_run:
        print(f"\n{'#' * 60}", flush=True)
        print(f" Cross-op comparison ({' vs '.join(args.ops)})", flush=True)
        print(f"{'#' * 60}", flush=True)

        cross_cases_file = output_root / args.cross_op_cases_file_name
        entries = write_cross_op_cases_file(cross_cases_file, args.ops, cases_in_order, output_root)
        if len(entries) < 2:
            print(
                f"[CROSS-OP] only {len(entries)} run(s) have results; need at least 2 to compare. Skipping.",
                flush=True,
            )
        else:
            rc = run_compare(args.compare_script, output_root, cross_cases_file, args.dpi, args.dry_run)
            if rc != 0:
                print(f"ERROR: compare_runs2.py failed for the cross-op comparison with return code {rc}.", file=sys.stderr)
                return rc
            cross_op_done = True

            active = None
            for op in args.ops:
                active = _load_program_intervals(output_root / op / ALL_ACTIVE_CASE / "program_intervals.csv")
                if active is not None:
                    break
            grid_order = list(active.keys()) if active else []
            plot_cross_op_energy_by_engine_stacked(
                args.ops,
                output_root,
                grid_order,
                output_root / "compare_runs_out" / "energy_by_engine_stacked_cross_op.png",
                args.dpi,
            )
            plot_cross_op_energy_share_by_engine_stacked(
                args.ops,
                output_root,
                grid_order,
                output_root / "compare_runs_out" / "energy_share_by_engine_stacked_cross_op.png",
                args.dpi,
            )

            if flops is not None:
                plot_total_energy_per_flop_by_op(
                    args.ops,
                    output_root,
                    grid_order,
                    flops,
                    output_root / "compare_runs_out" / "total_energy_per_flop_by_op.png",
                    args.dpi,
                )
                plot_cross_op_energy_per_flop_by_engine_stacked(
                    args.ops,
                    output_root,
                    grid_order,
                    flops,
                    output_root / "compare_runs_out" / "energy_per_flop_by_engine_stacked_cross_op.png",
                    args.dpi,
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
