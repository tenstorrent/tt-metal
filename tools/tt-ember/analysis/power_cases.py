# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Analysis steps for a POWER_CASE sweep, used by run_sweep.py.

Everything here reads the <output-root>/<op>/<case>/program_intervals.csv tree that auto.py and
parser.py write, and produces the per-engine energy ablations and cross-op charts. This is the
analysis half of the former run_power_cases.py; run_sweep.py does the runs.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

from common import run_status, whole_run_value
from compare_cases import grouped_bar_chart

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
# as opposed to the writer's full on/off contribution.
POWER_CASE_SPECS: Dict[int, Tuple[str, str]] = {
    0: ("regular", "Regular (baseline)"),
    1: ("writer_amp", "All active"),
    2: ("compute_idle", "Compute idle"),
    3: ("reader_compute_idle", "Reader+Compute idle"),
    4: ("reader_idle2", "Reader idle"),
    5: ("writer_idle", "Writer idle"),
}
SUBDIR_TO_POWER_CASE: Dict[str, int] = {subdir: case for case, (subdir, _label) in POWER_CASE_SPECS.items()}

# Which per-tile instruction the compute kernel runs (LONG_MATMUL_OP, see long_matmul's
# kernels/compute/compute.cpp).
ALL_OPS: List[str] = ["matmul", "add", "silu", "exp", "sigmoid", "gelu", "recip"]


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


@dataclass
class Runs:
    """The runs of one sweep, and how to read them. A run is read only if its run_complete.json
    records the settings the sweep gave it, so a run made with other settings (another shape or
    mode), or one that did not finish, never reaches a chart."""

    output_root: Path
    # "<op>/<case>" -> the settings that run was made with (common.run_settings)
    settings_by_run: Dict[str, Dict[str, object]]
    # LONG_MATMUL_MODE=legacy: use the published analysis (see common.whole_run_value)
    legacy: bool

    def complete(self, op: str, subdir: str) -> bool:
        settings = self.settings_by_run.get(f"{op}/{subdir}")
        return settings is not None and run_status(self.output_root / op / subdir, settings) == "done"

    def rows(self, op: str, subdir: str, note_prefix: str) -> Optional[Dict[str, Dict[str, str]]]:
        """program_intervals.csv of op/subdir by grid, or None (with a note) if that run is not a
        complete run of this sweep."""
        if not self.complete(op, subdir):
            print(f"{note_prefix} note: no complete '{op}/{subdir}' run in this sweep, skipping it", flush=True)
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
            active_as = whole_run_value(active_row, "dynamic_charge_avg_as", runs.legacy)
            idle_as = whole_run_value(idle[grid], "dynamic_charge_avg_as", runs.legacy)
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
            energy_j = whole_run_value(row, "energy_j_vi", runs.legacy)
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
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
            energy_j = whole_run_value(row, "energy_j_vi", runs.legacy)
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
