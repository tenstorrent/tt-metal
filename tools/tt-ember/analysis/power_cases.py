# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Analysis steps for a POWER_CASE sweep, shared by run_sweep.py.

Everything here reads the <output-root>/<op>/<case>/program_intervals.csv tree that auto.py and
parser.py write, and produces the per-engine energy ablations and cross-op charts. Moved verbatim
out of run_power_cases.py, which mixed these with the sweep orchestration.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
            f"{note_prefix} note: no {ALL_ACTIVE_CASE}/program_intervals.csv under " f"{op_output_root}, skipping",
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
        engine: {grid: j / flops * 1e12 for grid, j in per_grid.items()} for engine, per_grid in per_engine_j.items()
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
                f"[ENERGY/FLOP/CORE] note: no {subdir}/program_intervals.csv under " f"{op_output_root}, skipping it",
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

    if clipped:
        print(
            f"[ENERGY] note: clipped {len(clipped)} negative op/engine/grid value(s) to 0 for "
            f"the cross-op stacked chart (noise floor, not a real negative value): {clipped}",
            flush=True,
        )

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
        print(
            f"[ENERGY] note: no op has energy-by-engine data under {output_root}; skipping the cross-op stacked chart",
            flush=True,
        )
        return False

    return _plot_cross_op_stacked_by_engine(
        per_op_engine,
        ops,
        grid_order,
        out_path,
        dpi,
        ylabel="Dynamic energy [J]",
        title="Total dynamic energy by engine",
        value_unit="J",
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
        print(
            f"[ENERGY/FLOP] note: no op has per-engine pJ/FLOP data under {output_root}; skipping the cross-op pJ/FLOP stacked chart",
            flush=True,
        )
        return False

    return _plot_cross_op_stacked_by_engine(
        per_op_engine,
        ops,
        grid_order,
        out_path,
        dpi,
        ylabel="Energy per FLOP [pJ]",
        title="Energy per FLOP by engine",
        value_unit="pJ/FLOP",
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
        print(
            f"[ENERGY] note: no op has energy-by-engine data under {output_root}; skipping the cross-op share chart",
            flush=True,
        )
        return False

    return _plot_cross_op_stacked_by_engine(
        per_op_engine,
        ops,
        grid_order,
        out_path,
        dpi,
        ylabel="Share of dynamic energy [%]",
        title="Reader/Writer/Compute share of dynamic energy",
        value_unit="J",
        normalize_pct=True,
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
        print(
            f"[ENERGY/FLOP] note: no op has {ALL_ACTIVE_CASE} data under {output_root}; skipping the total-energy-per-FLOP-by-op chart",
            flush=True,
        )
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
