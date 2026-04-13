# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance metrics derived from Wormhole hardware counters.

All metrics are bounded 0-100% unless noted otherwise.

== Compute Utilization ==
- fpu_utilization: FPU_INSTRUCTION / ref_cycles (FPU bank)
    % of total time the FPU was executing matrix ops.
- compute_utilization: FPU_OR_SFPU_INSTRN / ref_cycles (FPU bank)
    % of total time either FPU or SFPU was active (combined).

== Thread Stall Rates (INSTRN_THREAD bank, normalized by ref_cycles) ==
- unpack_thread_stall_rate: THREAD_STALLS_0 / ref_cycles
    % of time the unpack thread (T0) was stalled.
- math_thread_stall_rate: THREAD_STALLS_1 / ref_cycles
    % of time the math thread (T1) was stalled.
- pack_thread_stall_rate: THREAD_STALLS_2 / ref_cycles
    % of time the pack thread (T2) was stalled.

== Semaphore Wait Rates (INSTRN_THREAD bank, normalized by ref_cycles) ==
- math_sem_wait_rate: WAITING_FOR_NONZERO_SEM_1 / ref_cycles
    % of time the math thread was waiting on a semaphore > 0 (data not ready).
- pack_sem_wait_rate: WAITING_FOR_NONZERO_SEM_2 / ref_cycles
    % of time the pack thread was waiting on a semaphore > 0.

== Unpacker Metrics (TDMA_UNPACK bank) ==
- unpack0_write_efficiency: SRCA_WRITE / UNPACK0_BUSY_THREAD0
    Fraction of unpacker0 busy cycles actually writing to srcA.
- unpack1_write_efficiency: SRCB_WRITE / UNPACK1_BUSY_THREAD0
    Fraction of unpacker1 busy cycles actually writing to srcB.
- unpack_write_efficiency: average of unpack0 + unpack1.
- unpack_to_math_flow0: SRCA_WRITE_AVAILABLE / UNPACK0_BUSY_THREAD0
    srcA buffer availability during unpack — high = no backpressure from math.
- unpack_to_math_flow1: SRCB_WRITE_AVAILABLE / UNPACK1_BUSY_THREAD0
    srcB buffer availability during unpack.
- unpack_to_math_flow: average of flow0 + flow1.

== Packer Metrics (TDMA_PACK bank) ==
- pack_utilization: PACKER_BUSY / ref_cycles (TDMA_PACK bank)
    % of total time any packer engine was busy.
- pack_dest_eff: PACKER_DEST_READ_AVAILABLE_0 / PACKER_BUSY
    Fraction of packer busy time where dest data was available to read.

== Math Pipeline Stalls (TDMA_UNPACK bank — same bank, reliable) ==
- fidelity_stall_rate: FIDELITY_PHASE_STALLS / MATH_INSTRN_AVAILABLE
    % of math-available cycles stalled by HiFi fidelity phases. 0% at LoFi.
- math_src_stall_rate: 1 - (MATH_NOT_BLOCKED_BY_SRC / MATH_INSTRN_AVAILABLE)
    % of math-available cycles where source data was NOT ready.
"""

import pandas as pd
from loguru import logger

# ── Helpers ──────────────────────────────────────────────────────────


def _avg_count(df: pd.DataFrame, bank: str, counter_name: str) -> float:
    """Average count for a specific counter across all threads."""
    mask = (df["bank"] == bank) & (df["counter_name"] == counter_name)
    result = df.loc[mask, "count"]
    return float(result.mean()) if len(result) > 0 else 0.0


def _avg_cycles(df: pd.DataFrame, bank: str) -> float:
    """Average reference cycle count for a bank (from any counter in that bank)."""
    mask = df["bank"] == bank
    result = df.loc[mask, "cycles"]
    return float(result.mean()) if len(result) > 0 else 0.0


def _safe_div(numerator: float, denominator: float) -> float | None:
    """Safe division returning None if denominator is 0."""
    return (numerator / denominator) if denominator > 0 else None


def _pct(value: float | None) -> float | None:
    """Convert ratio to percentage."""
    return (value * 100.0) if value is not None else None


def _one_minus(value: float | None) -> float | None:
    """Compute 1.0 - value, for inverting 'not stalled' into 'stalled'."""
    return (1.0 - value) if value is not None else None


def _avg_pair(a: float | None, b: float | None) -> float | None:
    """Average of two optional values."""
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


# ── Compute ──────────────────────────────────────────────────────────


def _compute_single(df: pd.DataFrame) -> dict:
    """
    Compute derived efficiency metrics from a single (zone, run) slice of counter data.

    Returns a flat dict of efficiency percentages (all bounded 0-100%).
    """
    if df.empty:
        return {}

    # ── Reference cycles per bank ──
    fpu_cycles = _avg_cycles(df, "FPU")
    instrn_cycles = _avg_cycles(df, "INSTRN_THREAD")
    unpack_cycles = _avg_cycles(df, "TDMA_UNPACK")
    pack_cycles = _avg_cycles(df, "TDMA_PACK")

    # ── Compute Utilization (FPU bank) ──
    fpu_instruction = _avg_count(df, "FPU", "FPU_INSTRUCTION")
    fpu_or_sfpu = _avg_count(df, "FPU", "FPU_OR_SFPU_INSTRN")
    fpu_utilization = _safe_div(fpu_instruction, fpu_cycles)
    compute_utilization = _safe_div(fpu_or_sfpu, fpu_cycles)

    # ── Thread Stall Rates (INSTRN_THREAD bank) ──
    stalls_0 = _avg_count(df, "INSTRN_THREAD", "THREAD_STALLS_0")
    stalls_1 = _avg_count(df, "INSTRN_THREAD", "THREAD_STALLS_1")
    stalls_2 = _avg_count(df, "INSTRN_THREAD", "THREAD_STALLS_2")
    unpack_thread_stall = _safe_div(stalls_0, instrn_cycles)
    math_thread_stall = _safe_div(stalls_1, instrn_cycles)
    pack_thread_stall = _safe_div(stalls_2, instrn_cycles)

    # ── Semaphore Wait Rates (INSTRN_THREAD bank) ──
    sem_wait_1 = _avg_count(df, "INSTRN_THREAD", "WAITING_FOR_NONZERO_SEM_1")
    sem_wait_2 = _avg_count(df, "INSTRN_THREAD", "WAITING_FOR_NONZERO_SEM_2")
    math_sem_wait = _safe_div(sem_wait_1, instrn_cycles)
    pack_sem_wait = _safe_div(sem_wait_2, instrn_cycles)

    # ── Unpacker Write Efficiency (TDMA_UNPACK bank) ──
    srca_write = _avg_count(df, "TDMA_UNPACK", "SRCA_WRITE")
    srcb_write = _avg_count(df, "TDMA_UNPACK", "SRCB_WRITE")
    unpack0_busy = _avg_count(df, "TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    unpack1_busy = _avg_count(df, "TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")
    unpack0_eff = _safe_div(srca_write, unpack0_busy)
    unpack1_eff = _safe_div(srcb_write, unpack1_busy)
    unpack_eff = _avg_pair(unpack0_eff, unpack1_eff)

    # ── Unpacker-to-Math Data Flow (TDMA_UNPACK bank) ──
    srca_avail = _avg_count(df, "TDMA_UNPACK", "SRCA_WRITE_AVAILABLE")
    srcb_avail = _avg_count(df, "TDMA_UNPACK", "SRCB_WRITE_AVAILABLE")
    flow0 = _safe_div(srca_avail, unpack0_busy)
    flow1 = _safe_div(srcb_avail, unpack1_busy)
    flow_avg = _avg_pair(flow0, flow1)

    # ── Packer Metrics (TDMA_PACK bank) ──
    packer_busy = _avg_count(df, "TDMA_PACK", "PACKER_BUSY")
    pack_utilization = _safe_div(packer_busy, pack_cycles)
    # Pack dest efficiency: use per-engine matching pair (DEST_READ_AVAILABLE_0 / PACKER_BUSY_0)
    # to avoid cross-engine mismatch that causes >100%
    dest_read_0 = _avg_count(df, "TDMA_PACK", "PACKER_DEST_READ_AVAILABLE_0")
    packer_busy_0 = _avg_count(df, "TDMA_PACK", "PACKER_BUSY_0")
    pack_dest_eff = _safe_div(dest_read_0, packer_busy_0)

    # ── Math Pipeline Stalls (TDMA_UNPACK bank only — same bank, reliable) ──
    math_available = _avg_count(df, "TDMA_UNPACK", "MATH_INSTRN_AVAILABLE")
    math_started = _avg_count(df, "TDMA_UNPACK", "MATH_INSTRN_STARTED")
    fidelity_stalls = _avg_count(df, "TDMA_UNPACK", "FIDELITY_PHASE_STALLS")
    math_not_blocked = _avg_count(df, "TDMA_UNPACK", "MATH_NOT_BLOCKED_BY_SRC")

    fidelity_stall_rate = _safe_div(fidelity_stalls, math_available)
    # Math src data stall: fraction of math-available cycles where src data was NOT ready
    math_src_stall_rate = _one_minus(_safe_div(math_not_blocked, math_available))

    return {
        # Compute utilization
        "fpu_utilization_pct": _pct(fpu_utilization),
        "compute_utilization_pct": _pct(compute_utilization),
        # Thread stall rates
        "unpack_thread_stall_pct": _pct(unpack_thread_stall),
        "math_thread_stall_pct": _pct(math_thread_stall),
        "pack_thread_stall_pct": _pct(pack_thread_stall),
        # Semaphore waits
        "math_sem_wait_pct": _pct(math_sem_wait),
        "pack_sem_wait_pct": _pct(pack_sem_wait),
        # Unpacker write efficiency
        "unpack0_write_eff_pct": _pct(unpack0_eff),
        "unpack1_write_eff_pct": _pct(unpack1_eff),
        "unpack_write_eff_pct": _pct(unpack_eff),
        # Unpacker-to-math flow
        "unpack_to_math_flow0_pct": _pct(flow0),
        "unpack_to_math_flow1_pct": _pct(flow1),
        "unpack_to_math_flow_pct": _pct(flow_avg),
        # Packer metrics
        "pack_utilization_pct": _pct(pack_utilization),
        "pack_dest_eff_pct": _pct(pack_dest_eff),
        # Math pipeline stalls
        "fidelity_stall_pct": _pct(fidelity_stall_rate),
        "math_src_stall_pct": _pct(math_src_stall_rate),
    }


def compute_metrics(df: pd.DataFrame) -> list[dict]:
    """
    Compute derived metrics for each (zone, run_index) combination.

    Args:
        df: Raw counter DataFrame from read_counters(), optionally with
            'zone' and 'run_index' columns.

    Returns:
        List of dicts, each containing zone, run_index, and all computed metrics.
    """
    if df.empty:
        return []

    zones = sorted(df["zone"].unique()) if "zone" in df.columns else ["ZONE_0"]
    has_runs = "run_index" in df.columns

    results = []
    for zone in zones:
        zone_df = df[df["zone"] == zone] if "zone" in df.columns else df
        runs = sorted(zone_df["run_index"].unique()) if has_runs else [0]

        for run_idx in runs:
            run_df = zone_df[zone_df["run_index"] == run_idx] if has_runs else zone_df
            metrics = _compute_single(run_df)
            if metrics:
                metrics["zone"] = zone
                metrics["run_index"] = run_idx
                results.append(metrics)

    return results


# ── Export ────────────────────────────────────────────────────────────


def export_metrics(
    computed: list[dict],
    run_type_name: str,
    zone_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate computed metrics per zone and return a DataFrame for CSV export.

    For multiple runs: exports mean/std per metric.
    For single run: exports raw values.

    Args:
        computed: Output of compute_metrics().
        run_type_name: Run type prefix for column names (e.g., "L1_TO_L1").
        zone_names: Optional list mapping zone index to display name.
                    e.g., ["INIT", "TILE_LOOP"] maps ZONE_0→INIT, ZONE_1→TILE_LOOP.

    Returns:
        DataFrame with one row per zone, columns prefixed with run_type_name.
    """
    if not computed:
        return pd.DataFrame()

    zone_to_marker = {}
    if zone_names:
        for i, name in enumerate(zone_names):
            zone_to_marker[f"ZONE_{i}"] = name

    zones = sorted(set(m["zone"] for m in computed))
    rows = []

    for zone in zones:
        zone_metrics = [m for m in computed if m["zone"] == zone]
        marker_name = zone_to_marker.get(zone, zone)
        row = {"marker": marker_name}

        # Only export efficiency percentages to the main CSV
        def _exportable(key: str) -> bool:
            return key.endswith("_pct")

        if len(zone_metrics) >= 2:
            metrics_df = pd.DataFrame(zone_metrics)
            for col in metrics_df.columns:
                if not _exportable(col):
                    continue
                values = metrics_df[col].dropna()
                if len(values) >= 2:
                    row[f"{run_type_name}_mean({col})"] = float(values.mean())
                    row[f"{run_type_name}_std({col})"] = float(values.std())
        else:
            for k, v in zone_metrics[0].items():
                if not _exportable(k):
                    continue
                row[f"{run_type_name}_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


# ── Counter CSV Export ────────────────────────────────────────────────


def export_counters(
    all_counters: pd.DataFrame,
    run_type_name: str,
    zone_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Export raw hardware counter values as a DataFrame for a separate counters CSV.

    Produces one row per zone with columns: marker, then
    ``{run_type_name}_mean({bank}.{counter_name})`` and
    ``{run_type_name}_std({bank}.{counter_name})`` for every counter observed.

    Args:
        all_counters: Concatenated raw counter DataFrame from read_counters()
                      (with ``zone`` and ``run_index`` columns).
        run_type_name: Run type prefix for column names (e.g., "L1_TO_L1").
        zone_names: Optional list mapping zone index to display name.

    Returns:
        DataFrame with one row per zone.
    """
    if all_counters.empty:
        return pd.DataFrame()

    zone_to_marker = {}
    if zone_names:
        for i, name in enumerate(zone_names):
            zone_to_marker[f"ZONE_{i}"] = name

    zones = sorted(all_counters["zone"].unique())
    has_runs = "run_index" in all_counters.columns
    rows = []

    for zone in zones:
        zone_df = all_counters[all_counters["zone"] == zone]
        marker_name = zone_to_marker.get(zone, zone)
        row = {"marker": marker_name}

        # Get unique counters in this zone (preserving discovery order)
        counter_keys = (
            zone_df[["bank", "counter_name"]].drop_duplicates().values.tolist()
        )

        for bank, counter_name in counter_keys:
            mask = (zone_df["bank"] == bank) & (zone_df["counter_name"] == counter_name)
            col_name = f"{bank}.{counter_name}"

            if has_runs:
                per_run = zone_df.loc[mask].groupby("run_index")["count"].mean()
                if len(per_run) >= 2:
                    row[f"{run_type_name}_mean({col_name})"] = float(per_run.mean())
                    row[f"{run_type_name}_std({col_name})"] = float(per_run.std())
                elif len(per_run) == 1:
                    row[f"{run_type_name}_{col_name}"] = float(per_run.iloc[0])
            else:
                values = zone_df.loc[mask, "count"]
                row[f"{run_type_name}_{col_name}"] = float(values.mean())

            # Also export cycles for this counter
            col_cycles = f"{col_name}.cycles"
            if has_runs:
                per_run_cyc = zone_df.loc[mask].groupby("run_index")["cycles"].mean()
                if len(per_run_cyc) >= 2:
                    row[f"{run_type_name}_mean({col_cycles})"] = float(per_run_cyc.mean())
                    row[f"{run_type_name}_std({col_cycles})"] = float(per_run_cyc.std())
                elif len(per_run_cyc) == 1:
                    row[f"{run_type_name}_{col_cycles}"] = float(per_run_cyc.iloc[0])
            else:
                cyc_values = zone_df.loc[mask, "cycles"]
                row[f"{run_type_name}_{col_cycles}"] = float(cyc_values.mean())

        rows.append(row)

    return pd.DataFrame(rows)


# ── Print ────────────────────────────────────────────────────────────


def _print_detail(metrics: dict) -> None:
    """Log detailed efficiency metrics for a single (zone, run) result."""

    def fmt(value, decimals=2):
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}%"

    m = metrics
    sep = "─" * 70

    lines = [
        f"\n{sep}",
        "  COMPUTE UTILIZATION",
        sep,
        f"  {'FPU Utilization:':<40} {fmt(m.get('fpu_utilization_pct')):>12}",
        f"  {'Compute (FPU+SFPU) Utilization:':<40} {fmt(m.get('compute_utilization_pct')):>12}",
        f"\n{sep}",
        "  THREAD STALL RATES",
        sep,
        f"  {'Unpack Thread (T0) Stall:':<40} {fmt(m.get('unpack_thread_stall_pct')):>12}",
        f"  {'Math Thread (T1) Stall:':<40} {fmt(m.get('math_thread_stall_pct')):>12}",
        f"  {'Pack Thread (T2) Stall:':<40} {fmt(m.get('pack_thread_stall_pct')):>12}",
        f"\n{sep}",
        "  SEMAPHORE WAIT RATES",
        sep,
        f"  {'Math Semaphore Wait:':<40} {fmt(m.get('math_sem_wait_pct')):>12}",
        f"  {'Pack Semaphore Wait:':<40} {fmt(m.get('pack_sem_wait_pct')):>12}",
        f"\n{sep}",
        "  UNPACKER WRITE EFFICIENCY",
        sep,
        f"  {'Unpacker0 (srcA):':<40} {fmt(m.get('unpack0_write_eff_pct')):>12}",
        f"  {'Unpacker1 (srcB):':<40} {fmt(m.get('unpack1_write_eff_pct')):>12}",
        f"  {'Combined:':<40} {fmt(m.get('unpack_write_eff_pct')):>12}",
        f"\n{sep}",
        "  UNPACKER-TO-MATH DATA FLOW",
        sep,
        f"  {'srcA Buffer Availability:':<40} {fmt(m.get('unpack_to_math_flow0_pct')):>12}",
        f"  {'srcB Buffer Availability:':<40} {fmt(m.get('unpack_to_math_flow1_pct')):>12}",
        f"  {'Combined:':<40} {fmt(m.get('unpack_to_math_flow_pct')):>12}",
        f"\n{sep}",
        "  PACKER METRICS",
        sep,
        f"  {'Pack Utilization:':<40} {fmt(m.get('pack_utilization_pct')):>12}",
        f"  {'Pack Dest Data Efficiency:':<40} {fmt(m.get('pack_dest_eff_pct')):>12}",
        f"\n{sep}",
        "  MATH PIPELINE STALLS",
        sep,
        f"  {'Fidelity Phase Stall:':<40} {fmt(m.get('fidelity_stall_pct')):>12}",
        f"  {'Math Src Data Stall:':<40} {fmt(m.get('math_src_stall_pct')):>12}",
    ]
    logger.info("\n".join(lines))


def _print_stability(zone_metrics: list[dict]) -> None:
    """Log mean/std summary for multiple runs of the same zone."""
    if len(zone_metrics) < 2:
        return

    metrics_df = pd.DataFrame(zone_metrics)

    pct_cols = [c for c in metrics_df.columns if c.endswith("_pct")]

    lines = [
        f"\n  STABILITY ACROSS {len(zone_metrics)} RUNS (mean +/- std)",
        f"  {'─' * 66}",
        f"  {'Metric':<40} {'Mean':>12} {'Std':>12}",
        f"  {'─' * 40} {'─' * 12} {'─' * 12}",
    ]

    for col in pct_cols:
        values = metrics_df[col].dropna()
        if len(values) >= 2:
            mean_val = float(values.mean())
            std_val = float(values.std())
            label = col.replace("_pct", "").replace("_", " ")
            lines.append(
                f"  {label:<40} {mean_val:>11.2f}% {std_val:>11.2f}%"
            )

    logger.info("\n".join(lines))


def print_metrics(df_or_computed) -> None:
    """
    Log performance metrics, grouped by zone.
    If multiple runs, also logs mean/std stability summary per zone.

    Accepts either:
    - A raw counter DataFrame (computes metrics automatically)
    - A list of dicts from compute_metrics()
    """
    if isinstance(df_or_computed, pd.DataFrame):
        computed = compute_metrics(df_or_computed)
    else:
        computed = df_or_computed

    if not computed:
        logger.info("No metrics to display.")
        return

    logger.info("\n{}\nPERFORMANCE METRICS\n{}", "=" * 70, "=" * 70)

    zones = sorted(set(m["zone"] for m in computed))

    for zone in zones:
        zone_metrics = [m for m in computed if m["zone"] == zone]

        logger.info("\n{}\nZONE: {}\n{}", "═" * 70, zone, "═" * 70)

        # Print detailed metrics for the last run (most representative, after warmup)
        _print_detail(zone_metrics[-1])

        # Print stability summary if multiple runs
        _print_stability(zone_metrics)
