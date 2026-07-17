#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from math import nan
from loguru import logger


def _import_perf_metrics_common():
    """Import the shared metric/L1-grouping module from tt_metal/tools/profiler.

    Single source of the L1 client-port groupings and the arch-aware port-1 name resolver, also used by
    the LLK test harness (helpers/metrics.py). Path-based import — the profiler dir is not a package.
    """
    import sys
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        cand = parent / "tt_metal" / "tools" / "profiler"
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            break
    import perf_metrics_common

    return perf_metrics_common


_mc = _import_perf_metrics_common()

OpDict = Dict[str, Any]
DeviceOpsDict = Dict[int, List[OpDict]]


class _TracyCounterView:
    """Adapts one (op, core)'s counter rows to perf_metrics_common.CounterView.

    Tracy's device frame gives per-counter `value` + `ref cnt`; the shared engine reads a counter's
    value via count(bank, name) and a bank's reference cycles via cycles(bank). All counters in a bank
    share one ref_cnt, so cycles(bank) returns the ref_cnt of any present counter in that bank.
    """

    _BANK_REF = {
        "FPU": ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"),
        "INSTRN_THREAD": ("THREAD_STALLS_0", "THREAD_STALLS_1", "THREAD_STALLS_2", "THREAD_INSTRUCTIONS_0"),
        "TDMA_PACK": ("PACKER_BUSY", "PACKER_DEST_READ_AVAILABLE", "AVAILABLE_MATH"),
        "TDMA_UNPACK": ("MATH_INSTRN_AVAILABLE", "SRCA_WRITE_AVAILABLE", "UNPACK0_BUSY_THREAD0"),
    }

    def __init__(self, values: dict, refs: dict):
        self._v = values
        self._r = refs

    def count(self, bank: str, counter_name: str) -> float:
        return float(self._v.get(counter_name, 0.0))

    def cycles(self, bank: str) -> float:
        for cand in self._BANK_REF.get(bank, ()):
            if cand in self._r:
                return float(self._r[cand])
        if bank == "L1":  # any L1 client shares the bank ref count
            for name, rc in self._r.items():
                if name.startswith("L1_"):
                    return float(rc)
        return 0.0

    def has(self, counter_name: str) -> bool:
        return counter_name in self._v


def compute_metrics_per_op(perf_counter_df):
    """Derived metrics per op via the shared compute_metrics, computed per core then aggregated.

    For each op (run_host_id, trace_id_count) and each core, build a _TracyCounterView and call the
    shared perf_metrics_common.compute_metrics; then reduce across cores to min/median/max/avg per
    metric key (None/NaN excluded). Returns {op_key: {metric_key: {min, median, max, avg}}}.
    """
    import math

    result = {}
    for op, op_df in perf_counter_df.groupby(["run_host_id", "trace_id_count"]):
        per_core = []
        for _, core_df in op_df.groupby(["core_x", "core_y"]):
            values = dict(zip(core_df["counter type"], core_df["value"]))
            refs = dict(zip(core_df["counter type"], core_df["ref cnt"]))
            per_core.append(_mc.compute_metrics(_TracyCounterView(values, refs)))
        agg = {}
        for key in per_core[0].keys() if per_core else []:
            vals = [
                d[key]
                for d in per_core
                if d.get(key) is not None and not (isinstance(d[key], float) and math.isnan(d[key]))
            ]
            if vals:
                s = sorted(vals)
                n = len(s)
                median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
                agg[key] = {"min": min(vals), "median": median, "max": max(vals), "avg": sum(vals) / n}
            else:
                agg[key] = {"min": None, "median": None, "max": None, "avg": None}
        result[op] = agg
    return result


# Counter type name table, parsed from the PerfCounterType enum in perf_counters.hpp so it can't
# drift from the compiled ordinals (the profiler stores counter_type as that enum ordinal).
COUNTER_TYPE_NAMES = _mc.perf_counter_type_names()


# Perf-counter CSV column allowlist, DERIVED from the shared METRIC_LABELS (single source of truth)
# so it cannot drift from the computed metrics and automatically covers newly added metrics. Each
# metric contributes Min/Median/Max/Avg columns; three utilization metrics keep their legacy
# "Avg ... on full grid" column name. The column UNIT is decided centrally by _metric_suffix:
# bounded metrics get " (%)"; the unbounded ratio family (perf_metrics_common.RATIO_LABELS) gets
# " (ratio)" (raw value, may exceed 1.0).
_LEGACY_AVG_GRID_COLUMNS = {
    "SFPU Util": "Avg SFPU util on full grid (%)",
    "FPU Util": "Avg FPU util on full grid (%)",
    "MATH Util": "Avg Math util on full grid (%)",
}
# Re-exported for other consumers (process_ops_logs) so the classification stays single-sourced.
RATIO_LABELS = _mc.RATIO_LABELS


def _metric_suffix(label):
    """Display unit for a metric's columns: ' (ratio)' for the unbounded ratio family, else ' (%)'."""
    return " (ratio)" if label in RATIO_LABELS else " (%)"


def _build_perf_counter_csv_headers():
    headers = []
    for label in _mc.METRIC_LABELS.values():
        suffix = _metric_suffix(label)
        headers.append(f"{label} Min{suffix}")
        headers.append(f"{label} Median{suffix}")
        headers.append(f"{label} Max{suffix}")
        headers.append(_LEGACY_AVG_GRID_COLUMNS.get(label, f"{label} Avg{suffix}"))
    return headers


PERF_COUNTER_CSV_HEADERS = _build_perf_counter_csv_headers()


def extract_perf_counters(events: List[Any]) -> Optional[pd.DataFrame]:
    # If perf counter data exists, extract relevant columns and return as a dataframe
    EVENT_METADATA_IDX = 0
    EVENT_TIMESTAMP_IDX = 1
    EVENT_RISC_TYPE_IDX = 3
    EVENT_CORE_COORDS_IDX = 4
    PERF_COUNTER_ID = 9090

    try:
        # Process events: extract metadata, add timestamp and coords
        perf_counter_events = []
        for event in events:
            metadata = event[EVENT_METADATA_IDX]
            if metadata["id"] == PERF_COUNTER_ID:
                raw_md = metadata.get("meta_data", "")
                if not raw_md:
                    continue
                try:
                    meta_dict = json.loads(raw_md.replace(";", ",").replace("'", '"'))
                except (json.JSONDecodeError, AttributeError):
                    continue

                # Decode counter type to human-readable name
                counter_type_raw = meta_dict.get("counter type", 0)
                # Handle both integer ID and string name formats
                if isinstance(counter_type_raw, str):
                    counter_type_name = counter_type_raw
                else:
                    counter_type_name = COUNTER_TYPE_NAMES.get(counter_type_raw, f"UNKNOWN_{counter_type_raw}")

                perf_counter_events.append(
                    {
                        "run_host_id": metadata["run_host_id"],
                        "trace_id_count": metadata["trace_id_count"],
                        "record time": event[EVENT_TIMESTAMP_IDX],
                        "core_x": event[EVENT_CORE_COORDS_IDX][0],
                        "core_y": event[EVENT_CORE_COORDS_IDX][1],
                        "risc_type": event[EVENT_RISC_TYPE_IDX],
                        "counter type": counter_type_name,  # Use human-readable name
                        "value": meta_dict.get("value", 0),
                        "ref cnt": meta_dict.get("ref cnt", 0),
                    }
                )

        if perf_counter_events:
            return pd.DataFrame(perf_counter_events)
    except (KeyError, TypeError, AttributeError) as e:
        logger.exception("Failed to extract perf counter events: %s", e)
    return None


def print_counter_statistics_summary(perf_counter_df: pd.DataFrame, device_id: int) -> None:
    """Print statistics for all raw performance counters."""
    if perf_counter_df is None or perf_counter_df.empty:
        return

    print("\n" + "=" * 100)
    print(f"PERFORMANCE COUNTER STATISTICS - DEVICE {device_id}")
    print("=" * 100)

    # Group by operation
    grouped = perf_counter_df.groupby(["run_host_id", "trace_id_count"])
    total_ops = len(grouped)

    print(f"\nTotal operations with counter data: {total_ops}")

    # Get all unique counter types
    counter_types = sorted(perf_counter_df["counter type"].unique())

    print("\n" + "=" * 100)
    print("RAW COUNTER VALUES")
    print("=" * 100)
    print(f"{'Counter Type':<40} {'Statistic':<12} {'Ops':>8} {'Min':>15} {'Median':>15} {'Max':>15} {'Avg':>15}")
    print("-" * 100)

    for counter_type in counter_types:
        counter_data = perf_counter_df[perf_counter_df["counter type"] == counter_type]
        counter_grouped = counter_data.groupby(["run_host_id", "trace_id_count"])

        # Calculate statistics across operations
        min_vals = counter_grouped["value"].min()
        median_vals = counter_grouped["value"].median()
        max_vals = counter_grouped["value"].max()
        avg_vals = counter_grouped["value"].mean()

        ops_with_data = len(counter_grouped)

        # Print value statistics
        print(
            f"{counter_type:<40} {'Value':<12} {ops_with_data:>8} "
            f"{min_vals.min():>15.1f} {median_vals.median():>15.1f} "
            f"{max_vals.max():>15.1f} {avg_vals.mean():>15.1f}"
        )

    print("\n" + "=" * 100 + "\n")


def print_efficiency_metrics_summary(metrics_df: pd.DataFrame, device_id: int) -> None:
    """Print a summary of calculated efficiency metrics grouped by metric type."""
    if metrics_df is None or metrics_df.empty:
        return

    print("\n" + "=" * 100)
    print(f"EFFICIENCY METRICS SUMMARY - DEVICE {device_id}")
    print("=" * 100)

    print(f"\nTotal operations with metrics: {len(metrics_df)}")

    # Metric display names derived from the shared METRIC_LABELS (single source): the unbounded ratio
    # family (RATIO_LABELS) prints with a "(ratio)" unit and raw values; every other metric is a
    # bounded percentage. Deriving here means new metrics appear automatically and can't drift.
    ratio_metrics = [label for label in _mc.METRIC_LABELS.values() if label in RATIO_LABELS]
    pct_metrics = [label for label in _mc.METRIC_LABELS.values() if label not in RATIO_LABELS]

    # For each base metric, display a table with Min/Median/Max/Avg rows
    for base_metric in pct_metrics + ratio_metrics:
        is_ratio = base_metric in RATIO_LABELS
        suffix = " (ratio)" if is_ratio else " (%)"
        unit = "" if is_ratio else "%"

        # Skip metrics that have no data columns (e.g. BH-dead counters)
        avg_col = f"{base_metric} Avg{suffix}"
        if avg_col not in metrics_df.columns or metrics_df[avg_col].dropna().empty:
            continue

        print("\n" + "=" * 80)
        print(f"{base_metric.upper()}")
        print("=" * 80)

        # Create table header
        print(f"{'Statistic':<12} {'Ops with Data':>15} {'Range':>30} {'Mean':>12}")
        print("-" * 80)

        # Check each statistic
        total_ops = len(metrics_df)
        for stat in ["Min", "Median", "Max", "Avg"]:
            col_name = f"{base_metric} {stat}{suffix}"
            if col_name in metrics_df.columns:
                non_nan = metrics_df[col_name].dropna()
                if len(non_nan) > 0:
                    ops_with_data = f"{len(non_nan)}/{total_ops}"
                    range_str = f"{non_nan.min():.2f}{unit} - {non_nan.max():.2f}{unit}"
                    mean_str = f"{non_nan.mean():.2f}{unit}"
                else:
                    ops_with_data = f"0/{total_ops}"
                    range_str = "N/A"
                    mean_str = "N/A"

                print(f"{stat:<12} {ops_with_data:>15} {range_str:>30} {mean_str:>12}")

    print("\n" + "=" * 100 + "\n")


def compute_perf_counter_metrics(perf_counter_df, total_compute_cores):
    """Compute per-op perf counter metrics and return {per_op_stats, per_op_counts}.

    Metric FORMULAS come from the shared perf_metrics_common.compute_metrics (single source, also used
    by the LLK harness): computed per (op, core) via compute_metrics_per_op, reduced to min/median/max/
    avg per op, and keyed by the shared METRIC_LABELS display names. per_op_counts (avg raw counts on
    the full grid) are counts, not derived ratios, so they stay computed locally here.
    """
    per_op = compute_metrics_per_op(perf_counter_df)
    per_op_stats = {}
    for op, metrics in per_op.items():
        for key, stats in metrics.items():
            label = _mc.METRIC_LABELS.get(key, key)
            slot = per_op_stats.setdefault(label, {"min": {}, "median": {}, "max": {}, "avg": {}})
            for stat in ("min", "median", "max", "avg"):
                if stats[stat] is not None:
                    slot[stat][op] = stats[stat]

    per_op_counts = {}
    for out_key, cname in (
        ("avg_sfpu_count", "SFPU_COUNTER"),
        ("avg_fpu_count", "FPU_COUNTER"),
        ("avg_math_count", "MATH_COUNTER"),
    ):
        mask = perf_counter_df["counter type"] == cname
        if mask.any():
            per_op_counts[out_key] = (
                perf_counter_df[mask].groupby(["run_host_id", "trace_id_count"])["value"].sum() / total_compute_cores
            ).to_dict()

    return {"per_op_stats": per_op_stats, "per_op_counts": per_op_counts}


def compute_device_only_metrics(
    perf_counter_df: pd.DataFrame,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Compute device-only efficiency metrics; returns (agg_metrics, eff_summary_rows).

    Formulas come from the shared perf_metrics_common.compute_metrics (single source, also used by the
    LLK harness): computed per (op, core) via compute_metrics_per_op, reduced to min/median/max/avg per
    op, keyed by METRIC_LABELS display names. eff_summary_rows keeps the historical CSV columns/order.
    """
    per_op = compute_metrics_per_op(perf_counter_df)
    agg_metrics: Dict[str, Dict] = {}
    for op, metrics in per_op.items():
        for key, stats in metrics.items():
            label = _mc.METRIC_LABELS.get(key, key)
            slot = agg_metrics.setdefault(label, {"min": {}, "median": {}, "max": {}, "avg": {}})
            for stat in ("min", "median", "max", "avg"):
                if stats[stat] is not None:
                    slot[stat][op] = stats[stat]

    # Device-only summary metric names, derived from the shared METRIC_LABELS (single source) so the
    # summary can't drift and new metrics appear automatically. The unbounded ratio family gets a
    # "(ratio)" unit; every other metric (including the instruction-issue rates) is a percentage.
    _ratio_metric_names = [label for label in _mc.METRIC_LABELS.values() if label in RATIO_LABELS]
    _pct_metric_names = [label for label in _mc.METRIC_LABELS.values() if label not in RATIO_LABELS]

    eff_summary_rows: List[Dict] = []
    first_stat = next(iter(agg_metrics.values()), {}).get("min", {})
    for key in first_stat.keys():
        row: Dict[str, object] = {}
        for base_name in _pct_metric_names:
            if base_name in agg_metrics:
                mm = agg_metrics[base_name]
                for stat in ["min", "median", "max", "avg"]:
                    stat_cap = stat.capitalize() if stat != "avg" else "Avg"
                    row[f"{base_name} {stat_cap} (%)"] = mm[stat].get(key, nan)
        for base_name in _ratio_metric_names:
            if base_name in agg_metrics:
                mm = agg_metrics[base_name]
                for stat in ["min", "median", "max", "avg"]:
                    stat_cap = stat.capitalize() if stat != "avg" else "Avg"
                    row[f"{base_name} {stat_cap} (ratio)"] = mm[stat].get(key, nan)
        eff_summary_rows.append(row)

    return agg_metrics, eff_summary_rows


def get_device_op_data(ops: Dict[int, OpDict], host_device_op_compare) -> Tuple[DeviceOpsDict, bool]:
    """Group host ops per device and record whether trace runs exist."""

    logger.info(f"Getting device ops")
    deviceOps = {}
    hasTraceRuns = False
    for opID, opData in ops.items():
        if "device_id" in opData:
            deviceID = opData["device_id"]
            if deviceID not in deviceOps:
                deviceOps[deviceID] = [opData]
            else:
                deviceOps[deviceID].append(opData)
        if "metal_trace_id" in opData and opData["metal_trace_id"] is not None:
            hasTraceRuns = True

    for deviceID in deviceOps:
        deviceOps[deviceID].sort(key=host_device_op_compare)

    return deviceOps, hasTraceRuns
