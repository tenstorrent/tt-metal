#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from math import nan
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from tracy import perf_metrics_common as _mc

OpDict = Dict[str, Any]
DeviceOpsDict = Dict[int, List[OpDict]]


class _TracyCounterView:
    """CounterView over one (op, core)'s Tracy rows; cycles(bank) is the ref cnt of any counter in that bank."""

    _BANK_REF = {
        "FPU": ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"),
        "INSTRN_THREAD": ("THREAD_STALLS_0", "THREAD_STALLS_1", "THREAD_STALLS_2", "THREAD_INSTRUCTIONS_0"),
        "TDMA_PACK": ("PACKER_BUSY", "PACKER0_DEST_READ_REQ", "MATH_NOT_SCOREBOARD_STALLED"),
        "TDMA_UNPACK": ("MATH_INSTRN_AVAILABLE", "SRCA_WRITE_REQ", "UNPACK0_BUSY_THREAD0"),
    }

    @staticmethod
    def _bank_of(name: str) -> str:
        """Bank a counter name belongs to, for captures that hold none of the _BANK_REF anchors (a
        partial group, or Quasar's thread-3 / stall-reason INSTRN counters)."""
        if name.startswith(_mc.L1_CLIENT_PREFIX):
            return "L1_CLIENT"
        if name.startswith("L1_"):
            return "L1"
        if name in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"):
            return "FPU"
        if (
            "_INSTRN_AVAILABLE_" in name
            or name.startswith(("THREAD_STALLS_", "THREAD_INSTRUCTIONS_", "WAITING_FOR_"))
            or name == "ANY_THREAD_STALL"
            or name in _mc.STALL_REASON_COUNTERS.values()
        ):
            return "INSTRN_THREAD"
        if name.startswith(("PACKER_", "DEST_READ_GRANTED_")) or name in (
            "MATH_NOT_STALLED_DEST_WR_PORT",
            "MATH_NOT_SCOREBOARD_STALLED",
        ):
            return "TDMA_PACK"
        return "TDMA_UNPACK"

    def __init__(self, values: dict, refs: dict, is_blackhole: bool = False):
        self._v = values
        self._r = refs
        self._is_blackhole = is_blackhole

    def count(self, bank: str, counter_name: str) -> float:
        return float(self._v.get(counter_name, 0.0))

    def cycles(self, bank: str) -> float:
        for cand in self._BANK_REF.get(bank, ()):
            if cand in self._r:
                return float(self._r[cand])
        # Every counter of a bank shares its ref cnt (for L1_CLIENT it is the wall-clock span of the
        # capture window), so any present member of the bank will do.
        for name, rc in self._r.items():
            if self._bank_of(str(name)) == bank:
                return float(rc)
        return 0.0

    def has(self, counter_name: str) -> bool:
        return counter_name in self._v

    def is_blackhole(self) -> bool:
        return self._is_blackhole


def _is_blackhole(device_arch) -> bool:
    return "blackhole" in str(device_arch).lower()


def _is_quasar(device_arch) -> bool:
    return "quasar" in str(device_arch).lower()


# Quasar l1_client selection -> counter name, re-exported from the engine for the decode path and its tests.
quasar_l1_client_label = _mc.quasar_l1_client_label


def compute_metrics_per_op(perf_counter_df, device_arch=""):
    """Per-op metrics per reader (one BRISC per core on tt-1xx, one NEO each on Quasar), then min/median/max/avg
    per key; Quasar's l1_client rates come from compute_l1_client_metrics."""
    import math

    result = {}
    for op, op_df in perf_counter_df.groupby(["run_host_id", "trace_id_count"]):
        per_core = []
        for _, core_df in op_df.groupby(["core_x", "core_y", "risc_type"]):
            values = dict(zip(core_df["counter type"], core_df["value"]))
            refs = dict(zip(core_df["counter type"], core_df["ref cnt"]))
            view = _TracyCounterView(values, refs, _is_blackhole(device_arch))
            metrics = _mc.compute_metrics(view)
            metrics.update(_mc.compute_l1_client_metrics(view, values.keys()))
            per_core.append(metrics)
        agg = {}
        keys = []
        for d in per_core:
            keys += [k for k in d if k not in keys]
        for key in keys:
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


# Parsed from the enum so the names cannot drift from the compiled ordinals.
try:
    COUNTER_TYPE_NAMES = _mc.perf_counter_type_names()
except OSError:  # installed wheel: no header to parse, use the table shipped with the package
    with open(Path(__file__).with_name("perf_counter_type_names.json")) as _f:
        COUNTER_TYPE_NAMES = {int(k): v for k, v in json.load(_f).items()}


# Columns derive from METRIC_LABELS; three utilizations keep their legacy "Avg ... on full grid" name.
_LEGACY_AVG_GRID_COLUMNS = {
    "SFPU Util": "Avg SFPU util on full grid (%)",
    "FPU Util": "Avg FPU util on full grid (%)",
    "MATH Util": "Avg Math util on full grid (%)",
}
# Re-exported for other consumers (process_ops_logs) so the classification stays single-sourced.
RATIO_LABELS = _mc.RATIO_LABELS
is_ratio_label = _mc.is_ratio_label


def _metric_suffix(label):
    """Display unit for a metric's columns: ' (ratio)' for the unbounded ratio family, else ' (%)'."""
    return " (ratio)" if _mc.is_ratio_label(label) else " (%)"


def _build_perf_counter_csv_headers():
    headers = []
    for label in _mc.METRIC_LABELS.values():
        suffix = _metric_suffix(label)
        headers.append(f"{label} Min{suffix}")
        headers.append(f"{label} Median{suffix}")
        headers.append(f"{label} Max{suffix}")
        headers.append(f"{label} Avg{suffix}")
    # Grid-wide averages over the kernel duration (all cores, idle ones included); only a host+device run can fill them.
    headers.extend(_LEGACY_AVG_GRID_COLUMNS.values())
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
                # Quasar's l1_client records name the run's selection, not an enum value.
                counter_sel = meta_dict.get("counter sel")
                if counter_type_name == "QUASAR_L1_CLIENT_EVENT" and counter_sel is not None:
                    counter_type_name = quasar_l1_client_label(counter_sel)

                risc_type = event[EVENT_RISC_TYPE_IDX]
                neo = meta_dict.get("neo")
                if neo is not None and str(risc_type).startswith("QUASAR_"):
                    # Quasar's DM0 reads all four NEOs; keep each NEO its own reader row.
                    risc_type = f"QUASAR_NEO{neo}"
                perf_counter_events.append(
                    {
                        "run_host_id": metadata["run_host_id"],
                        "trace_id_count": metadata["trace_id_count"],
                        "record time": event[EVENT_TIMESTAMP_IDX],
                        "core_x": event[EVENT_CORE_COORDS_IDX][0],
                        "core_y": event[EVENT_CORE_COORDS_IDX][1],
                        "risc_type": risc_type,
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

    ratio_metrics = [label for label in _mc.METRIC_LABELS.values() if label in RATIO_LABELS]
    pct_metrics = [label for label in _mc.METRIC_LABELS.values() if label not in RATIO_LABELS]
    # Quasar l1_client metrics are named after the run's selection; pick them up from the frame.
    for _suffix, _family in ((" Avg (%)", pct_metrics), (" Avg (ratio)", ratio_metrics)):
        _family.extend(
            sorted(
                col[: -len(_suffix)]
                for col in metrics_df.columns
                if str(col).startswith(_mc.L1_CLIENT_PREFIX) and str(col).endswith(_suffix)
            )
        )

    # For each base metric, display a table with Min/Median/Max/Avg rows
    for base_metric in pct_metrics + ratio_metrics:
        is_ratio = _mc.is_ratio_label(base_metric)
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


def compute_perf_counter_metrics(perf_counter_df, device_arch, total_compute_cores):
    """{per_op_stats, per_op_counts}: engine metrics keyed by METRIC_LABELS, plus raw average counts."""
    per_op = compute_metrics_per_op(perf_counter_df, device_arch)
    per_op_stats = {}
    for op, metrics in per_op.items():
        for key, stats in metrics.items():
            label = _mc.metric_label(key)
            # A metric with no value anywhere gets no column: the two-pass merge finds the second
            # pass's metrics by their absence from the first pass's CSV.
            for stat in ("min", "median", "max", "avg"):
                if stats[stat] is not None:
                    per_op_stats.setdefault(label, {"min": {}, "median": {}, "max": {}, "avg": {}})[stat][op] = stats[
                        stat
                    ]

    per_op_counts = {}
    for out_key, cname in (
        ("avg_sfpu_count", "SFPU_COUNTER"),
        ("avg_fpu_count", "FPU_COUNTER"),
        ("avg_math_count", "MATH_COUNTER"),
    ):
        mask = perf_counter_df["counter type"] == cname
        if mask.any():
            grouped = perf_counter_df[mask].groupby(["run_host_id", "trace_id_count"])["value"]
            if _is_quasar(device_arch):
                # Four NEO readers per core: a per-core divisor would overstate the average ~4x.
                per_op_counts[out_key] = grouped.mean().to_dict()
            else:
                per_op_counts[out_key] = (grouped.sum() / total_compute_cores).to_dict()

    return {"per_op_stats": per_op_stats, "per_op_counts": per_op_counts}


def compute_device_only_metrics(
    perf_counter_df: pd.DataFrame,
    device_arch: str = "",
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """(agg_metrics, eff_summary_rows) from the shared engine; the summary rows keep the historical CSV order."""
    per_op = compute_metrics_per_op(perf_counter_df, device_arch)
    agg_metrics: Dict[str, Dict] = {}
    for op, metrics in per_op.items():
        for key, stats in metrics.items():
            label = _mc.metric_label(key)
            for stat in ("min", "median", "max", "avg"):
                if stats[stat] is not None:
                    agg_metrics.setdefault(label, {"min": {}, "median": {}, "max": {}, "avg": {}})[stat][op] = stats[
                        stat
                    ]

    # Quasar's l1_client metrics are named after the run's selection, so they are appended from the data.
    _ratio_metric_names = [label for label in _mc.METRIC_LABELS.values() if label in RATIO_LABELS]
    _pct_metric_names = [label for label in _mc.METRIC_LABELS.values() if label not in RATIO_LABELS]
    for label in sorted(label for label in agg_metrics if str(label).startswith(_mc.L1_CLIENT_PREFIX)):
        (_ratio_metric_names if _mc.is_ratio_label(label) else _pct_metric_names).append(label)

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
