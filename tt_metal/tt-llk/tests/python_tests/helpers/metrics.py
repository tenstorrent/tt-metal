# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance metrics from the LLK test counter dump.

The metric FORMULAS live in the shared module tt_metal/tools/profiler/perf_metrics_common.py
(single source of truth, also used by the tt-metal Tracy tool). This file only adapts the
counters.py DataFrame to that module's CounterView, then aggregates / exports results to CSV.
"""

import sys
from pathlib import Path

import pandas as pd


def _import_perf_metrics_common():
    """Locate + import the shared metric-formula module from tt_metal/tools/profiler."""
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tt_metal" / "tools" / "profiler"
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            break
    import perf_metrics_common

    return perf_metrics_common


_mc = _import_perf_metrics_common()


class _DfCounterView:
    """Adapts the counters.py long-form DataFrame to perf_metrics_common.CounterView.

    Each counter reports one row per thread/core, so count/cycles average across those rows.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._names = set(df["counter_name"]) if not df.empty else set()

    def count(self, bank: str, counter_name: str) -> float:
        """Average count for a specific counter across all threads (0.0 if absent)."""
        mask = (self._df["bank"] == bank) & (self._df["counter_name"] == counter_name)
        result = self._df.loc[mask, "count"]
        return float(result.mean()) if len(result) > 0 else 0.0

    def cycles(self, bank: str) -> float:
        """Average reference-cycle count for a bank, from any counter in it (0.0 if absent)."""
        result = self._df.loc[self._df["bank"] == bank, "cycles"]
        return float(result.mean()) if len(result) > 0 else 0.0

    def has(self, counter_name: str) -> bool:
        return counter_name in self._names


def _compute_single(df: pd.DataFrame) -> dict:
    """Compute derived metrics for one (zone, run) slice via the shared formula module."""
    if df.empty:
        return {}
    return _mc.compute_metrics(_DfCounterView(df))


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

        # Export both metric families: bounded percentages (*_pct) and unbounded ratios (*_ratio).
        def _exportable(key: str) -> bool:
            return key.endswith("_pct") or key.endswith("_ratio")

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
