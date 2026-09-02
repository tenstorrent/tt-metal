# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance metrics from the LLK test counter dump.

The metric formulas live in the shared module tools/tracy/perf_metrics_common.py (single source,
also used by the Tracy tool). This file adapts the counters.py DataFrame to that module's
CounterView, computes per (zone, run) metrics, and aggregates/exports them to CSV.
"""

import pandas as pd

from .perf.schema import (
    MARKER,
    MEAN,
)
from .perf.schema import PERF_METRICS_COMMON as _mc
from .perf.schema import (
    STD,
    counter_base,
    cycles_of,
    metric_column,
    stat_column,
)


class _DfCounterView:
    """Adapts the counters.py long-form DataFrame to perf_metrics_common.CounterView.

    Each counter reports one row per thread/core, so count/cycles average across those rows.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._names = set(df["counter_name"]) if not df.empty else set()

    def count(self, bank: str, counter_name: str) -> float:
        mask = (self._df["bank"] == bank) & (self._df["counter_name"] == counter_name)
        result = self._df.loc[mask, "count"]
        return float(result.mean()) if len(result) > 0 else 0.0

    def cycles(self, bank: str) -> float:
        result = self._df.loc[self._df["bank"] == bank, "cycles"]
        return float(result.mean()) if len(result) > 0 else 0.0

    def has(self, counter_name: str) -> bool:
        return counter_name in self._names

    def is_blackhole(self) -> bool:
        from ..chip_architecture import ChipArchitecture, get_chip_architecture

        return get_chip_architecture() == ChipArchitecture.BLACKHOLE


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
        row = {MARKER: marker_name}

        # Export both metric families: bounded percentages and unbounded ratios.
        def _exportable(key: str) -> bool:
            return key.endswith("_pct") or key.endswith("_ratio")

        if len(zone_metrics) >= 2:
            metrics_df = pd.DataFrame(zone_metrics)
            for col in metrics_df.columns:
                if not _exportable(col):
                    continue
                values = metrics_df[col].dropna()
                if len(values) >= 2:
                    row[metric_column(run_type_name, stat_column(col, MEAN))] = float(
                        values.mean()
                    )
                    row[metric_column(run_type_name, stat_column(col, STD))] = float(
                        values.std()
                    )
        else:
            for k, v in zone_metrics[0].items():
                if not _exportable(k):
                    continue
                row[metric_column(run_type_name, k)] = v

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
        row = {MARKER: marker_name}

        # Get unique counters in this zone (preserving discovery order)
        counter_keys = (
            zone_df[["bank", "counter_name"]].drop_duplicates().values.tolist()
        )

        for bank, counter_name in counter_keys:
            mask = (zone_df["bank"] == bank) & (zone_df["counter_name"] == counter_name)
            col_name = counter_base(bank, counter_name)

            if has_runs:
                per_run = zone_df.loc[mask].groupby("run_index")["count"].mean()
                if len(per_run) >= 2:
                    row[metric_column(run_type_name, stat_column(col_name, MEAN))] = (
                        float(per_run.mean())
                    )
                    row[metric_column(run_type_name, stat_column(col_name, STD))] = (
                        float(per_run.std())
                    )
                elif len(per_run) == 1:
                    row[metric_column(run_type_name, col_name)] = float(per_run.iloc[0])
            else:
                values = zone_df.loc[mask, "count"]
                row[metric_column(run_type_name, col_name)] = float(values.mean())

            # Also export cycles for this counter
            col_cycles = cycles_of(col_name)
            if has_runs:
                per_run_cyc = zone_df.loc[mask].groupby("run_index")["cycles"].mean()
                if len(per_run_cyc) >= 2:
                    row[metric_column(run_type_name, stat_column(col_cycles, MEAN))] = (
                        float(per_run_cyc.mean())
                    )
                    row[metric_column(run_type_name, stat_column(col_cycles, STD))] = (
                        float(per_run_cyc.std())
                    )
                elif len(per_run_cyc) == 1:
                    row[metric_column(run_type_name, col_cycles)] = float(
                        per_run_cyc.iloc[0]
                    )
            else:
                cyc_values = zone_df.loc[mask, "cycles"]
                row[metric_column(run_type_name, col_cycles)] = float(cyc_values.mean())

        rows.append(row)

    return pd.DataFrame(rows)
