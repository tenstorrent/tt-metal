# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metrics from the LLK counter dump: adapts the counters.py DataFrame to tt_llk_perf.metrics (the formulas shared
with the Tracy tool), then aggregates per (zone, run) and exports to CSV.
"""

import pandas as pd
from tt_llk_perf import metrics as _mc

from .perf.schema import (
    MARKER,
    MEAN,
    STD,
    counter_base,
    cycles_of,
    metric_column,
    stat_column,
)


class _DfCounterView:
    """CounterView over the counters.py long-form frame; count/cycles average over the per-thread/core rows.
    Grouped once: a boolean mask per counter made post-processing, not the readout, the cost of a counter run.
    """

    def __init__(self, df: pd.DataFrame):
        self._names = set(df["counter_name"]) if not df.empty else set()
        if df.empty:
            self._count, self._cycles_named, self._cycles_bank = {}, {}, {}
            return
        by_counter = df.groupby(["bank", "counter_name"], sort=False)
        self._count = by_counter["count"].mean().to_dict()
        self._cycles_named = by_counter["cycles"].mean().to_dict()
        self._cycles_bank = df.groupby("bank", sort=False)["cycles"].mean().to_dict()

    def count(self, bank: str, counter_name: str) -> float:
        return float(self._count.get((bank, counter_name), 0.0))

    def cycles(self, bank: str, counter_name: "str | None" = None) -> float:
        if counter_name is not None and (bank, counter_name) in self._cycles_named:
            return float(self._cycles_named[(bank, counter_name)])
        return float(self._cycles_bank.get(bank, 0.0))

    def has(self, counter_name: str) -> bool:
        return counter_name in self._names

    def is_blackhole(self) -> bool:
        from .chip_architecture import ChipArchitecture, get_chip_architecture

        return get_chip_architecture() == ChipArchitecture.BLACKHOLE


def _compute_single(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    view = _DfCounterView(df)
    metrics = _mc.compute_metrics(view)
    # The l1_client counter is named after the run selection, so its metric cannot live in the static table.
    l1_client = df.loc[df["bank"] == "L1_CLIENT", "counter_name"]
    if not l1_client.empty:
        metrics.update(_mc.compute_l1_client_metrics(view, l1_client))
    return metrics


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
                    e.g., ["INIT", "TILE_LOOP"] maps ZONE_0 -> INIT, ZONE_1 -> TILE_LOOP.

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

        def _exportable(key: str) -> bool:
            return key.endswith("_pct") or key.endswith("_ratio")

        if len(zone_metrics) >= 2:
            metrics_df = pd.DataFrame(zone_metrics)
            for col in metrics_df.columns:
                if not _exportable(col):
                    continue
                # Always emit the column (NaN if under 2 runs had a value). A metric whose denominator never counts is
                # None on every run; dropping its column made the column set vary by variant and fail the schema check.
                values = metrics_df[col].dropna()
                enough = len(values) >= 2
                row[metric_column(run_type_name, stat_column(col, MEAN))] = (
                    float(values.mean()) if enough else float("nan")
                )
                row[metric_column(run_type_name, stat_column(col, STD))] = (
                    float(values.std()) if enough else float("nan")
                )
        else:
            for k, v in zone_metrics[0].items():
                if not _exportable(k):
                    continue
                row[metric_column(run_type_name, k)] = v

        rows.append(row)

    return pd.DataFrame(rows)


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

        # One grouping per zone (discovery order kept), not one boolean mask per counter.
        keys = ["bank", "counter_name"]
        if has_runs:
            per_run = zone_df.groupby(keys + ["run_index"], sort=False)[
                ["count", "cycles"]
            ].mean()
            if not per_run.index.droplevel(2).duplicated().any():
                # One run per counter (the default run_count): its per-run mean is the value.
                for (bank, counter_name, _), count, cycles in zip(
                    per_run.index, per_run["count"], per_run["cycles"]
                ):
                    col_name = counter_base(bank, counter_name)
                    row[metric_column(run_type_name, col_name)] = float(count)
                    row[metric_column(run_type_name, cycles_of(col_name))] = float(
                        cycles
                    )
                rows.append(row)
                continue
            groups = per_run.groupby(level=[0, 1], sort=False)
        else:
            groups = zone_df.groupby(keys, sort=False)[["count", "cycles"]]

        for (bank, counter_name), g in groups:
            col_name = counter_base(bank, counter_name)
            for field, base in (("count", col_name), ("cycles", cycles_of(col_name))):
                values = g[field]
                if not has_runs:
                    row[metric_column(run_type_name, base)] = float(values.mean())
                elif len(values) >= 2:
                    row[metric_column(run_type_name, stat_column(base, MEAN))] = float(
                        values.mean()
                    )
                    row[metric_column(run_type_name, stat_column(base, STD))] = float(
                        values.std()
                    )
                else:
                    row[metric_column(run_type_name, base)] = float(values.iloc[0])

        rows.append(row)

    return pd.DataFrame(rows)
