#!/usr/bin/env python3
"""
All-Gather Hyperparam Sweep Analyzer

Inputs:
  - A CSV with columns (case-sensitive) including:
    ['Output Shape','Dim','Cluster Axis','Num Devices','Num Links','Topology',
     'Chunks Per Sync','Num Workers Per Link','Data Size in MB','Data Moved in MB',
     'Measured Average (us)','Measured Max (us)','Standard deviation (us)','Bandwidth (GB/s)']

Semantics:
  - New defaults = rows where both 'Chunks Per Sync' and 'Num Workers Per Link' are empty/NaN
  - Old = rows where at least one of them is populated

Outputs:
  1) ring_before_after.csv: Old vs New for ring
  2) linear_before_after.csv: Old vs New for linear
  3) ring_vs_linear_defaults.csv: Ring vs Linear using only the default rows; includes
     separate "Data Moved in MB (Ring)" and "(Linear)"

Usage:
  pip install pandas
  python analyze_all_gather.py /path/to/DiTAllGatherDefaultVsOldFinal.csv --outdir ./out
"""

import argparse
import os
import pandas as pd

JOIN_KEYS = ["Output Shape", "Dim", "Cluster Axis", "Num Devices", "Topology"]

AGG_MAP = {
    "Data Size in MB": "first",
    "Data Moved in MB": "first",
    "Measured Average (us)": "mean",
    "Measured Max (us)": "mean",
    "Bandwidth (GB/s)": "mean",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize whitespace and empties in object columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["", "nan", "None", "NaN", "null", "NULL"]), c] = None

    # Coerce numerics
    numeric_cols = [
        "Dim",
        "Cluster Axis",
        "Num Devices",
        "Num Links",
        "Chunks Per Sync",
        "Num Workers Per Link",
        "Data Size in MB",
        "Data Moved in MB",
        "Measured Average (us)",
        "Measured Max (us)",
        "Standard deviation (us)",
        "Bandwidth (GB/s)",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize topology case
    if "Topology" in df.columns:
        df["Topology"] = df["Topology"].astype(str).str.strip()
        df["Topology"] = df["Topology"].str.lower()

    return df


def split_new_old(df: pd.DataFrame):
    is_new = df["Chunks Per Sync"].isna() & df["Num Workers Per Link"].isna()
    new_df = df[is_new].copy()
    old_df = df[~is_new].copy()
    return new_df, old_df


def summarize_by_keys(df: pd.DataFrame, keys=JOIN_KEYS) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=keys + list(AGG_MAP.keys()))
    out = df.groupby(keys, dropna=False).agg(AGG_MAP).reset_index()
    return out


def _safe_pct_improvement(old_series: pd.Series, new_series: pd.Series) -> pd.Series:
    # Positive => improvement (lower is better)
    denom = old_series.replace(0, pd.NA)
    return ((old_series - new_series) / denom * 100).round(3)


def make_before_after_by_topology(df: pd.DataFrame, topology: str) -> pd.DataFrame:
    new_df, old_df = split_new_old(df)
    new_t = summarize_by_keys(new_df[new_df["Topology"] == topology])
    old_t = summarize_by_keys(old_df[old_df["Topology"] == topology])

    if new_t.empty or old_t.empty:
        return pd.DataFrame(
            columns=[
                "Output Shape",
                "Dim",
                "Cluster Axis",
                "Num Devices",
                "Data Size in MB",
                "Data Moved in MB",
                "Old Measured Max (us)",
                "New Measured Max (us)",
                "Old Measured Average (us)",
                "New Measured Average (us)",
                "Improvement on Average (%)",
                "Improvement on Max (%)",
                "Old Bandwidth (GB/s)",
                "New Bandwidth",
            ]
        )

    merged = pd.merge(
        old_t,
        new_t,
        on=JOIN_KEYS,
        suffixes=(" (Old)", " (New)"),
        how="inner",
    )

    # Compute improvements (lower is better)
    merged["Improvement on Average (%)"] = _safe_pct_improvement(
        merged["Measured Average (us) (Old)"],
        merged["Measured Average (us) (New)"],
    )
    merged["Improvement on Max (%)"] = _safe_pct_improvement(
        merged["Measured Max (us) (Old)"],
        merged["Measured Max (us) (New)"],
    )

    # Round relevant columns
    for col in [
        "Data Size in MB (Old)",
        "Data Moved in MB (Old)",
        "Data Size in MB (New)",
        "Data Moved in MB (New)",
        "Measured Average (us) (Old)",
        "Measured Max (us) (Old)",
        "Measured Average (us) (New)",
        "Measured Max (us) (New)",
        "Bandwidth (GB/s) (Old)",
        "Bandwidth (GB/s) (New)",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].round(3)

    # Arrange & rename columns per spec
    out_cols = [
        "Output Shape",
        "Dim",
        "Cluster Axis",
        "Num Devices",
        "Data Size in MB (Old)",
        "Data Moved in MB (Old)",
        "Measured Max (us) (Old)",
        "Measured Max (us) (New)",
        "Measured Average (us) (Old)",
        "Measured Average (us) (New)",
        "Improvement on Average (%)",
        "Improvement on Max (%)",
        "Bandwidth (GB/s) (Old)",
        "Bandwidth (GB/s) (New)",
    ]
    col_rename = {
        "Data Size in MB (Old)": "Data Size in MB",
        "Data Moved in MB (Old)": "Data Moved in MB",
        "Measured Max (us) (Old)": "Old Measured Max (us)",
        "Measured Max (us) (New)": "New Measured Max (us)",
        "Measured Average (us) (Old)": "Old Measured Average (us)",
        "Measured Average (us) (New)": "New Measured Average (us)",
        "Bandwidth (GB/s) (Old)": "Old Bandwidth (GB/s)",
        "Bandwidth (GB/s) (New)": "New Bandwidth",
    }

    out = merged[out_cols].rename(columns=col_rename)
    out = out.sort_values(["Dim", "Cluster Axis", "Num Devices", "Output Shape"]).reset_index(drop=True)
    return out


def make_ring_vs_linear_defaults(df: pd.DataFrame) -> pd.DataFrame:
    # Only defaults
    new_df, _ = split_new_old(df)
    base_keys = ["Output Shape", "Dim", "Cluster Axis", "Num Devices"]
    agg = {
        "Data Size in MB": "first",
        "Data Moved in MB": "first",
        "Measured Average (us)": "mean",
        "Measured Max (us)": "mean",
        "Bandwidth (GB/s)": "mean",
    }

    ring = (
        new_df[new_df["Topology"] == "ring"]
        .groupby(base_keys, dropna=False)
        .agg(agg)
        .reset_index()
        .add_prefix("Ring: ")
    )
    linear = (
        new_df[new_df["Topology"] == "linear"]
        .groupby(base_keys, dropna=False)
        .agg(agg)
        .reset_index()
        .add_prefix("Linear: ")
    )

    if ring.empty or linear.empty:
        return pd.DataFrame(
            columns=[
                "Output Shape",
                "Dim",
                "Cluster Axis",
                "Num Devices",
                "Data Size in MB",
                "Data Moved in MB (Ring)",
                "Data Moved in MB (Linear)",
                "Linear Measured Max (us)",
                "Ring Measured Max (us)",
                "Linear Measured Average (us)",
                "Ring Measured Average (us)",
                "Improvement on Average (%)",
                "Improvement on Max (%)",
                "Linear Bandwidth (GB/s)",
                "Ring Bandwidth (GB/s)",
            ]
        )

    key_map = {f"Ring: {k}": f"Linear: {k}" for k in base_keys}
    merged = pd.merge(
        linear,
        ring,
        left_on=list(key_map.values()),
        right_on=list(key_map.keys()),
        how="inner",
        suffixes=(" (Linear)", " (Ring)"),
    )

    tidy = pd.DataFrame(
        {
            "Output Shape": merged["Linear: Output Shape"],
            "Dim": merged["Linear: Dim"],
            "Cluster Axis": merged["Linear: Cluster Axis"],
            "Num Devices": merged["Linear: Num Devices"],
            # Data Size likely identical across topologies; show ring's version
            "Data Size in MB": merged["Ring: Data Size in MB"].round(3),
            "Data Moved in MB (Ring)": merged["Ring: Data Moved in MB"].round(3),
            "Data Moved in MB (Linear)": merged["Linear: Data Moved in MB"].round(3),
            "Linear Measured Max (us)": merged["Linear: Measured Max (us)"].round(3),
            "Ring Measured Max (us)": merged["Ring: Measured Max (us)"].round(3),
            "Linear Measured Average (us)": merged["Linear: Measured Average (us)"].round(3),
            "Ring Measured Average (us)": merged["Ring: Measured Average (us)"].round(3),
            "Linear Bandwidth (GB/s)": merged["Linear: Bandwidth (GB/s)"].round(3),
            "Ring Bandwidth (GB/s)": merged["Ring: Bandwidth (GB/s)"].round(3),
        }
    )

    # Positive => ring faster than linear (lower time)
    denom_avg = tidy["Linear Measured Average (us)"].replace(0, pd.NA)
    denom_max = tidy["Linear Measured Max (us)"].replace(0, pd.NA)
    tidy["Improvement on Average (%)"] = (
        (tidy["Linear Measured Average (us)"] - tidy["Ring Measured Average (us)"]) / denom_avg * 100
    ).round(3)
    tidy["Improvement on Max (%)"] = (
        (tidy["Linear Measured Max (us)"] - tidy["Ring Measured Max (us)"]) / denom_max * 100
    ).round(3)

    final_cols = [
        "Output Shape",
        "Dim",
        "Cluster Axis",
        "Num Devices",
        "Data Size in MB",
        "Data Moved in MB (Ring)",
        "Data Moved in MB (Linear)",
        "Linear Measured Max (us)",
        "Ring Measured Max (us)",
        "Linear Measured Average (us)",
        "Ring Measured Average (us)",
        "Improvement on Average (%)",
        "Improvement on Max (%)",
        "Linear Bandwidth (GB/s)",
        "Ring Bandwidth (GB/s)",
    ]
    tidy = tidy[final_cols].sort_values(["Dim", "Cluster Axis", "Num Devices", "Output Shape"]).reset_index(drop=True)
    return tidy


def main():
    ap = argparse.ArgumentParser(description="Analyze All-Gather hyperparam sweep CSVs.")
    ap.add_argument("csv", help="Path to DiTAllGatherDefaultVsOldFinal.csv (or similarly structured file).")
    ap.add_argument("--outdir", default=".", help="Directory to write output CSVs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.csv)

    ring_table = make_before_after_by_topology(df, "ring")
    linear_table = make_before_after_by_topology(df, "linear")
    cross_table = make_ring_vs_linear_defaults(df)

    ring_path = os.path.join(args.outdir, "ring_before_after.csv")
    lin_path = os.path.join(args.outdir, "linear_before_after.csv")
    cross_path = os.path.join(args.outdir, "ring_vs_linear_defaults.csv")

    ring_table.to_csv(ring_path, index=False)
    linear_table.to_csv(lin_path, index=False)
    cross_table.to_csv(cross_path, index=False)

    print(f"Wrote: {ring_path}  ({len(ring_table)} rows)")
    print(f"Wrote: {lin_path}  ({len(linear_table)} rows)")
    print(f"Wrote: {cross_path}  ({len(cross_table)} rows)")


if __name__ == "__main__":
    main()
