#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Build grouped bar charts comparing rows from two Tracy ``ops_perf_results*.csv`` reports.

**Matching:** Rows are paired by **normalized ``ATTRIBUTES``** (not ``GLOBAL CALL COUNT``), so the same
logical op lines up across runs even when dispatch order or GCC differs.

Normalization (``--normalize-attributes``) folds differences that commonly vary between builds—such as
``is_sfpu``, so SFPU vs non-SFPU paths can still be compared when you intend to.

Identical ``ATTRIBUTES`` strings can repeat **four times** per capture (two tensor sizes × warmup/measured,
e.g. interleaved L1 where ``memory_config`` does not encode H/W). Those groups are split using
``GLOBAL CALL COUNT`` order: the first two GCCs are treated as the **32×32** pair, the next two as
**1024×1024**. **``--pair-keep``** then picks warmup vs measured within each pair.

Two PNG/CSV pairs are written: **32×32** and **1024×1024**.

Default metric is ``DEVICE KERNEL DURATION [ns]``.

Examples:

.. code-block:: bash

   python tools/tracy/compare_ops_perf_bar_chart.py \\
     generated/profiler/reports/run_a \\
     generated/profiler/reports/run_b \\
     -o eq_gt_compare.png

   python tools/tracy/compare_ops_perf_bar_chart.py \\
     report_a/ops_perf.csv report_b/ops_perf.csv \\
     --metric "DEVICE FW DURATION [ns]" --pair-keep second
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_ops_csv(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")
    matches = sorted(path.glob("ops_perf_results*.csv"))
    if not matches:
        raise FileNotFoundError(f"No ops_perf_results*.csv under {path}")
    return matches[-1]


def _load_device_ops(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "OP TYPE" not in df.columns:
        raise ValueError(f"Unexpected CSV (no OP TYPE column): {csv_path}")
    return df[df["OP TYPE"].astype(str) == "tt_dnn_device"].copy()


def normalize_attributes_for_match(attr: str, *, normalize_sfpu: bool) -> str:
    """Stable key for joining rows across runs."""
    s = str(attr).strip()
    if normalize_sfpu:
        s = re.sub(r"'is_sfpu': 'true'", "'is_sfpu': '*'", s)
        s = re.sub(r"'is_sfpu': 'false'", "'is_sfpu': '*'", s)
    # CSV escaping differences for embedded JSON in shard grid
    s = s.replace('""', '"')
    return s


def _binary_op_from_attributes(s: str) -> str:
    m = re.search(r"BinaryOpType::(\w+)", s)
    return m.group(1).lower() if m else "?"


def _layout_from_attributes(s: str) -> str:
    if "TensorMemoryLayout::INTERLEAVED" in s:
        return "L1_intl"
    if "TensorMemoryLayout::HEIGHT_SHARDED" in s:
        return "height"
    if "TensorMemoryLayout::BLOCK_SHARDED" in s:
        return "block"
    return "?"


def _logical_extent_int(cell) -> int | None:
    if pd.isna(cell):
        return None
    t = str(cell).strip()
    m = re.match(r"(\d+)", t)
    return int(m.group(1)) if m else None


def _infer_shape_bucket_from_row(row: pd.Series) -> Literal["32x32", "1024x1024", "other"]:
    attr = str(row.get("ATTRIBUTES", ""))
    y = _logical_extent_int(row.get("INPUT_0_Y_PAD[LOGICAL]"))
    x = _logical_extent_int(row.get("INPUT_0_X_PAD[LOGICAL]"))
    if y == 1024 and x == 1024:
        return "1024x1024"
    if y == 32 and x == 32:
        if "shape=[128; 1024]" in attr or "shape=[256; 256]" in attr:
            return "1024x1024"
        return "32x32"
    if "128" in attr and "1024" in attr:
        return "1024x1024"
    if "shape=[32; 32]" in attr or ("32" in attr and "HEIGHT_SHARDED" in attr and "1024" not in attr):
        return "32x32"
    return "other"


def _label_full_from_attr(attr: str, row: pd.Series) -> str:
    op = _binary_op_from_attributes(attr)
    lay = _layout_from_attributes(attr)
    yv = _logical_extent_int(row.get("INPUT_0_Y_PAD[LOGICAL]"))
    xv = _logical_extent_int(row.get("INPUT_0_X_PAD[LOGICAL]"))
    sh = f"{yv}×{xv}" if yv is not None and xv is not None else "?"
    return f"{op} {lay} {sh}"


def _label_compact_from_attr(attr: str) -> str:
    op = _binary_op_from_attributes(attr)
    lay = _layout_from_attributes(attr)
    return f"{op} {lay}"


def _dedupe_one_report(
    df: pd.DataFrame,
    *,
    pair_keep: Literal["first", "second"],
    normalize_sfpu: bool,
) -> pd.DataFrame:
    """One row per (match_key, tensor-size phase): collapse warmup/measured and 32/1024 interleaved."""
    df = df.copy()
    df["_match_key"] = df["ATTRIBUTES"].map(lambda a: normalize_attributes_for_match(a, normalize_sfpu=normalize_sfpu))
    pick = 1 if pair_keep == "second" else 0
    out_frames: list[pd.DataFrame] = []

    for _key, g in df.groupby("_match_key", sort=False):
        g = g.sort_values("GLOBAL CALL COUNT").reset_index(drop=True)
        n = len(g)

        if n == 2:
            sel = g.iloc[pick : pick + 1].copy()
            sel["_shape_bucket"] = sel.apply(_infer_shape_bucket_from_row, axis=1)
            out_frames.append(sel)
        elif n == 4:
            # Same ATTRIBUTES for 32×32 and 1024×1024 (e.g. interleaved); GCC order: w32, m32, w1024, m1024
            sel_32 = g.iloc[pick : pick + 1].copy()
            sel_32["_shape_bucket"] = "32x32"
            sel_1024 = g.iloc[2 + pick : 3 + pick].copy()
            sel_1024["_shape_bucket"] = "1024x1024"
            out_frames.append(sel_32)
            out_frames.append(sel_1024)
        elif n == 1:
            sel = g.copy()
            sel["_shape_bucket"] = sel.apply(_infer_shape_bucket_from_row, axis=1)
            out_frames.append(sel)
        else:
            raise ValueError(
                f"Unexpected rows per normalized ATTRIBUTES key ({n}); expected 1, 2, or 4. "
                f"Key prefix: {str(_key)[:120]}..."
            )

    return pd.concat(out_frames, ignore_index=True)


def _merge_by_attributes(da: pd.DataFrame, db: pd.DataFrame) -> pd.DataFrame:
    """Join on (_match_key, _shape_bucket) from :func:`_dedupe_one_report`."""
    merged = da.merge(
        db,
        on=["_match_key", "_shape_bucket"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        raise RuntimeError(
            "No overlapping (_match_key, _shape_bucket) between reports. "
            "Try without --no-normalize-attributes or verify both runs exercise the same ops."
        )
    return merged


def _plot_one_chart(
    merged: pd.DataFrame,
    *,
    metric: str,
    metric_a: str,
    metric_b: str,
    label_a: str,
    label_b: str,
    title: str,
    xtick_labels: list[str],
    figsize: tuple[float, float],
) -> plt.Figure:
    va = pd.to_numeric(merged[metric_a], errors="coerce")
    vb = pd.to_numeric(merged[metric_b], errors="coerce")
    n = len(merged)
    x = np.arange(n)
    width = 0.38
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, va, width, label=label_a)
    ax.bar(x + width / 2, vb, width, label=label_b)
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def compare_reports(
    path_a: Path,
    path_b: Path,
    *,
    metric: str,
    label_a: str,
    label_b: str,
    pair_keep: Literal["first", "second"],
    normalize_sfpu: bool,
    match_on: Literal["attributes", "gcc"],
    figsize: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, plt.Figure]]]:
    csv_a = _resolve_ops_csv(path_a)
    csv_b = _resolve_ops_csv(path_b)
    da = _load_device_ops(csv_a)
    db = _load_device_ops(csv_b)

    if metric not in da.columns or metric not in db.columns:
        raise KeyError(
            f"Metric {metric!r} not in both CSVs. Example columns: "
            + ", ".join(c for c in da.columns if "DURATION" in c or "LATENCY" in c)
        )

    if match_on == "gcc":
        merged = da.merge(db, on="GLOBAL CALL COUNT", how="inner", suffixes=("_a", "_b"))
        if merged.empty:
            raise RuntimeError("No overlapping GLOBAL CALL COUNT between reports.")
        merged = merged.sort_values("GLOBAL CALL COUNT").reset_index(drop=True)
        merged = (
            merged.iloc[0::2].reset_index(drop=True)
            if pair_keep == "first"
            else merged.iloc[1::2].reset_index(drop=True)
        )

        # Ambiguous interleaved 32 vs 1024 when only GCC merge: reuse occurrence heuristic
        seen: set[str] = set()
        buckets = []
        for _, r in merged.iterrows():
            attr = str(r.get("ATTRIBUTES_a", ""))
            ck = _label_compact_from_attr(attr)
            y = _logical_extent_int(r.get("INPUT_0_Y_PAD[LOGICAL]_a"))
            x = _logical_extent_int(r.get("INPUT_0_X_PAD[LOGICAL]_a"))
            if y == 32 and x == 32 and "INTERLEAVED" in attr:
                if ck not in seen:
                    seen.add(ck)
                    buckets.append("32x32")
                else:
                    buckets.append("1024x1024")
            else:
                buckets.append(
                    _infer_shape_bucket_from_row(
                        pd.Series(
                            {
                                "ATTRIBUTES": attr,
                                "INPUT_0_Y_PAD[LOGICAL]": r.get("INPUT_0_Y_PAD[LOGICAL]_a"),
                                "INPUT_0_X_PAD[LOGICAL]": r.get("INPUT_0_X_PAD[LOGICAL]_a"),
                            }
                        )
                    )
                )
        merged["_shape_bucket"] = buckets

        merged["_label"] = merged.apply(
            lambda r: _label_full_from_attr(
                str(r.get("ATTRIBUTES_a", "")),
                pd.Series(
                    {
                        "INPUT_0_Y_PAD[LOGICAL]": r.get("INPUT_0_Y_PAD[LOGICAL]_a"),
                        "INPUT_0_X_PAD[LOGICAL]": r.get("INPUT_0_X_PAD[LOGICAL]_a"),
                    }
                ),
            ),
            axis=1,
        )
    else:
        da_d = _dedupe_one_report(da, pair_keep=pair_keep, normalize_sfpu=normalize_sfpu)
        db_d = _dedupe_one_report(db, pair_keep=pair_keep, normalize_sfpu=normalize_sfpu)
        merged = _merge_by_attributes(da_d, db_d)

        merged["_label"] = merged.apply(
            lambda r: _label_full_from_attr(
                str(r.get("ATTRIBUTES_a", "")),
                pd.Series(
                    {
                        "INPUT_0_Y_PAD[LOGICAL]": r.get("INPUT_0_Y_PAD[LOGICAL]_a"),
                        "INPUT_0_X_PAD[LOGICAL]": r.get("INPUT_0_X_PAD[LOGICAL]_a"),
                    }
                ),
            ),
            axis=1,
        )

    merged = merged.sort_values(by=["_shape_bucket", "_label"], kind="stable").reset_index(drop=True)

    metric_a = f"{metric}_a"
    metric_b = f"{metric}_b"
    match_desc = (
        f"GLOBAL CALL COUNT + pair_keep={pair_keep}" if match_on == "gcc" else "normalized ATTRIBUTES + _shape_bucket"
    )

    charts: dict[str, tuple[pd.DataFrame, plt.Figure]] = {}
    for bucket, pretty in (("32x32", "32×32"), ("1024x1024", "1024×1024")):
        sub = merged[merged["_shape_bucket"] == bucket].copy()
        if sub.empty:
            charts[bucket] = (sub, None)  # type: ignore[assignment]
            continue
        xlabs = sub.apply(lambda r: _label_compact_from_attr(str(r["ATTRIBUTES_a"])), axis=1).tolist()
        title = f"Profiler ops — {pretty} (matched by {match_desc})"
        fig = _plot_one_chart(
            sub,
            metric=metric,
            metric_a=metric_a,
            metric_b=metric_b,
            label_a=label_a,
            label_b=label_b,
            title=title,
            xtick_labels=xlabs,
            figsize=figsize,
        )
        sum_cols = [
            "GLOBAL CALL COUNT_a",
            "GLOBAL CALL COUNT_b",
            "_match_key",
            "_label",
            "_shape_bucket",
            metric_a,
            metric_b,
        ]
        sum_cols = [c for c in sum_cols if c in sub.columns]
        summary = sub[sum_cols].copy()
        summary.rename(
            columns={
                "GLOBAL CALL COUNT_a": "GLOBAL CALL COUNT (A)",
                "GLOBAL CALL COUNT_b": "GLOBAL CALL COUNT (B)",
                "_match_key": "match_key (normalized ATTRIBUTES)",
            },
            inplace=True,
        )
        va = pd.to_numeric(sub[metric_a], errors="coerce")
        vb = pd.to_numeric(sub[metric_b], errors="coerce")
        summary["ratio_b_over_a"] = np.where(va != 0, vb / va, np.nan)
        charts[bucket] = (summary, fig)

    full_summary = merged.copy()
    va = pd.to_numeric(merged[metric_a], errors="coerce")
    vb = pd.to_numeric(merged[metric_b], errors="coerce")
    full_summary["ratio_b_over_a"] = np.where(va != 0, vb / va, np.nan)

    skipped = merged[merged["_shape_bucket"] == "other"]
    if not skipped.empty:
        print(f"Note: {len(skipped)} row(s) with _shape_bucket == 'other' (see full summary CSV).")

    return full_summary, charts  # type: ignore[return-value]


def _out_paths(base: Path) -> tuple[Path, Path, Path, Path, Path]:
    base = base.expanduser()
    stem = base.stem
    parent = base.parent
    suf = base.suffix if base.suffix else ".png"
    return (
        parent / f"{stem}_32x32{suf}",
        parent / f"{stem}_1024x1024{suf}",
        parent / f"{stem}_32x32.csv",
        parent / f"{stem}_1024x1024.csv",
        parent / f"{stem}_all.csv",
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report_a", type=Path, help="Directory containing ops_perf_results*.csv, or path to CSV")
    p.add_argument("report_b", type=Path, help="Second report directory or CSV")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Base output path; writes <stem>_32x32.png, <stem>_1024x1024.png (and CSVs). "
        "Default: compare_ops_perf_<a>_vs_<b>.png in cwd.",
    )
    p.add_argument(
        "--metric",
        default="DEVICE KERNEL DURATION [ns]",
        help='Numeric column to compare (default: "DEVICE KERNEL DURATION [ns]"). '
        'Try "DEVICE FW DURATION [ns]" or "HOST DURATION [ns]" as needed.',
    )
    p.add_argument(
        "--pair-keep",
        choices=("first", "second"),
        default="first",
        help="Within each (warmup, measured) pair, keep first (warmup) or second (measured). "
        "Also applies inside each 32/1024 quadruple for identical ATTRIBUTES.",
    )
    p.add_argument(
        "--match-on",
        choices=("attributes", "gcc"),
        default="attributes",
        help="Join key between reports: normalized ATTRIBUTES + shape bucket (default), or GLOBAL CALL COUNT.",
    )
    p.add_argument(
        "--no-normalize-attributes",
        action="store_true",
        help="Disable is_sfpu normalization (stricter ATTRIBUTES matching).",
    )
    p.add_argument("--label-a", default="run A", help="Legend label for first report")
    p.add_argument("--label-b", default="run B", help="Legend label for second report")
    p.add_argument("--show", action="store_true", help="Show interactive matplotlib windows")
    args = p.parse_args(argv)

    out = args.output
    if out is None:
        sa = _resolve_ops_csv(args.report_a).stem
        sb = _resolve_ops_csv(args.report_b).stem
        out = Path(f"compare_ops_perf_{sa}_vs_{sb}.png")

    full_summary, charts = compare_reports(
        args.report_a,
        args.report_b,
        metric=args.metric,
        label_a=args.label_a,
        label_b=args.label_b,
        pair_keep=args.pair_keep,
        normalize_sfpu=not args.no_normalize_attributes,
        match_on=args.match_on,
        figsize=(12, 5),
    )

    # Drop internal merge helpers from full CSV for readability when present
    drop_cols = [c for c in ("_attr_key_a", "_attr_key_b") if c in full_summary.columns]
    for c in drop_cols:
        del full_summary[c]

    p32, p1024, c32, c1024, c_all = _out_paths(out)
    full_summary.to_csv(c_all, index=False)
    print(f"Wrote combined table: {c_all.resolve()}")

    for bucket, path_png, path_csv in (
        ("32x32", p32, c32),
        ("1024x1024", p1024, c1024),
    ):
        summary, fig = charts[bucket]
        if summary.empty:
            print(f"Skipping empty chart for {bucket} (no rows).")
            continue
        if fig is None:
            continue
        fig.savefig(path_png, dpi=150)
        print(f"Wrote chart: {path_png.resolve()}")
        summary.to_csv(path_csv, index=False)
        print(f"Wrote table: {path_csv.resolve()}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
