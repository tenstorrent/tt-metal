# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Summarise a Tracy `ops_perf_results_*.csv` for the TI2V-5B: device time per step and top ops.

Validates a reported per-step time against the device: the sum of `DEVICE KERNEL DURATION`
over one device's ops, divided by the number of denoise steps in the capture, is the device
kernel time per step. The pipeline is sequence- and tensor-parallel, so every device runs the
same op stream and the per-device totals should agree; the report shows min / mean / max over
devices. `OP TO OP LATENCY` and wall-clock are not trustworthy under Tracy (see
`Wan2_2_TI2V_nadim_opt.md` section 5) and are not used.

Ops are ranked twice: by `OP CODE` alone, and by `OP CODE` plus the shapes and dtypes of the
first inputs, which is what tells the three AGMM call sites (qkv N=2304, to_out N=768, ff1
N=3584) or the two LayerNorm flavours apart without a Python call stack. `--traced-only` keeps
the rows that carry a `METAL TRACE ID`, i.e. the traced steady state without the eager compile
run and the trace capture, and is the right view for a per-step number.

    python models/tt_dit/tests/models/wan2_2/tracy_summarize_ops.py \\
        generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv --steps 20 --traced-only --top 25
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

KERNEL = "DEVICE KERNEL DURATION [ns]"
TRACE_ID = "METAL TRACE ID"
_SHAPE_RE = re.compile(r"^INPUT_(\d+)_([WZYX])_PAD\[LOGICAL\]$")


def _input_signature_columns(df: pd.DataFrame, max_inputs: int) -> list[tuple[int, list[str], str | None]]:
    """(index, [W, Z, Y, X columns], dtype column) for each of the first `max_inputs` inputs present."""
    by_idx: dict[int, dict[str, str]] = {}
    for col in df.columns:
        m = _SHAPE_RE.match(col)
        if m:
            by_idx.setdefault(int(m.group(1)), {})[m.group(2)] = col
    out = []
    for i in sorted(by_idx):
        if i >= max_inputs:
            break
        dims = by_idx[i]
        if set("WZYX") <= set(dims):
            dtype_col = f"INPUT_{i}_DATATYPE" if f"INPUT_{i}_DATATYPE" in df.columns else None
            out.append((i, [dims[d] for d in "WZYX"], dtype_col))
    return out


def _signature(df: pd.DataFrame, max_inputs: int) -> pd.Series:
    cols = _input_signature_columns(df, max_inputs)
    if not cols:
        return pd.Series([""] * len(df), index=df.index)

    def fmt(row):
        parts = []
        for i, dims, dtype_col in cols:
            vals = [row[c] for c in dims]
            if all(pd.isna(v) for v in vals):
                continue
            # values come as "padded[logical]" strings (e.g. "2336[2336]", "1[1]"); keep them verbatim
            shape = "x".join(str(v).strip() if pd.notna(v) else "?" for v in vals)
            dt = str(row[dtype_col]).replace("DataType.", "").lower() if dtype_col and pd.notna(row[dtype_col]) else ""
            parts.append(f"in{i}={shape}{(':' + dt) if dt else ''}")
        return " ".join(parts)

    return df.apply(fmt, axis=1)


def _rank(df: pd.DataFrame, keys: list[str], n_dev: int, steps: int) -> pd.DataFrame:
    by = (
        df.groupby(keys, dropna=False)
        .agg(total_ms=(KERNEL, lambda s: s.sum() / 1e6 / n_dev), count=(KERNEL, lambda s: len(s) / n_dev))
        .sort_values("total_ms", ascending=False)
    )
    by["share_%"] = 100 * by["total_ms"] / by["total_ms"].sum()
    by["mean_us"] = 1e3 * by["total_ms"] / by["count"]
    if steps:
        by["ms_per_step"] = by["total_ms"] / steps
        by["calls_per_step"] = by["count"] / steps
    return by


def summarise(
    csv_path: str,
    *,
    steps: int,
    top: int,
    op_filter: str | None = None,
    traced_only: bool = False,
    max_inputs: int = 3,
    device: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    n_all = len(df)
    df = df[df[KERNEL].notna() & (df[KERNEL] > 0)]
    if traced_only and TRACE_ID in df.columns:
        traced = df[TRACE_ID].notna() & (pd.to_numeric(df[TRACE_ID], errors="coerce").fillna(-1) >= 0)
        print(f"rows: {n_all} total, {len(df)} with device time, {int(traced.sum())} inside a trace replay")
        df = df[traced]
    if device is not None:
        df = df[df["DEVICE ID"] == device]
    if op_filter:
        df = df[df["OP CODE"].astype(str).str.contains(op_filter, case=False, regex=True)]

    per_dev = df.groupby("DEVICE ID")[KERNEL].sum() / 1e6  # ms
    print(f"ops with device time: {len(df)}  devices: {len(per_dev)}  steps in capture: {steps}")
    print(
        f"device kernel time, total per device: min {per_dev.min():.1f}  mean {per_dev.mean():.1f}  "
        f"max {per_dev.max():.1f} ms"
    )
    if steps:
        print(
            f"device kernel time per step:        min {per_dev.min() / steps:.2f}  mean {per_dev.mean() / steps:.2f}  "
            f"max {per_dev.max() / steps:.2f} ms/step"
        )
        per_dev_ops = df.groupby("DEVICE ID")[KERNEL].count()
        print(f"ops per step per device:            {per_dev_ops.mean() / steps:.1f} (mean over devices)")

    # Rank by mean-over-devices total so a chip with an outlier does not dominate.
    n_dev = max(len(per_dev), 1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    fmt = lambda x: f"{x:,.2f}"

    by_op = _rank(df, ["OP CODE"], n_dev, steps)
    print(f"\ntop {top} ops by device kernel time (per device, mean over devices):")
    print(by_op.head(top).to_string(float_format=fmt))

    df = df.assign(SIGNATURE=_signature(df, max_inputs))
    by_sig = _rank(df, ["OP CODE", "SIGNATURE"], n_dev, steps)
    print(f"\ntop {top} ops by device kernel time, split by input shapes/dtypes:")
    print(by_sig.head(top).to_string(float_format=fmt))
    return by_sig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--steps", type=int, default=0, help="denoise steps in the capture (0: skip per-step numbers)")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--filter", default=None, help="regex on OP CODE")
    p.add_argument("--traced-only", action="store_true", help="only rows executed inside a trace replay")
    p.add_argument("--max-inputs", type=int, default=3, help="inputs included in the shape signature")
    p.add_argument("--device", type=int, default=None, help="restrict to one DEVICE ID")
    a = p.parse_args(argv)
    summarise(
        a.csv,
        steps=a.steps,
        top=a.top,
        op_filter=a.filter,
        traced_only=a.traced_only,
        max_inputs=a.max_inputs,
        device=a.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
