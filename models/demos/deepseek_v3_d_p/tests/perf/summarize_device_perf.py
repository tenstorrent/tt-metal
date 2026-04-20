#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Summarize DeepSeek V3 dispatch/combine device perf reports.

Reads the small `device_perf_<model>_<setting>_<YYYY_MM_DD>.csv` files that
`prep_device_perf_report` drops in the tt-metal working directory after each
call to `run_model_device_perf_test_with_merge`, then prints the same
cross-config comparison tables (Ring vs Linear, 2L vs 1L, 14K vs 7K) that
the old hand-rolled perf test logged inline.

Usage:
    python summarize_device_perf.py                              # scan tt-metal/
    python summarize_device_perf.py PATH [PATH ...]              # specific files or dirs
    python summarize_device_perf.py --date 2026_04_20             # restrict by date stamp
    python summarize_device_perf.py --latest                      # keep only the newest file per config

Each CSV's `Model` column is expected to match:
    deepseek_v3_(dispatch|combine)_(linear|ring)_<ndev>_<nlinks>link_<payload>

`payload` is the literal suffix used when constructing the model_name
(e.g. `7k`, `14k`). Unknown filenames are skipped with a warning.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

MODEL_RE = re.compile(
    r"deepseek_v3_(?P<op>dispatch|combine)_"
    r"(?P<topo>linear|ring)_"
    r"(?P<ndev>\d+)_"
    r"(?P<nlinks>\d+)link_"
    r"(?P<payload>\d+k)",
    re.IGNORECASE,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[5]  # tt-metal/


def _parse_model(model: str):
    m = MODEL_RE.search(model)
    if not m:
        return None
    return {
        "op": m.group("op").lower(),
        "topo": m.group("topo").capitalize(),
        "ndev": int(m.group("ndev")),
        "nlinks": int(m.group("nlinks")),
        "payload": m.group("payload").lower(),
    }


def _iter_csv_paths(paths, date_filter):
    """Expand each path: if it's a file, yield it; if a dir, yield every device_perf_*.csv inside."""
    for p in paths:
        p = Path(p)
        if p.is_file():
            yield p
        elif p.is_dir():
            pattern = f"device_perf_*{date_filter}*.csv" if date_filter else "device_perf_*.csv"
            yield from sorted(p.glob(pattern))
        else:
            print(f"[warn] {p}: not found, skipping", file=sys.stderr)


def _collect(csv_paths, keep_latest):
    """Return {(op, topo, ndev, nlinks, payload): (ns, csv_path)}."""
    per_config = {}
    for csv in csv_paths:
        try:
            df = pd.read_csv(csv, skipinitialspace=True)
        except Exception as exc:
            print(f"[warn] {csv}: cannot read ({exc}), skipping", file=sys.stderr)
            continue
        if df.empty or "Model" not in df.columns or "AVG DEVICE KERNEL DURATION [ns]" not in df.columns:
            print(f"[warn] {csv}: missing expected columns, skipping", file=sys.stderr)
            continue
        model = str(df["Model"].iloc[0])
        info = _parse_model(model)
        if info is None:
            print(f"[warn] {csv}: model {model!r} does not match expected pattern, skipping", file=sys.stderr)
            continue
        try:
            ns = float(df["AVG DEVICE KERNEL DURATION [ns]"].iloc[0])
        except (TypeError, ValueError) as exc:
            print(f"[warn] {csv}: cannot parse duration ({exc}), skipping", file=sys.stderr)
            continue
        key = (info["topo"], info["ndev"], info["nlinks"], info["payload"])
        # Index by (topo, ndev, nlinks, payload) then op → ns, keeping newest file when asked.
        bucket = per_config.setdefault(key, {})
        prev = bucket.get(info["op"])
        if prev is None or (keep_latest and csv.stat().st_mtime > prev[1]):
            bucket[info["op"]] = (ns, csv.stat().st_mtime)
    # Collapse to ns only.
    return {key: {op: ns for op, (ns, _) in ops.items()} for key, ops in per_config.items()}


def _fmt_us(ns):
    if ns is None or (isinstance(ns, float) and math.isnan(ns)):
        return "     N/A "
    return f"{ns / 1000:>9.1f}"


def _fmt_speedup(a, b):
    if (
        a is None
        or b is None
        or not a
        or not b
        or (isinstance(a, float) and math.isnan(a))
        or (isinstance(b, float) and math.isnan(b))
        or b == 0
    ):
        return "     N/A "
    return f"{a / b:>9.2f}x"


def _print_per_config(results):
    print(f"\n{'=' * 72}")
    print("  Per-config device kernel duration (μs)")
    print(f"{'=' * 72}")
    print(f"  {'Config':<30} {'Dispatch':>10} {'Combine':>10} {'Total':>10}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")
    for (topo, ndev, nlinks, payload), ops in sorted(results.items()):
        d = ops.get("dispatch")
        c = ops.get("combine")
        total = (d or 0) + (c or 0) if (d is not None or c is not None) else None
        label = f"{topo}-{ndev} {nlinks}L {payload}"
        print(f"  {label:<30} {_fmt_us(d)} {_fmt_us(c)} {_fmt_us(total)}")


def _print_speedups(results):
    header_printed = False

    def _header():
        nonlocal header_printed
        if not header_printed:
            print(f"\n  {'Speedup (base / fast)':<30} {'Dispatch':>10} {'Combine':>10}")
            print(f"  {'-' * 30} {'-' * 10} {'-' * 10}")
            header_printed = True

    payloads = sorted({k[3] for k in results})
    ndevs = sorted({k[1] for k in results})

    for ndev in ndevs:
        for nlinks in (1, 2):
            for ps in payloads:
                lin = results.get(("Linear", ndev, nlinks, ps))
                ring = results.get(("Ring", ndev, nlinks, ps))
                if lin and ring:
                    _header()
                    label = f"Ring/Lin {ndev}dev {nlinks}L {ps}"
                    print(
                        f"  {label:<30} "
                        f"{_fmt_speedup(lin.get('dispatch'), ring.get('dispatch'))} "
                        f"{_fmt_speedup(lin.get('combine'), ring.get('combine'))}"
                    )

    for ndev in ndevs:
        for topo in ("Linear", "Ring"):
            for ps in payloads:
                one = results.get((topo, ndev, 1, ps))
                two = results.get((topo, ndev, 2, ps))
                if one and two:
                    _header()
                    label = f"2L/1L {topo[:3]} {ndev}dev {ps}"
                    print(
                        f"  {label:<30} "
                        f"{_fmt_speedup(one.get('dispatch'), two.get('dispatch'))} "
                        f"{_fmt_speedup(one.get('combine'), two.get('combine'))}"
                    )

    if "14k" in payloads and "7k" in payloads:
        for ndev in ndevs:
            for topo in ("Linear", "Ring"):
                for nlinks in (1, 2):
                    small = results.get((topo, ndev, nlinks, "7k"))
                    large = results.get((topo, ndev, nlinks, "14k"))
                    if small and large:
                        _header()
                        label = f"14K/7K {topo[:3]} {ndev}dev {nlinks}L"
                        print(
                            f"  {label:<30} "
                            f"{_fmt_speedup(small.get('dispatch'), large.get('dispatch'))} "
                            f"{_fmt_speedup(small.get('combine'), large.get('combine'))}"
                        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "paths",
        nargs="*",
        help=f"CSV files or directories to scan. Default: tt-metal root ({DEFAULT_ROOT}).",
    )
    ap.add_argument(
        "--date",
        default="",
        help="Only include files whose name contains this date stamp (e.g. 2026_04_20).",
    )
    ap.add_argument(
        "--latest",
        action="store_true",
        help="When multiple CSVs describe the same config, keep only the newest.",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths] if args.paths else [DEFAULT_ROOT]
    csvs = list(_iter_csv_paths(paths, args.date))
    if not csvs:
        print(f"No device_perf_*.csv files found under: {', '.join(str(p) for p in paths)}", file=sys.stderr)
        sys.exit(1)

    results = _collect(csvs, args.latest)
    if not results:
        print("No matching configs found in the supplied CSVs.", file=sys.stderr)
        sys.exit(1)

    _print_per_config(results)
    _print_speedups(results)


if __name__ == "__main__":
    main()
