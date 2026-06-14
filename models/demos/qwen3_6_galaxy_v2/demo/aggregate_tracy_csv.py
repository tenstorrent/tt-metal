#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2 tracy CSV aggregator (with fallback path).

Two modes:

1. Normal mode (preferred): reads the post-processed
   ``ops_perf_results_<timestamp>.csv`` from
   ``generated/profiler/reports/<timestamp>/``.  Signpost rows are inline
   (``OP TYPE == 'signpost'``) so the prefill vs decode split is direct.

2. Fallback mode: when ``process_ops_logs.py`` fails (typical cause: the
   tracy DRAM ring-buffer-full warning dropped device markers; the
   processor asserts on the missing ops), reconstruct the breakdown
   straight from the raw tracy artifacts under
   ``generated/profiler/.logs/``:

   - ``cpp_device_perf_report.csv``  — per (op_id, device_id) device kernel duration
   - ``tracy_ops_times.csv``         — per op_id host start timestamp (``ns_since_start``)
   - ``tracy_ops_data.csv``          — signpost lines with ``total_ns`` timestamps

   We build an op_id → host_ts map from tracy_ops_times (TT_DNN_DEVICE_OP
   rows), look up signpost timestamps from tracy_ops_data
   (TT_SIGNPOST: start / prefill_done / stop), then walk
   cpp_device_perf_report and bucket each row into prefill / decode based
   on the host_ts of its op_id.

Usage:

    python models/demos/qwen3_6_galaxy_v2/demo/aggregate_tracy_csv.py [csv_path]

with no argument it auto-detects whichever artifacts are available.
"""
from __future__ import annotations

import csv
import os
import pathlib
import re
import sys
from collections import defaultdict

# --- Schema ---


# Profiler artifacts land under $TT_METAL_HOME/generated/profiler (cwd when run from
# the repo root). Fall back to the cwd so this works across worktrees.
_TT_METAL = pathlib.Path(os.environ.get("TT_METAL_HOME", os.getcwd()))
_PROFILER_DIR = _TT_METAL / "generated/profiler"
_LOGS_DIR = _PROFILER_DIR / ".logs"
_REPORTS_DIR = _PROFILER_DIR / "reports"


_MATMUL_OPS = {"MatmulDeviceOperation", "MinimalMatmulDeviceOperation"}
_CCL_OPS = {
    "AllGatherDeviceOperation",
    "AllGatherAsyncDeviceOperation",
    "ReduceScatterDeviceOperation",
    "ReduceScatterAsyncDeviceOperation",
    "AllReduceDeviceOperation",
}


# --- Section / signpost handling ---
#
# Three-stage VL split (mm_perf_qwen36.py):
#     start -> [vision] -> vision_done -> [prefill] -> prefill_done -> [decode] -> stop
# Two-stage text split (tracy_perf_*.py) is still supported (no vision_done):
#     start -> [prefill] -> prefill_done -> [decode] -> stop


def _section_after(signpost_name: str, _prev):
    """Map a signpost name to the section that FOLLOWS it.

    After `start` the section is the provisional `s1`: ops there belong to VISION
    if a `vision_done` signpost follows, or to PREFILL otherwise (2-stage text
    runs). The `_bucket` helper resolves `s1` at `vision_done`/`prefill_done` time.
    """
    if signpost_name == "start":
        return "s1"
    if signpost_name == "vision_done":
        return "prefill"
    if signpost_name == "prefill_done":
        return "decode"
    if signpost_name == "stop":
        return None
    return _prev


# Ops seen in the provisional `s1` window are held here until we learn (from the
# next signpost) whether s1 was VISION (vision_done follows) or PREFILL (prefill_done).
_S1_PENDING: list = []


def _bucket(section, op_code, ns, vis_pairs, pf_pairs, dec_pairs):
    if section == "s1":
        _S1_PENDING.append((op_code, ns))
    elif section == "prefill":
        pf_pairs.append((op_code, ns))
    elif section == "decode":
        dec_pairs.append((op_code, ns))


def _resolve_s1(signpost_name: str, vis_pairs, pf_pairs):
    """Flush the provisional s1 buffer once we know what it was."""
    if not _S1_PENDING:
        return
    if signpost_name == "vision_done":
        vis_pairs.extend(_S1_PENDING)
    elif signpost_name == "prefill_done":
        pf_pairs.extend(_S1_PENDING)  # 2-stage run: s1 was prefill
    _S1_PENDING.clear()


# --- Aggregation helpers ---


def _aggregate_pairs(pairs):
    """``pairs`` is iterable of ``(op_code, dev_kernel_ns)``.  Returns
    ``(entries, total_us)`` sorted by sum_us desc."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    for op_code, ns in pairs:
        sums[op_code] += ns
        counts[op_code] += 1
    total_us = sum(sums.values()) / 1000.0
    entries = []
    for op_code, sum_ns in sums.items():
        sum_us = sum_ns / 1000.0
        cnt = counts[op_code]
        avg_us = sum_us / cnt if cnt else 0.0
        pct = (sum_us / total_us) * 100.0 if total_us > 0 else 0.0
        entries.append((op_code, cnt, sum_us, avg_us, pct))
    entries.sort(key=lambda e: -e[2])
    return entries, total_us


def _summarize_categories(entries):
    total_us = sum(e[2] for e in entries)
    mm_us = sum(e[2] for e in entries if e[0] in _MATMUL_OPS)
    ccl_us = sum(e[2] for e in entries if e[0] in _CCL_OPS)
    other_us = total_us - mm_us - ccl_us
    return {
        "total_us": total_us,
        "matmul_us": mm_us,
        "ccl_us": ccl_us,
        "other_us": other_us,
        "matmul_pct": (mm_us / total_us * 100.0) if total_us else 0,
        "ccl_pct": (ccl_us / total_us * 100.0) if total_us else 0,
        "other_pct": (other_us / total_us * 100.0) if total_us else 0,
    }


def _print_table(label: str, entries, total_us: float) -> None:
    n_rows = sum(c for _, c, _, _, _ in entries)
    print(f"\n### {label}  ({n_rows} op rows, total_dev_us = {total_us:,.1f})\n")
    print("| op | count | sum_dev_us | avg_us | % of dev time |")
    print("|---|---:|---:|---:|---:|")
    for op_code, cnt, sum_us, avg_us, pct in entries:
        print(f"| {op_code:<40s} | {cnt:>5} | {sum_us:>11,.1f} | {avg_us:>7.2f} | {pct:>5.1f} % |")


# --- Mode 1: post-processed CSV ---


def _try_processed_csv():
    candidates = sorted(_REPORTS_DIR.glob("*/ops_perf_results_*.csv"))
    if not candidates:
        return None
    csv_path = candidates[-1]
    vis_pairs, pf_pairs, dec_pairs, signposts, n_rows = [], [], [], [], 0
    section = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            op_type = (row.get("OP TYPE") or "").strip().lower()
            op_code = (row.get("OP CODE") or "").strip()
            if op_type == "signpost":
                signposts.append(op_code)
                _resolve_s1(op_code, vis_pairs, pf_pairs)
                section = _section_after(op_code, section)
                continue
            if section is None:
                continue
            dur = (row.get("DEVICE KERNEL DURATION [ns]") or "").strip()
            try:
                ns = float(dur)
            except ValueError:
                continue
            _bucket(section, op_code, ns, vis_pairs, pf_pairs, dec_pairs)
    return {
        "csv_path": csv_path,
        "vis_pairs": vis_pairs,
        "pf_pairs": pf_pairs,
        "dec_pairs": dec_pairs,
        "signposts": signposts,
        "n_rows": n_rows,
    }


# --- Mode 2: fallback from raw logs ---


_SIGNPOST_RE = re.compile(r"TT_SIGNPOST:\s*(\S+)")
# `TT_DNN_DEVICE_OP: "OpName", hash, deviceID, cacheHit, opID`
_TT_DNN_RE = re.compile(r'TT_DNN_DEVICE_OP:\s*"([^"]+)",\s*\S+,\s*\S+,\s*\S+,\s*(\d+)')


def _parse_signposts_and_op_meta(tracy_data_csv: pathlib.Path):
    """Single-pass parse of ``tracy_ops_data.csv``.

    Returns ``(signposts_by_name, op_meta_by_id)``:
      * ``signposts_by_name``: ``{label: total_ns}`` for ``start``,
        ``prefill_done``, ``stop``.
      * ``op_meta_by_id``: ``{op_id: (op_name, host_total_ns)}`` for every
        TT_DNN_DEVICE_OP row.  ``host_total_ns`` is the row's ``total_ns``.
    """
    signposts: dict[str, int] = {}
    op_meta: dict[int, tuple[str, int]] = {}
    with open(tracy_data_csv, newline="") as f:
        reader = csv.reader(f, delimiter=";", quotechar="`")
        for row in reader:
            if len(row) < 2:
                continue
            msg = row[0]
            try:
                ts = int(row[1])
            except ValueError:
                continue
            if "TT_SIGNPOST" in msg:
                m = _SIGNPOST_RE.search(msg)
                if m:
                    signposts[m.group(1)] = ts
                continue
            if "TT_DNN_DEVICE_OP" in msg:
                m = _TT_DNN_RE.search(msg)
                if not m:
                    continue
                op_name = m.group(1)
                op_id = int(m.group(2))
                # Same op_id may appear multiple times (per device); keep
                # the minimum ts so the bucket boundary is well-defined.
                prev = op_meta.get(op_id)
                if prev is None or ts < prev[1]:
                    op_meta[op_id] = (op_name, ts)
    return signposts, op_meta


def _try_fallback_from_logs():
    cpp_csv = _LOGS_DIR / "cpp_device_perf_report.csv"
    data_csv = _LOGS_DIR / "tracy_ops_data.csv"
    for f in (cpp_csv, data_csv):
        if not f.exists():
            return None

    print(f"[fallback] parsing signposts + op meta from {data_csv} ...", file=sys.stderr)
    signposts_ts, op_meta = _parse_signposts_and_op_meta(data_csv)
    if "start" not in signposts_ts or "prefill_done" not in signposts_ts or "stop" not in signposts_ts:
        print(f"[fallback] missing signposts: {list(signposts_ts.keys())}", file=sys.stderr)
        return None
    print(f"[fallback] signposts {signposts_ts}", file=sys.stderr)
    print(f"[fallback] op_meta entries : {len(op_meta):,}", file=sys.stderr)

    start_ts = signposts_ts["start"]
    pf_ts = signposts_ts["prefill_done"]
    stop_ts = signposts_ts["stop"]
    # VL 3-stage: ops with host_ts in [start, vision_done) are VISION; [vision_done,
    # prefill_done) are PREFILL. Text 2-stage (no vision_done): [start, prefill_done)
    # is all PREFILL (vis_ts == start_ts -> empty vision bucket).
    vis_ts = signposts_ts.get("vision_done", start_ts)

    vis_pairs, pf_pairs, dec_pairs = [], [], []
    rows_total = 0
    rows_no_meta = 0
    rows_no_dur = 0
    rows_outside = 0

    print(f"[fallback] walking {cpp_csv} ...", file=sys.stderr)
    with open(cpp_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            try:
                op_id = int(row["GLOBAL CALL COUNT"])
            except (KeyError, ValueError):
                continue
            dur_str = (row.get("DEVICE KERNEL DURATION [ns]") or "").strip()
            if not dur_str:
                rows_no_dur += 1
                continue
            try:
                ns = float(dur_str)
            except ValueError:
                rows_no_dur += 1
                continue
            meta = op_meta.get(op_id)
            if meta is None:
                rows_no_meta += 1
                continue
            op_name, host_ts = meta
            if host_ts < start_ts or host_ts > stop_ts:
                rows_outside += 1
                continue
            if host_ts < vis_ts:
                vis_pairs.append((op_name, ns))
            elif host_ts <= pf_ts:
                pf_pairs.append((op_name, ns))
            else:
                dec_pairs.append((op_name, ns))

    print(
        f"[fallback] device rows: {rows_total:,} total / "
        f"{len(vis_pairs):,} vision / {len(pf_pairs):,} prefill / {len(dec_pairs):,} decode / "
        f"{rows_no_meta:,} no_op_meta / {rows_no_dur:,} no_dur / "
        f"{rows_outside:,} outside_window",
        file=sys.stderr,
    )
    return {
        "csv_path": f"fallback (cpp_device_perf_report.csv + tracy_ops_data.csv signposts/meta)",
        "vis_pairs": vis_pairs,
        "pf_pairs": pf_pairs,
        "dec_pairs": dec_pairs,
        "signposts": list(signposts_ts.keys()),
        "n_rows": rows_total,
    }


def main() -> None:
    if len(sys.argv) > 1:
        # Force a specific CSV
        csv_path = pathlib.Path(sys.argv[1])
        vis_pairs, pf_pairs, dec_pairs, signposts, n_rows = [], [], [], [], 0
        section = None
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                n_rows += 1
                op_type = (row.get("OP TYPE") or "").strip().lower()
                op_code = (row.get("OP CODE") or "").strip()
                if op_type == "signpost":
                    signposts.append(op_code)
                    _resolve_s1(op_code, vis_pairs, pf_pairs)
                    section = _section_after(op_code, section)
                    continue
                if section is None:
                    continue
                dur = (row.get("DEVICE KERNEL DURATION [ns]") or "").strip()
                try:
                    ns = float(dur)
                except ValueError:
                    continue
                _bucket(section, op_code, ns, vis_pairs, pf_pairs, dec_pairs)
        result = {
            "csv_path": csv_path,
            "vis_pairs": vis_pairs,
            "pf_pairs": pf_pairs,
            "dec_pairs": dec_pairs,
            "signposts": signposts,
            "n_rows": n_rows,
        }
    else:
        result = _try_processed_csv()
        if result is None:
            print("# (fallback path: post-processed CSV not present)", file=sys.stderr)
            result = _try_fallback_from_logs()
        if result is None:
            print("ERROR: no tracy data found", file=sys.stderr)
            sys.exit(1)

    print(f"# tracy source: {result['csv_path']}")
    print(f"signpost events seen: {result['signposts']}")
    print(f"rows scanned        : {result['n_rows']:,}")

    vis_entries, vis_total = _aggregate_pairs(result.get("vis_pairs", []))
    pf_entries, pf_total = _aggregate_pairs(result["pf_pairs"])
    dec_entries, dec_total = _aggregate_pairs(result["dec_pairs"])

    vis_n_rows = sum(c for _, c, _, _, _ in vis_entries)
    pf_n_rows = sum(c for _, c, _, _, _ in pf_entries)
    dec_n_rows = sum(c for _, c, _, _, _ in dec_entries)
    if vis_n_rows:
        print(f"vision  rows : {vis_n_rows:,}  ({vis_total/1000.0:,.1f} ms total dev work)")
    print(f"prefill rows : {pf_n_rows:,}  ({pf_total/1000.0:,.1f} ms total dev work)")
    print(f"decode  rows : {dec_n_rows:,}  ({dec_total/1000.0:,.1f} ms total dev work)")

    if vis_n_rows:
        _print_table("VISION", vis_entries, vis_total)
    _print_table("PREFILL", pf_entries, pf_total)
    _print_table("DECODE", dec_entries, dec_total)

    vis_categ = _summarize_categories(vis_entries)
    pf_categ = _summarize_categories(pf_entries)
    dec_categ = _summarize_categories(dec_entries)
    print("\n### Category split\n")
    print("| section | total_dev_us | matmul % | CCL % | other % |")
    print("|---|---:|---:|---:|---:|")
    if vis_n_rows:
        print(
            f"| vision  | {vis_categ['total_us']:>11,.1f} | "
            f"{vis_categ['matmul_pct']:.1f} | {vis_categ['ccl_pct']:.1f} | "
            f"{vis_categ['other_pct']:.1f} |"
        )
    print(
        f"| prefill | {pf_categ['total_us']:>11,.1f} | "
        f"{pf_categ['matmul_pct']:.1f} | {pf_categ['ccl_pct']:.1f} | "
        f"{pf_categ['other_pct']:.1f} |"
    )
    print(
        f"| decode  | {dec_categ['total_us']:>11,.1f} | "
        f"{dec_categ['matmul_pct']:.1f} | {dec_categ['ccl_pct']:.1f} | "
        f"{dec_categ['other_pct']:.1f} |"
    )


if __name__ == "__main__":
    main()
