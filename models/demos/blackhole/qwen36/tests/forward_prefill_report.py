#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Post-process a Tracy ops-perf CSV into per-stage device-time totals for forward_prefill.

Companion to test_forward_prefill_perf.py. Reads the merged ops report
(ops_perf_results*.csv), keeps only the ops inside each MEASURE_T{T} .. _END
window, and buckets DEVICE KERNEL DURATION [ns] by the PF_* signpost that
precedes them (PF_proj / PF_conv / PF_gdn_core / PF_state_carry / PF_gate_norm /
PF_out_proj / PF_all_reduce). Also prints a coarse op-CATEGORY split
(matmul / elementwise / layout-churn / reduce / norm / scan / ccl / other)
across the whole window, mirroring handoff §0b.

Usage:
    # explicit CSV
    python models/demos/blackhole/qwen36/tests/forward_prefill_report.py \
        generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv

    # auto-pick the newest ops_perf_results*.csv under generated/profiler/reports/
    python models/demos/blackhole/qwen36/tests/forward_prefill_report.py

    # show the top-N op codes within each PF_* region
    python .../forward_prefill_report.py <csv> --top 8

Pure stdlib (csv only) — no pandas / ttnn import, so it runs anywhere the CSV is.
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

# --- column names in the merged report (tools/tracy/process_ops_logs.py OPS_CSV_HEADER) ---
COL_CODE = "OP CODE"
COL_TYPE = "OP TYPE"
COL_DUR = "DEVICE KERNEL DURATION [ns]"

# Logical forward_prefill stages, in data-flow order (matches gdn/tp.py _sp() markers).
PF_ORDER = [
    "PF_proj",
    "PF_conv",
    "PF_gdn_core",
    "PF_state_carry",
    "PF_gate_norm",
    "PF_out_proj",
    "PF_all_reduce",
]

# Coarse op-category heuristics on OP CODE (substring, case-insensitive). Order matters:
# first match wins. These are approximate — an unmatched op lands in "other".
CATEGORY_RULES = [
    ("scan", ("gateddeltaattn",)),
    ("ccl", ("allreduce", "reducescatter", "allgather", "reduce_scatter", "all_gather", "ccl")),
    ("norm", ("rmsnorm", "layernorm", "rms_norm")),
    ("matmul", ("matmul", "linear", "bmm", "moreh_matmul")),
    ("reduce", ("reduce", "moreh_sum", "sum")),
    (
        "elementwise",
        (
            "binaryng",
            "binary_ng",
            "eltwisebinary",
            "eltwiseunary",
            "binary",
            "unary",
            "silu",
            "sigmoid",
            "softplus",
            "exp",
            "recip",
            "mul",
            "add",
            "sub",
            "neg",
            "clip",
        ),
    ),
    (
        "layout-churn",
        (
            "reshape",
            "permute",
            "transpose",
            "tilize",
            "untilize",
            "typecast",
            "slice",
            "tolayout",
            "to_layout",
            "copy",
            "move",
            "interleavedtosharded",
            "shardedto",
            "repeat",
            "concat",
            "pad",
            "clone",
            "l2_norm",
        ),
    ),
]


def categorize(op_code):
    lc = op_code.lower()
    for cat, keys in CATEGORY_RULES:
        if any(k in lc for k in keys):
            return cat
    return "other"


def _clean(s):
    # signpost OP CODE arrives like "PF_proj`" (stray backticks/space from the tracy message).
    return (s or "").strip().strip("`").strip()


def _dur(s):
    s = (s or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def find_latest_csv():
    pats = [
        "generated/profiler/reports/**/ops_perf_results*.csv",
        "generated/profiler/reports/ops_perf_results*.csv",
    ]
    hits = []
    for p in pats:
        hits.extend(glob.glob(p, recursive=True))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def parse(csv_path):
    """Return {T: {"regions": {pf: [ns,...]}, "cats": {cat: [ns,...]}}} for each MEASURE window."""
    windows = {}  # T -> dict
    cur_T = None
    cur_pf = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if COL_CODE not in reader.fieldnames or COL_DUR not in reader.fieldnames:
            sys.exit(
                f"error: '{csv_path}' is missing expected columns.\n"
                f"  need: {COL_CODE!r}, {COL_DUR!r}\n  have: {reader.fieldnames}\n"
                "Is this the merged ops_perf_results CSV (not cpp_device_perf_report.csv)?"
            )
        for r in reader:
            code = _clean(r.get(COL_CODE))
            is_sp = (r.get(COL_TYPE) or "").strip().lower() == "signpost"
            if is_sp:
                m = re.fullmatch(r"MEASURE_T(\d+)", code)
                if m:
                    cur_T = int(m.group(1))
                    cur_pf = "PF_(pre)"
                    windows.setdefault(cur_T, {"regions": defaultdict(list), "cats": defaultdict(list)})
                    continue
                if code.startswith("MEASURE_T") and code.endswith("_END"):
                    cur_T = None
                    cur_pf = None
                    continue
                if code.startswith("PF_"):
                    cur_pf = code
                continue
            # a real op
            if cur_T is None:
                continue  # outside any MEASURE window (warmup / other tests)
            ns = _dur(r.get(COL_DUR))
            windows[cur_T]["regions"][cur_pf or "PF_(pre)"].append((code, ns))
            windows[cur_T]["cats"][categorize(code)].append((code, ns))
    return windows


def _fmt_ms(ns):
    return f"{ns / 1e6:8.3f}"


def report(windows, top=0):
    if not windows:
        print(
            "No MEASURE_T* windows found. Was the run done with GDN_PROFILE=1 and\n"
            "the tracy signposts enabled? (python -m tracy -r -p -m pytest ...)"
        )
        return
    for T in sorted(windows):
        regions = windows[T]["regions"]
        cats = windows[T]["cats"]
        total = sum(ns for lst in regions.values() for _, ns in lst)
        nops = sum(len(lst) for lst in regions.values())
        print("=" * 68)
        print(f"forward_prefill  T={T}    total device-kernel = {_fmt_ms(total)} ms   ({nops} ops)")
        print("=" * 68)

        # ---- per-stage (PF_* signpost) breakdown, in data-flow order ----
        print("  stage (signpost)         ms        %      ops")
        print("  " + "-" * 48)
        ordered = [p for p in PF_ORDER if p in regions] + [p for p in regions if p not in PF_ORDER]
        for pf in ordered:
            lst = regions[pf]
            s = sum(ns for _, ns in lst)
            pct = 100.0 * s / total if total else 0.0
            print(f"  {pf:<22} {_fmt_ms(s)} {pct:6.1f}%  {len(lst):5d}")
            if top:
                by_code = defaultdict(lambda: [0.0, 0])
                for code, ns in lst:
                    by_code[code][0] += ns
                    by_code[code][1] += 1
                for code, (s2, c2) in sorted(by_code.items(), key=lambda kv: -kv[1][0])[:top]:
                    print(f"      {code:<28} {_fmt_ms(s2)} ms  x{c2}")

        # ---- coarse op-category split across the whole window ----
        print("  " + "-" * 48)
        print("  category                 ms        %      ops")
        print("  " + "-" * 48)
        for cat in sorted(cats, key=lambda c: -sum(ns for _, ns in cats[c])):
            lst = cats[cat]
            s = sum(ns for _, ns in lst)
            pct = 100.0 * s / total if total else 0.0
            print(f"  {cat:<22} {_fmt_ms(s)} {pct:6.1f}%  {len(lst):5d}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="ops_perf_results*.csv (default: newest under generated/profiler/reports/)")
    ap.add_argument("--top", type=int, default=0, help="show top-N op codes within each PF_* stage")
    args = ap.parse_args()

    csv_path = args.csv or find_latest_csv()
    if not csv_path:
        sys.exit("error: no CSV given and none found under generated/profiler/reports/**/ops_perf_results*.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"error: no such file: {csv_path}")
    print(f"# reading {csv_path}\n")
    report(parse(csv_path), top=args.top)


if __name__ == "__main__":
    main()
