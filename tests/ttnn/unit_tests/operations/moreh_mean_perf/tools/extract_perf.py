#!/usr/bin/env python3
"""Extract per-case DEVICE KERNEL DURATION from a tracy ops_perf_results CSV.

Usage:
    extract_perf.py <label> [csv_path]      # csv_path defaults to newest report
    extract_perf.py --diff <base.json> <head.json>

The ops CSV has NO test-name column, so cases are recovered by CALL ORDER: the bench
emits exactly one MorehMean device op per case, in CASES order. That zip is then
VERIFIED against each case's expected origin_H by looking for it in the INPUTS
column -- a mis-zip or a dropped/extra op row is a hard error, never a silent
mislabel.
"""

import csv
import glob
import json
import os
import re
import sys

DUR = "DEVICE KERNEL DURATION [ns]"
CALL = "GLOBAL CALL COUNT"
CODE = "OP CODE"
YPAD = "INPUT_0_Y_PAD[LOGICAL]"  # e.g. "32[17]" -> padded 32, logical 17
KSRC = "COMPUTE KERNEL SOURCE"  # names moreh_mean_h.cpp vs moreh_mean_w.cpp
CORES = "CORE COUNT"
NOISE_PCT = 3.0  # perf-measure: deltas within ~2-3% are noise, not a win

HERE = os.path.dirname(os.path.abspath(__file__))

TILE = 32


def _ragged_h(ht):
    return (ht - 1) * TILE + 17


# MUST stay in the same order as CASES in bench_moreh_mean_h.py
EXPECTED = [
    ("ragged_ht1", _ragged_h(1), 2),
    ("ragged_ht4", _ragged_h(4), 2),
    ("ragged_ht16", _ragged_h(16), 2),
    ("ragged_ht32", _ragged_h(32), 2),
    ("aligned_ht4", 4 * TILE, 2),
    ("aligned_ht32", 32 * TILE, 2),
    ("w_control_ragged", _ragged_h(4), 3),
    ("w_control_aligned", 4 * TILE, 3),
]


def newest_csv():
    pat = "generated/profiler/reports/*/ops_perf_results_*.csv"
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    if not hits:
        sys.exit(f"no CSV matching {pat}")
    return hits[-1]


def load(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{csv_path} has no data rows")
    for col in (DUR, CALL, CODE):
        if col not in rows[0]:
            sys.exit(f"{csv_path} lacks {col!r}; got {list(rows[0])[:10]}...")

    hits = []
    for r in rows:
        code = (r.get(CODE) or "").strip()
        if "morehmean" not in code.lower():
            continue
        raw = (r.get(DUR) or "").strip()
        if not raw:
            continue
        try:
            ns = float(raw)
        except ValueError:
            continue
        try:
            call = int((r.get(CALL) or "0").strip())
        except ValueError:
            call = 0
        ypad = (r.get(YPAD) or "").strip()
        m = re.match(r"(\d+)\s*\[(\d+)\]", ypad)
        hits.append(
            {
                "call": call,
                "op": code,
                "ns": ns,
                "logical_h": int(m.group(2)) if m else None,
                "padded_h": int(m.group(1)) if m else None,
                "ksrc": (r.get(KSRC) or "").strip(),
                "cores": (r.get(CORES) or "").strip(),
            }
        )
    hits.sort(key=lambda h: h["call"])
    return hits, rows


def cmd_collect(label, csv_path):
    hits, rows = load(csv_path)
    print(f"== {label}   ({csv_path})")
    print(f"   MorehMean op rows: {len(hits)}   expected: {len(EXPECTED)}")

    if len(hits) != len(EXPECTED):
        print("\nERROR: row count != case count -- refusing to guess the mapping.")
        print("All OP CODEs seen in this CSV:")
        seen = {}
        for r in rows:
            c = (r.get(CODE) or "?").strip()
            seen[c] = seen.get(c, 0) + 1
        for c, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"   {n:>4}  {c}")
        sys.exit(1)

    print()
    print(f"{'case':<20} {'kernel':<18} {'logical_H':>9} {'cores':>6} {'ns':>12}  check")
    print("-" * 82)
    flat = {}
    bad = []
    for (case_id, origin_h, dim), h in zip(EXPECTED, hits):
        # Two independent checks on the call-order zip:
        #  1. logical H from INPUT_0_Y_PAD must equal this case's origin_H
        #  2. the compute kernel file must be the H kernel for dim=2, W kernel for dim=3
        want_kernel = "moreh_mean_h.cpp" if dim == 2 else "moreh_mean_w.cpp"
        ok_h = h["logical_h"] == origin_h
        ok_k = want_kernel in h["ksrc"]
        if not (ok_h and ok_k):
            bad.append((case_id, origin_h, want_kernel, h["logical_h"], h["ksrc"][:60]))
        kernel_short = want_kernel if ok_k else "?"
        print(
            f"{case_id:<20} {kernel_short:<18} {str(h['logical_h']):>9} "
            f"{h['cores']:>6} {h['ns']:>12,.0f}  {'ok' if ok_h and ok_k else 'MISMATCH'}"
        )
        flat[case_id] = h["ns"]

    if bad:
        print("\nERROR: call-order mapping does not match expected shape/kernel:")
        for case_id, origin_h, want_k, got_h, got_k in bad:
            print(
                f"   {case_id}: expected logical_H={origin_h} and {want_k}; " f"got logical_H={got_h}, kernel={got_k!r}"
            )
        sys.exit(1)

    out = os.path.join(HERE, f"{label}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "label": label,
                "csv": csv_path,
                "cases": flat,
                "detail": [{"case": c[0], **h} for c, h in zip(EXPECTED, hits)],
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out}")


def cmd_diff(base_path, head_path):
    base = json.load(open(base_path))
    head = json.load(open(head_path))
    b, h = base["cases"], head["cases"]
    keys = [k for k in b if k in h]
    missing = sorted(set(b) ^ set(h))

    print(f"base = {base['label']}   head = {head['label']}")
    print(f"noise band = +/-{NOISE_PCT}%   (negative delta = head faster)\n")
    print(f"{'case':<20} {'base ns':>12} {'head ns':>12} {'delta':>10} {'':>8}")
    print("-" * 66)

    order = {c[0]: i for i, c in enumerate(EXPECTED)}
    for k in sorted(keys, key=lambda k: order.get(k, 99)):
        pct = (h[k] - b[k]) / b[k] * 100.0
        tag = "noise" if abs(pct) <= NOISE_PCT else ("FASTER" if pct < 0 else "SLOWER")
        print(f"{k:<20} {b[k]:>12,.0f} {h[k]:>12,.0f} {pct:>+9.1f}% {tag:>8}")

    ctl = [k for k in keys if k.startswith("w_control")]
    if ctl:
        worst = max(abs((h[k] - b[k]) / b[k] * 100.0) for k in ctl)
        print()
        if worst <= NOISE_PCT:
            print(f"CONTROL OK: untouched W path moved <= {worst:.1f}% (within noise).")
        else:
            print(
                f"CONTROL FAILED: untouched W path moved {worst:.1f}% "
                f"(> {NOISE_PCT}%). The H numbers above are NOT trustworthy."
            )
    if missing:
        print(f"\nWARNING: cases present in only one run: {missing}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "--diff":
        if len(a) != 3:
            sys.exit("usage: extract_perf.py --diff <base.json> <head.json>")
        cmd_diff(a[1], a[2])
    else:
        cmd_collect(a[0], a[1] if len(a) > 1 else newest_csv())
