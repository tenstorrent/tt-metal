#!/usr/bin/env python3
"""Extract per-case DEVICE KERNEL DURATION for the reduce-partial-scaler suite.

Usage:
    extract_suite.py <label> <bench.py> [csv_path]
    extract_suite.py --diff <base.json> <head.json>

Mapping strategy: the ops CSV has no test-name column, and a single test can emit
more than one device op, so a positional zip is unsafe. Instead each case is matched
SEQUENTIALLY and greedily: scan forward from the last consumed row for the first row
whose OP CODE, COMPUTE KERNEL SOURCE and (where known) reduce-dim logical length all
agree with that case. Every case must match or this exits non-zero -- a silent
mislabel is never produced.

Case metadata is imported from the bench file itself, so there is one source of truth.
"""

import csv
import glob
import importlib.util
import json
import os
import sys

DUR = "DEVICE KERNEL DURATION [ns]"
CALL = "GLOBAL CALL COUNT"
CODE = "OP CODE"
KSRC = "COMPUTE KERNEL SOURCE"
CORES = "CORE COUNT"
NOISE_PCT = 3.0

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cases(bench_path):
    spec = importlib.util.spec_from_file_location("bench_mod", bench_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = getattr(mod, "CASES")
    out = []
    for c in cases:
        if isinstance(c, dict):
            out.append(c)
        else:  # topk bench: (id, K) tuples, no kernel metadata
            out.append({"id": c[0], "op": "TopK", "kernel": "", "axis": None, "logical": c[1]})
    return out


def newest_csv():
    hits = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
    if not hits:
        sys.exit("no profiler CSV found")
    return hits[-1]


def logical_of(row, axis):
    if not axis:
        return None
    raw = (row.get(f"INPUT_0_{axis}_PAD[LOGICAL]") or "").strip()
    if "[" in raw and "]" in raw:
        try:
            return int(raw.split("[", 1)[1].split("]", 1)[0])
        except ValueError:
            return None
    return None


def cmd_collect(label, bench_path, csv_path):
    cases = load_cases(bench_path)
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get(DUR) or "").strip()]
    rows.sort(key=lambda r: int((r.get(CALL) or "0").strip() or 0))

    print(f"== {label}   ({csv_path})")
    print(f"   cases: {len(cases)}   timed op rows in CSV: {len(rows)}\n")
    print(f"{'case':<30}{'op':<24}{'red.dim':>8}{'cores':>6}{'ns':>12}")
    print("-" * 80)

    flat, detail, unmatched = {}, [], []
    idx = 0
    for c in cases:
        found = None
        j = idx
        while j < len(rows):
            r = rows[j]
            code = (r.get(CODE) or "").strip()
            ksrc = (r.get(KSRC) or "").strip()
            ok_op = c["op"].lower() in code.lower()
            ok_k = (not c["kernel"]) or (c["kernel"] in ksrc)
            lg = logical_of(r, c["axis"])
            ok_s = (c["axis"] is None) or (lg == c["logical"])
            if ok_op and ok_k and ok_s:
                found = (j, r, lg)
                break
            j += 1
        if not found:
            unmatched.append(c)
            print(f"{c['id']:<30}{'-- NO MATCH --':<24}")
            continue
        j, r, lg = found
        idx = j + 1
        ns = float((r.get(DUR) or "0").strip())
        code = (r.get(CODE) or "").strip()
        print(
            f"{c['id']:<30}{code[:23]:<24}{str(lg if lg is not None else c['logical']):>8}"
            f"{(r.get(CORES) or '').strip():>6}{ns:>12,.0f}"
        )
        flat[c["id"]] = ns
        detail.append({"case": c["id"], "op": code, "ns": ns, "logical": lg})

    if unmatched:
        print(f"\nERROR: {len(unmatched)} case(s) did not match any CSV row:")
        for c in unmatched:
            print(f"   {c['id']}: op~{c['op']} kernel~{c['kernel'] or '(any)'} " f"{c['axis']}={c['logical']}")
        print("\nOP CODEs present in CSV:")
        seen = {}
        for r in rows:
            k = (r.get(CODE) or "?").strip()
            seen[k] = seen.get(k, 0) + 1
        for k, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"   {n:>4}  {k}")
        sys.exit(1)

    out = os.path.join(HERE, f"{label}.json")
    json.dump({"label": label, "csv": csv_path, "cases": flat, "detail": detail}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


def cmd_diff(base_path, head_path):
    base, head = json.load(open(base_path)), json.load(open(head_path))
    b, h = base["cases"], head["cases"]
    keys = [k for k in b if k in h]
    order = list(b)

    print(f"base = {base['label']}   head = {head['label']}")
    print(f"noise band = +/-{NOISE_PCT}%   (negative = head faster)\n")
    print(f"{'case':<30}{'base ns':>12}{'head ns':>12}{'delta':>10}{'':>9}")
    print("-" * 74)
    for k in sorted(keys, key=lambda k: order.index(k)):
        pct = (h[k] - b[k]) / b[k] * 100.0
        tag = "noise" if abs(pct) <= NOISE_PCT else ("FASTER" if pct < 0 else "SLOWER")
        print(f"{k:<30}{b[k]:>12,.0f}{h[k]:>12,.0f}{pct:>+9.1f}%{tag:>9}")

    ctl = [k for k in keys if "control" in k]
    if ctl:
        worst = max(abs((h[k] - b[k]) / b[k] * 100.0) for k in ctl)
        print()
        if worst <= NOISE_PCT:
            print(f"CONTROLS OK: untouched paths moved <= {worst:.1f}% (within noise).")
        else:
            print(
                f"CONTROLS FAILED: untouched paths moved {worst:.1f}% "
                f"(> {NOISE_PCT}%) -- deltas above are NOT trustworthy."
            )
    missing = sorted(set(b) ^ set(h))
    if missing:
        print(f"\nWARNING: cases in only one run: {missing}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "--diff":
        cmd_diff(a[1], a[2])
    else:
        cmd_collect(a[0], a[1], a[2] if len(a) > 2 else newest_csv())
