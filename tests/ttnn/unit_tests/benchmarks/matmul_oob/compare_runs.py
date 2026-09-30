# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Join two run_all.sh result directories (e.g. Wormhole reference and a Blackhole run) per case and per test.

  python compare_runs.py results/wormhole_b0 generated/matmul_oob/blackhole_<rev> --out comparison

Writes, into --out:
  suite_by_case.csv   one row per suite case: for each run the legacy and v2 time, status, config and v2/legacy
                      speedup, plus where the two runs disagree (regression in one run only, status changes)
  pytest_by_test.csv  one row per matmul pytest test: outcome off/on and device time off/on for each run
  summary.txt         counts: regressions in each run, shared vs run-specific regressions, status differences
Speedup is legacy time / v2 time; a regression is below 0.95.
"""

import argparse
import csv
import gzip
import json
import math
import os
from collections import Counter


def open_text(path):
    """`path`, or `path`.gz (the published reference results are compressed)."""
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return None


def load_suite(run_dir):
    rows = {}
    for r in csv.DictReader(open_text(os.path.join(run_dir, "suite.csv"))):
        rows.setdefault(r["case"], {})[r["mode"]] = r
    return rows


def load_pytest(run_dir, mode):
    f = open_text(os.path.join(run_dir, f"pytest_{mode}.jsonl"))
    if f is None:
        return {}
    out = {}
    for line in f:
        r = json.loads(line)
        out[r["test"]] = r
    return out


def speedup(modes):
    o, v = modes.get("oob"), modes.get("v2")
    if not o or not v or o["status"] != "ok" or v["status"] != "ok":
        return None
    return float(o["device_ns"]) / float(v["device_ns"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--out", default="comparison")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = [os.path.basename(os.path.normpath(p)) for p in (args.ref, args.new)]

    ref, new = load_suite(args.ref), load_suite(args.new)
    cases = sorted(set(ref) | set(new))
    fields = ["case", "tier", "M", "K", "N", "a_dtype", "b_dtype", "a_mem", "b_mem", "out_mem", "fidelity"]
    per_run = ["legacy_us", "v2_us", "speedup", "legacy_status", "v2_status", "legacy_config", "v2_config"]
    header = fields + [f"{n}_{c}" for n in names for c in per_run] + ["note"]
    counts = Counter()
    with open(os.path.join(args.out, "suite_by_case.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for case in cases:
            runs = [ref.get(case, {}), new.get(case, {})]
            any_row = next((m[k] for m in runs for k in ("oob", "v2") if k in m), None)
            row = [any_row.get(k, "") if any_row else "" for k in fields]
            sp = []
            for modes in runs:
                o, v = modes.get("oob", {}), modes.get("v2", {})
                s = speedup(modes)
                sp.append(s)
                row += [
                    f"{float(o['device_ns']) / 1e3:.1f}" if o.get("device_ns") else "",
                    f"{float(v['device_ns']) / 1e3:.1f}" if v.get("device_ns") else "",
                    f"{s:.3f}" if s else "",
                    o.get("status", ""),
                    v.get("status", ""),
                    o.get("config", ""),
                    v.get("config", ""),
                ]
            notes = []
            reg = [s is not None and s < 0.95 for s in sp]
            if reg[0] and reg[1]:
                notes.append("regression in both")
                counts["regression in both"] += 1
            elif reg[0] or reg[1]:
                notes.append(f"regression only in {names[reg.index(True)]}")
                counts[f"regression only in {names[reg.index(True)]}"] += 1
            for n, modes in zip(names, runs):
                v = modes.get("v2", {})
                o = modes.get("oob", {})
                if v.get("status") not in (None, "ok", "infeasible") and o.get("status") == "ok":
                    notes.append(f"v2 {v['status']} in {n}")
                    counts[f"v2 {v['status']} in {n}"] += 1
            row.append("; ".join(notes))
            w.writerow(row)

    lines = []
    for n, run in zip(names, (ref, new)):
        s = [x for x in (speedup(run[c]) for c in run) if x]
        if s:
            geo = math.exp(sum(math.log(x) for x in s) / len(s))
            lines.append(
                f"{n}: {len(s)} cases, geomean v2/legacy {geo:.3f}, faster >5% {sum(x > 1.05 for x in s)}, "
                f"slower >5% {sum(x < 0.95 for x in s)}"
            )
    lines += [f"{k}: {v}" for k, v in sorted(counts.items())]

    pt = [(load_pytest(d, "off"), load_pytest(d, "on")) for d in (args.ref, args.new)]
    tests = sorted(set().union(*[set(a) | set(b) for a, b in pt]))
    if tests:
        with open(os.path.join(args.out, "pytest_by_test.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test"] + [f"{n}_{c}" for n in names for c in ("off", "on", "off_us", "on_us", "speedup")])
            for t in tests:
                row = [t]
                for off, on in pt:
                    a, b = off.get(t, {}), on.get(t, {})
                    ta, tb = a.get("device_ns"), b.get("device_ns")
                    row += [
                        a.get("outcome", ""),
                        b.get("outcome", ""),
                        f"{ta / 1e3:.1f}" if ta else "",
                        f"{tb / 1e3:.1f}" if tb else "",
                        f"{ta / tb:.3f}" if ta and tb else "",
                    ]
                w.writerow(row)
        for n, (off, on) in zip(names, pt):
            changed = [t for t in off if t in on and off[t]["outcome"] != on[t]["outcome"]]
            lines.append(f"{n} pytest: {len(off)} off, {len(on)} on, {len(changed)} outcome changes: {changed[:10]}")
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
