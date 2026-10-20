#!/usr/bin/env python3
"""Bits 26/27/28: cycles/packet per test against the 13129088 / 15226240 baselines, and median net round-trip latency
(ns) against fm_13129088_lat. Same-tree baselines (fmb_*, fm_base_runs.sh) replace today's ak_* when present. Rows that
move by >= 0.5 cycles are listed (latency rows are all listed; "~" marks moves under 5 ns); the rest are counted."""
import csv, os, re, statistics, sys

R = "/home/aagarwal/tt-metal/fabric_exp/results"
BW = []
for f in (80237952, 147346816, 281564544):
    BW += [
        (f"fm_{f}_mix", "ak_13129088_mix"),
        (f"fm_{f}_4352", "ak_13129088_4352"),
        (f"fm_{f}_mix_trim", "ak_13129088_mix_trim"),
        (f"fm_{f}_4352_trim", "ak_13129088_4352_trim"),
    ]
for f in (348673408, 415782272):
    BW += [(f"fm_{f}_mix", "ak_13129088_mix"), (f"fm_{f}_4352", "ak_13129088_4352")]
for f in (82335104, 149443968):
    BW += [(f"fm_{f}w32_4352", "ak_15226240w32_4352"), (f"fm_{f}w32_mix", "ak_15226240w32_mix")]
BW += [("fm_281564544_small", "fm_13129088_small")]
LAT = [(f"fm_{f}_lat", "fm_13129088_lat") for f in (80237952, 147346816, 281564544)]
only = sys.argv[1:]


def load_bw(label):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"], r.get("packet_size", r.get("size", "")))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])  # n = 1 is the trimmed replay row
    return out


def load_lat(label):
    f = os.path.join(R, label + "_latency.csv")
    if not os.path.exists(f):
        return None
    groups = {}
    for r in csv.DictReader(open(f)):
        case = re.sub(r"_sample\d+$", "", r["test_name"])
        groups.setdefault(case, []).append(float(r["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in groups.items()}


for new, base in BW:
    if only and not any(o in new for o in only):
        continue
    same_tree = base.replace("ak_", "fmb_", 1)
    if base.startswith("ak_") and os.path.exists(os.path.join(R, same_tree + ".csv")):
        base = same_tree  # same-tree baseline (fm_base_runs.sh) when it exists
    a, b = load_bw(new), load_bw(base)
    if a is None or b is None:
        print(f'\n{new}: missing ({"new" if a is None else "baseline " + base})')
        continue
    print(f"\n{new} vs {base}:")
    same = 0
    for k in a:
        if k not in b:
            continue
        d = a[k] - b[k]
        name = (
            k[0]
            + ("" if "Mix" in k[0] or "Pure" in k[0] else " " + k[1][4:])
            + (" " + k[2] if k[2] and "Mix" not in k[0] and "Pure" not in k[0] else "")
            + (" (replay)" if k[3] else "")
        )
        if abs(d) >= 0.5:
            print(f"  {name:55s} {b[k]:7.1f} -> {a[k]:7.1f} ({d:+.1f})")
        else:
            same += 1
    print(f"  {same} other rows within 0.5")
for new, base in LAT:
    if only and not any(o in new for o in only):
        continue
    a, b = load_lat(new), load_lat(base)
    if a is None or b is None:
        print(f'\n{new}: missing ({"new" if a is None else "baseline " + base})')
        continue
    print(f"\n{new} vs {base} (median net round trip, ns):")
    for k in a:
        if k in b:
            d = a[k] - b[k]
            print(f'  {k:55s} {b[k]:7.1f} -> {a[k]:7.1f} ({d:+.1f}){"" if abs(d) >= 5 else "  ~"}')
