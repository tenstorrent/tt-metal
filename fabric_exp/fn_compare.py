#!/usr/bin/env python3
"""Reworked bits 26/27/28 (fn_*, fn_runs.sh): cycles/packet per test against same-tree baselines (fnb_*), with the
first version's number (fm_*, RESULTS §35) alongside, and median net round-trip latency (ns) against fn_13129088_lat.
Rows where either version moves by >= 0.5 cycles are listed (latency rows are all listed); the rest are counted."""
import csv, os, re, statistics, sys

R = "/home/aagarwal/tt-metal/fabric_exp/results"
BW = []
for f in (80237952, 147346816, 281564544):
    BW += [
        (f"{f}_mix", "13129088_mix"),
        (f"{f}_4352", "13129088_4352"),
        (f"{f}_mix_trim", "13129088_mix_trim"),
        (f"{f}_4352_trim", "13129088_4352_trim"),
    ]
for f in (348673408, 415782272):
    BW += [(f"{f}_mix", "13129088_mix"), (f"{f}_4352", "13129088_4352")]
for f in (82335104, 149443968):
    BW += [(f"{f}w32_4352", "15226240w32_4352"), (f"{f}w32_mix", "15226240w32_mix")]
SMALL = [("281564544_small", "13129088_small")]
LAT = [(f"{f}_lat", "13129088_lat") for f in (80237952, 147346816, 281564544)]
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


def row_name(k):
    plain = "Mix" in k[0] or "Pure" in k[0]
    return (
        k[0]
        + ("" if plain else " " + k[1][4:])
        + (" " + k[2] if k[2] and not plain else "")
        + (" (replay)" if k[3] else "")
    )


def compare(new, base, old, old_base):
    if only and not any(o in new for o in only):
        return
    a, b, o, ob = load_bw(new), load_bw(base), load_bw(old), load_bw(old_base)
    if a is None or b is None:
        print(f'\n{new}: missing ({"new" if a is None else "baseline " + base})')
        return
    print(f"\n{new} vs {base}  (base -> new (delta)   [{old} delta vs {old_base}]):")
    same = 0
    for k in a:
        if k not in b:
            continue
        d = a[k] - b[k]
        od = o[k] - ob[k] if o and ob and k in o and k in ob else None
        if abs(d) >= 0.5 or (od is not None and abs(od) >= 0.5):
            print(
                f"  {row_name(k):55s} {b[k]:7.1f} -> {a[k]:7.1f} ({d:+5.1f})"
                + (f"   [{od:+5.1f}]" if od is not None else "")
            )
        else:
            same += 1
    print(f"  {same} other rows within 0.5 in both")


for new, base in BW:
    compare(
        "fn_" + new,
        "fnb_" + base,
        "fm_" + new,
        "fmb_" + base if os.path.exists(f"{R}/fmb_{base}.csv") else "ak_" + base,
    )
for new, base in SMALL:
    compare("fn_" + new, "fn_" + base, "fm_" + new, "fm_" + base)
for new, base in LAT:
    if only and not any(o in new for o in only):
        continue
    a, b = load_lat("fn_" + new), load_lat("fn_" + base)
    o, ob = load_lat("fm_" + new), load_lat("fm_" + base)
    if a is None or b is None:
        print(f"\nfn_{new}: missing")
        continue
    print(f"\nfn_{new} vs fn_{base} (median net round trip, ns)   [fm_ delta]:")
    for k in a:
        if k in b:
            d = a[k] - b[k]
            od = o[k] - ob[k] if o and ob and k in o and k in ob else None
            print(f"  {k:55s} {b[k]:7.1f} -> {a[k]:7.1f} ({d:+6.1f})" + (f"   [{od:+6.1f}]" if od is not None else ""))
