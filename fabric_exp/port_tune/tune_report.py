#!/usr/bin/env python3
"""tune_report.py SET [--prefix pt] [--rows] [--base ARM] [SUITE ...]: the receiver tuning sweep (tune_runs.sh,
RECEIVER_TUNING.md) per suite: cycles/packet (1.35 GHz) of every arm of SET in passes a and b, the suite mean per arm
and pass, and each arm's mean change against the base arm (default: i4 for idle, c32 for proof, iN_d0 for pert rows of
idle N). MixCmd rows are left out of means (bimodal, as census_report.py). Rows whose passes differ by more than 3% are
marked '~'.
Rows: single-rank suites, one per (test, ntype, size) of the summary CSV (a stream mix is one row: its link total).
vc1 / vc1i, from the per-sender CSV: per test the slowest sender (census_report.py's rule) and, where two or more
senders share a link (same device and direction), that link's total (1.35e9 / the summed packet rates; the slowest
shared link); 'score' is the link total for tests with a shared link, else the slowest sender.
pert: per idle arm, the rows against no delay, and the slope over the delays (tx 8/16/32, rp 8/16/32)."""
import argparse
import collections
import csv
import os
import statistics

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CLK = 1.35e9
ARMS = {"idle": ["i1", "i2", "i4", "i8"], "proof": ["c8", "c32", "c128"], "idlealloc": ["i2", "i4", "i8"],
        "pert": [f"i{i}_{d}" for i in (1, 2, 4, 8) for d in ("d0", "tx8", "tx16", "tx32", "rp8", "rp16", "rp32")]}
BASE = {"idle": "i4", "proof": "c32", "idlealloc": "i4"}
SUITES = ["2d", "1d", "mix", "burst", "burst612", "vc1", "vc1i"]


def rows_summary(f):
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r["packet_size"])
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = CLK / float(r["avg_packets_per_s"])
    return out


def rows_vc1(f):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        by[r["test_name"]].append(r)
    out = {}
    for t, rs in by.items():
        slow = max(CLK / float(r["packets_per_second"]) for r in rs)
        links = collections.defaultdict(list)
        for r in rs:
            links[(r["device"], r["direction"])].append(float(r["packets_per_second"]))
        shared = [CLK / sum(v) for v in links.values() if len(v) > 1]
        out[(t, "", "", 0)] = max(shared) if shared else slow
    return out


def load(prefix, p, s, arm, suite):
    if suite in ("vc1", "vc1i"):
        f = f"{R}/{prefix}{p}_{s}_{arm}_{suite}_tests.csv"
        return rows_vc1(f) if os.path.exists(f) else None
    f = f"{R}/{prefix}{p}_{s}_{arm}_{suite}.csv"
    return rows_summary(f) if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=list(ARMS))
    ap.add_argument("suites", nargs="*")
    ap.add_argument("--prefix", default="pt")
    ap.add_argument("--rows", action="store_true")
    ap.add_argument("--base")
    a = ap.parse_args()
    arms = ARMS[a.set]
    suites = a.suites or (["pert"] if a.set == "pert" else SUITES)
    for suite in suites:
        data = {}
        for arm in arms:
            ds = {p: load(a.prefix, p, a.set, arm, suite) for p in "ab"}
            if all(d is None for d in ds.values()):
                continue
            data[arm] = ds
        if not data:
            continue
        cols = list(data)
        keys = sorted(set.intersection(*[set(d) for arm in cols for d in data[arm].values() if d]))
        mean_keys = [k for k in keys if "MixCmd" not in k[0]]
        groups = [cols] if a.set != "pert" else [[c for c in cols if c.startswith(f"i{i}_")] for i in (1, 2, 4, 8)]
        for g in groups:
            if not g:
                continue
            base = a.base or (BASE.get(a.set) if a.set != "pert" else g[0].split("_")[0] + "_d0")
            print(f"\n## {suite} [{a.set}] ({len(mean_keys)} rows in means; cycles/packet, pass a / pass b; base {base})")
            print(f"{'row':52}" + "".join(f"{c:>17}" for c in g))
            if a.rows:
                for k in keys:
                    lab = f"{k[0]} {k[1]} {k[2]}" + (f" #{k[3]}" if k[3] else "")
                    cells = ""
                    for c in g:
                        va, vb = (data[c][p][k] if data[c][p] and k in data[c][p] else float("nan") for p in "ab")
                        noisy = abs(va - vb) > 0.03 * min(va, vb)
                        cells += f"{va:8.1f}/{vb:7.1f}{'~' if noisy else ' '}"
                    print(f"{lab[:52]:52}" + cells)
            means = {}
            for c in g:
                means[c] = {p: statistics.mean(data[c][p][k] for k in mean_keys) for p in "ab" if data[c][p]}
            print(f"{'MEAN':52}" + "".join(f"{means[c].get('a', float('nan')):8.2f}/{means[c].get('b', float('nan')):7.2f} " for c in g))
            if base in means:
                print(f"{'MEAN - base':52}" + "".join(
                    f"{means[c].get('a', 0) - means[base].get('a', 0):+8.2f}/{means[c].get('b', 0) - means[base].get('b', 0):+7.2f} "
                    for c in g))
                # rows that moved by more than both passes' own spread at the base, and by more than 0.5 cycles
                for c in g:
                    if c == base:
                        continue
                    moved = []
                    for k in keys:
                        try:
                            d = [data[c][p][k] - data[base][p][k] for p in "ab"]
                        except (KeyError, TypeError):
                            continue
                        spread = abs(data[base]["a"][k] - data[base]["b"][k])
                        if all(abs(x) > max(0.5, spread) for x in d) and (d[0] > 0) == (d[1] > 0):
                            moved.append((k, d))
                    if moved:
                        print(f"   {c}: {len(moved)} rows moved beyond max(0.5, base spread) in both passes, same sign:")
                        for k, d in sorted(moved, key=lambda x: -abs(sum(x[1])))[:12]:
                            print(f"      {k[0]} {k[1]} {k[2]}: {d[0]:+.1f} / {d[1]:+.1f}")


if __name__ == "__main__":
    main()
