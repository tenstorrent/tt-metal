#!/usr/bin/env python3
"""Aggregate the A/B sweep CSVs into an old-vs-new comparison table.
Per cell: PM IDEAL [ns] (constant per op/topo), median+min DEVICE KERNEL DURATION [ns] over the
AllGather rows, util = PM_IDEAL / DEV_median. Then old vs new per (op,dtype) + per-layer totals."""
import csv, glob, os, sys, statistics as st

OPS = ["q_heads", "out_seq", "qdev_heads"]
DTS = ["bf16", "bfp8"]
CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else "ab_results/csv"
# tag suffix is the topology; infer from the first csv found (line|ring)
_topos = {os.path.basename(p).rsplit("_", 1)[-1].replace(".csv", "") for p in glob.glob(os.path.join(CSV_DIR, "*.csv"))}
TOPO = next(iter(_topos)) if _topos else "line"


def load(tag):
    f = os.path.join(CSV_DIR, f"{tag}.csv")
    if not os.path.exists(f):
        return None
    rows = [r for r in csv.DictReader(open(f)) if "AllGather" in r.get("OP CODE", "")]
    if not rows:
        return None
    dev = [int(r["DEVICE KERNEL DURATION [ns]"]) for r in rows if r.get("DEVICE KERNEL DURATION [ns]", "").strip()]
    pm = [int(r["PM IDEAL [ns]"]) for r in rows if r.get("PM IDEAL [ns]", "").strip()]
    if not dev:
        return None
    return {
        "n": len(dev),
        "pm": st.median(pm) if pm else 0,
        "dev_med": st.median(dev),
        "dev_min": min(dev),
        "op_code": rows[0]["OP CODE"],
    }


def us(ns):
    return ns / 1000.0


print(
    f"{'op':<11} {'dtype':<5} {'impl':<4} {'PM IDEAL us':>11} {'DEV med us':>11} {'DEV min us':>11} {'util%':>7} {'n':>5}"
)
print("-" * 74)
totals = {}  # (impl,dt) -> sum dev_med us
for op in OPS:
    for dt in DTS:
        for impl in ["old", "new"]:
            d = load(f"{impl}_{op}_{dt}_{TOPO}")
            if not d:
                print(f"{op:<11} {dt:<5} {impl:<4}  <missing/failed>")
                continue
            util = 100.0 * d["pm"] / d["dev_med"] if d["dev_med"] else 0
            print(
                f"{op:<11} {dt:<5} {impl:<4} {us(d['pm']):>11.2f} {us(d['dev_med']):>11.2f} "
                f"{us(d['dev_min']):>11.2f} {util:>6.1f}% {d['n']:>5}"
            )
            totals.setdefault((impl, dt), 0.0)
            totals[(impl, dt)] += us(d["dev_med"])
        print()

print("=== per-layer total (sum of 3 gathers, DEV median us) ===")
for dt in DTS:
    o = totals.get(("old", dt))
    n = totals.get(("new", dt))
    if o and n:
        delta = 100.0 * (n - o) / o
        print(f"{dt}: old={o:.1f} us  new={n:.1f} us  delta={delta:+.1f}%")
    else:
        print(f"{dt}: old={o}  new={n}  (incomplete)")
