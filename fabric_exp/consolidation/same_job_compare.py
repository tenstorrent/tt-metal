#!/usr/bin/env python3
"""same_job_compare.py CONTROL VARIANT [VARIANT ...] [--rows N]: cycles/packet of each variant against the control run in
the same job (labels LABEL_13129088_<suite> in fabric_exp/results), per suite (mean, min, max; MixCmd rows left out:
bimodal), the control's drift against --prev CONTROL2 when given, and latency (median net ns per row) when the
LABEL_13129088_lat_latency.csv files exist. Rows moving by at least --rows cycles are listed. 1.35 GHz."""
import argparse, collections, csv, os, statistics

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SUITES = ("4352", "mix", "flushmix", "flushmix4", "burst", "1d_m1", "ring", "torusy", "mux", "rse")


def rows(label, suite):
    out = {}
    f = f"{R}/{label}_13129088_{suite}.csv"
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r["packet_size"])
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def lat(label):
    f = f"{R}/{label}_13129088_lat_latency.csv"
    if not os.path.exists(f):
        return {}
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        d[r["test_name"].rsplit("_sample", 1)[0]].append(float(r["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control"); ap.add_argument("variants", nargs="+")
    ap.add_argument("--prev"); ap.add_argument("--rows", type=float, default=0)
    a = ap.parse_args()
    head = "".join(f"  {v:>9} {'min':>6} {'max':>6}" for v in a.variants)
    print(f"{'suite':10} {'rows':>4}{head}" + ("  ctl drift" if a.prev else ""))
    moving = []
    for s in SUITES:
        b = rows(a.control, s)
        vs = [rows(v, s) for v in a.variants]
        ks = [k for k in b if all(k in v for v in vs) and "MixCmd" not in k[0]]
        if not ks:
            continue
        line = f"{s:10} {len(ks):4}"
        for v in vs:
            d = [v[k] - b[k] for k in ks]
            line += f"  {statistics.mean(d):+9.2f} {min(d):+6.1f} {max(d):+6.1f}"
        if a.prev:
            p = rows(a.prev, s)
            dr = [b[k] - p[k] for k in ks if k in p]
            line += f"  {statistics.mean(dr):+9.2f}" if dr else ""
        print(line)
        for k in ks:
            ds = [v[k] - b[k] for v in vs]
            if a.rows and max(abs(x) for x in ds) >= a.rows:
                moving.append((s, k, b[k], ds))
    lb = lat(a.control)
    if lb:
        print("latency, median net ns vs control:")
        lv = [lat(v) for v in a.variants]
        for k in lb:
            print(f"  {k:52} {lb[k]:8.1f}" + "".join(f"  {l[k] - lb[k]:+7.1f}" for l in lv if k in l))
    for s, k, b, ds in moving:
        print(f"  {s:9} {k[0]:28} {k[1]:28} {k[2]:>5}  ctl {b:7.1f}" + "".join(f"  {d:+6.1f}" for d in ds))


if __name__ == "__main__":
    main()
