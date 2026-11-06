#!/usr/bin/env python3
"""idlesw_report.py [--glx JOB]: the IDLE_PASSES_BEFORE_PARTIAL_CREDIT sweep on e7fe1899049 (RECEIVER_TUNING.md section 8).
Per row and arm (i4 = the default value, i8, i16, i32, i64): the value of each pass, their mean and spread, and the mean's
change against i4. Rows are cycles/packet at 1.35 GHz (summary CSV; VC1: tune_report.py's score, the slowest shared link
or sender); latency rows are the median net round trip in ns over each pass's 1000 samples.
A row is flagged as a regression when its mean is worse than i4's by more than 1 cycle (latency: 1 cycle, 0.74 ns) or by
more than 0.5%, and both passes moved that way beyond i4's own pass spread.
  LoudBox (lb2 180292): results/pti{a,b}_idlesw_<arm>_<suite>[.csv|_tests.csv|_latency.csv]
  Galaxy (--glx JOB): results/glx<JOB>/{a,b}_<arm>_<suite>.csv"""
import argparse, csv, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_report as TR  # noqa: E402

R = TR.R
ARMS = ["i4", "i8", "i16", "i32", "i64"]
LB_SUITES = ["2d", "1d", "burst", "mix", "vc1", "nx4096", "ring", "lat"]
GLX_SUITES = ["nx", "mesh", "linear", "ring"]


def lat_rows(f):
    g = {}
    for r in csv.DictReader(open(f)):
        g.setdefault(re.sub(r"_sample\d+$", "", r["test_name"]), []).append(float(r["net_avg_ns"]))
    return {(k, "", "", 0): statistics.median(v) for k, v in g.items()}


def load_lb(p, arm, suite):
    base = f"{R}/pti{p}_idlesw_{arm}_{suite}"
    if suite == "lat":
        return lat_rows(base + "_latency.csv") if os.path.exists(base + "_latency.csv") else None
    if suite in ("vc1", "vc1i"):
        return TR.rows_vc1(base + "_tests.csv") if os.path.exists(base + "_tests.csv") else None
    return TR.rows_summary(base + ".csv") if os.path.exists(base + ".csv") else None


def load_glx(job, p, arm, suite):
    f = f"{R}/glx{job}/{p}_{arm}_{suite}.csv"
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r["packet_size"])
        out[k + (sum(1 for kk in out if kk[:3] == k),)] = TR.CLK / float(r["avg_packets_per_s"])
    return out


def report(title, suites, loader, unit_of):
    flagged = {a: [] for a in ARMS[1:]}
    print(f"# {title}")
    for suite in suites:
        data = {arm: {p: loader(p, arm, suite) for p in "ab"} for arm in ARMS}
        if not all(data[a][p] for a in ARMS for p in "ab"):
            print(f"\n## {suite}: incomplete ({[(a, p) for a in ARMS for p in 'ab' if not data[a][p]]})")
            continue
        keys = sorted(set.intersection(*[set(data[a][p]) for a in ARMS for p in "ab"]))
        unit = unit_of(suite)
        print(f"\n## {suite} ({len(keys)} rows, {unit}; per arm: mean [spread] change vs i4)")
        print(f"{'row':58}" + "".join(f"{a:>24}" for a in ARMS))
        for k in keys:
            lab = (f"{k[0]} {k[1]} {k[2]}" + (f" #{k[3]}" if k[3] else "")).strip()
            base = [data["i4"][p][k] for p in "ab"]
            bm, bs = statistics.mean(base), abs(base[0] - base[1])
            cells = ""
            for a in ARMS:
                v = [data[a][p][k] for p in "ab"]
                m, s = statistics.mean(v), abs(v[0] - v[1])
                d = m - bm
                cells += f"{m:10.2f} [{s:4.2f}] {d:+6.2f}" if a != "i4" else f"{m:10.2f} [{s:4.2f}]       "
                if a != "i4":
                    lim = 1.0 if unit == "cycles/packet" else 1e9 / TR.CLK
                    dp = [v[i] - base[i] for i in range(2)]
                    if (d > lim or d > 0.005 * bm) and all(x > bs for x in dp):
                        flagged[a].append((suite, lab, bm, m, d, 100 * d / bm))
            print(f"{lab[:58]:58}" + cells)
    print("\n## regressions against i4 (more than 1 cycle or 0.5%, both passes beyond i4's spread)")
    for a in ARMS[1:]:
        print(f"   {a}: {len(flagged[a])} rows" + "".join(
            f"\n      {s}: {lab}: {bm:.2f} -> {m:.2f} ({d:+.2f}, {dp:+.2f}%)" for s, lab, bm, m, d, dp in flagged[a]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glx")
    a = ap.parse_args()
    if a.glx:
        report(f"Galaxy 200G, Slurm {a.glx}", GLX_SUITES, lambda p, arm, s: load_glx(a.glx, p, arm, s),
               lambda s: "cycles/packet")
    else:
        report("LoudBox 400G, lb2 180292 (bh-lb-120-a04u24)", LB_SUITES, load_lb,
               lambda s: "ns" if s == "lat" else "cycles/packet")


if __name__ == "__main__":
    main()
