#!/usr/bin/env python3
"""jitter_compare.py [O3|Os ...] [--label=jit]: the jitter band of v3tk's ordinary edits (jitter_runs.sh, RESULTS §96).

Per level, the base is the mean of the two v3tk runs (first and last in the job); their difference is the run-to-run
noise. For each edit: the change of each suite's mean against the base, and its largest single-row change. Then the
band over all edits: the range of the suite-mean changes, and the spread of every row change (median and 90th
percentile of |change|, and the largest). Cycles/packet = 1.35e9 / packets per second; MixCmd rows excluded, as in
compare.py.
"""
import csv, os, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SUITES = ("4352", "mix", "flushmix", "flushmix4", "1d")
EDITS = ("reorder", "statefield", "hb", "coldmul", "rxreorder", "cursorfield", "init", "templates")


def bw(label, suite):
    f = os.path.join(R, f"{label}_13129088_{suite}.csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        if "MixCmd" in r["test_name"]:
            continue
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r.get("packet_size", ""))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


args = [a for a in sys.argv[1:] if not a.startswith("--label=")]
LABEL = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--label=")), "jit")
for lvl in args or ["O3", "Os"]:
    P = LABEL + {"O3": "3", "Os": "s"}[lvl]
    a = {s: bw(f"{P}_base", s) for s in SUITES}
    b = {s: bw(f"{P}_base2", s) for s in SUITES}
    if not all(a.values()):
        print(f"{lvl}: base runs missing")
        continue
    base = {s: {k: (a[s][k] + b[s][k]) / 2 if b[s] else a[s][k] for k in a[s]} for s in SUITES}
    print(f"\n=== -{lvl}: suite-mean change vs the base build (cycles/packet), largest row change in brackets")
    print("%-12s" % "" + "".join("%18s" % s for s in SUITES))
    if all(b.values()):
        noise = {s: [b[s][k] - a[s][k] for k in a[s]] for s in SUITES}
        print("%-12s" % "base2-base" + "".join(
            "%9.2f [%+6.1f]" % (statistics.mean(noise[s]), max(noise[s], key=abs)) for s in SUITES))
        allnoise = [abs(x) for s in SUITES for x in noise[s]]
        print(f"   run-to-run: median |row change| {statistics.median(allnoise):.2f}, 90th pct "
              f"{pct(allnoise, 0.9):.2f}, max {max(allnoise):.2f}")
    means = {s: [] for s in SUITES}
    rows = []
    for e in EDITS:
        line = "%-12s" % e
        for s in SUITES:
            d = bw(f"{P}_{e}", s)
            if not d:
                line += "%18s" % "-"
                continue
            ch = [d[k] - base[s][k] for k in base[s] if k in d]
            means[s].append(statistics.mean(ch))
            rows += [abs(x) for x in ch]
            line += "%9.2f [%+6.1f]" % (statistics.mean(ch), max(ch, key=abs))
        print(line)
    print("%-12s" % "band" + "".join(
        "%18s" % (f"{min(means[s]):+.2f}..{max(means[s]):+.2f}" if means[s] else "-") for s in SUITES))
    if rows:
        print(f"   all edits, every row: median |change| {statistics.median(rows):.2f}, 90th pct {pct(rows, 0.9):.2f}, "
              f"max {max(rows):.2f} ({len(rows)} row results)")
