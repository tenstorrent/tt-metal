#!/usr/bin/env python3
"""txo_report.py [--chain R1,R2,...] PASS [PASS ...] (a copy of txp_report.py for txo_runs.sh labels; with --chain, the
deltas are each router against the one before it in the chain): cycles/packet (1.35 GHz) of txp_runs.sh runs per row and router: each pass and the
mean. Rows: the summary CSV (for the vc02 rows, the link: the profiler gives the link the longest sender's cycles), the
vc02 rows' per-sender rates from the profiler's log lines (VC0 and VC2 senders), and the VC1 split / inter-mesh tests
(slowest sender per test, as census_report.py)."""
import collections, csv, glob, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import census_report as cr
R = cr.R
LINE = re.compile(r"(?:Sender core \S+ \(VC(\d)\) on device \d+ direction \S+ link \d+: (\d+) packets in (\d+) cycles"
                  r"|another sender core \S+ \(VC(\d)\) with (\d+) packets in (\d+) cycles)")
chain = sys.argv[sys.argv.index("--chain") + 1].split(",") if "--chain" in sys.argv else None
passes = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and (i < 2 or sys.argv[i - 1] != "--chain")]
data = collections.defaultdict(lambda: collections.defaultdict(dict))  # row -> router -> pass -> value
routers = []
for p in passes:
    for f in sorted(glob.glob(f"{R}/txo{p}__*__*.csv")):
        m = re.match(rf"txo{p}__([^_]+(?:_[^_]+)*)__(.+)\.csv$", os.path.basename(f))
        if not m or m.group(2).endswith("_detail") or m.group(2).endswith("_latency"):
            continue
        router, tag = m.groups()
        routers.append(router) if router not in routers else None
        vals = [1.35e9 / float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f))]
        if vals:
            data[tag + (" (link)" if tag.startswith(("v8-", "d-vc0")) else "")][router][p] = statistics.mean(vals)
        if tag.startswith(("v8-", "d-vc0")):
            text = re.sub(r"\x1b\[[0-9;]*m", "", open(f[:-4] + ".log", errors="replace").read())
            per = collections.defaultdict(list)
            for g in LINE.findall(text):
                vc, pk, cy = (g[0], g[1], g[2]) if g[0] else (g[3], g[4], g[5])
                per[vc].append(int(cy) / int(pk))
            for vc, v in per.items():
                data[f"{tag} VC{vc} sender"][router][p] = statistics.mean(v)
    for kind in ("vc1", "vc1i"):
        for f in sorted(glob.glob(f"{R}/txo{p}{kind}__*__o3_13129088_vc1_tests.csv")):
            router = re.match(rf"txo{p}{kind}__(.+)__o3_", os.path.basename(f)).group(1)
            rows = cr.rows(f) or {}
            means = []
            for k, v in rows.items():
                if "MixCmd" in k[0]:
                    continue
                lab = f"{kind}: {k[0].replace('_ntype_unicast_write', '')}"
                data[lab][router][p] = v
                means.append(v)
            if means:
                data[f"{kind}: MEAN"][router][p] = statistics.mean(means)
if chain:
    routers = [r for r in chain if r in routers]
base = routers[0] if routers else None
print(f"{'row':58}" + "".join(f"{r:>24}" for r in routers) + "".join(f"{'d ' + r:>10}" for r in routers[1:]))
for row in sorted(data, key=lambda k: (k.startswith("vc1"), k)):
    per = data[row]
    cells, means = "", {}
    for r in routers:
        v = per.get(r, {})
        if v:
            means[r] = statistics.mean(v.values())
            cells += f"{means[r]:8.1f} ({' / '.join(f'{v[p]:.1f}' for p in passes if p in v)})".rjust(24)
        else:
            cells += f"{'-':>24}"
    prev = {r: (routers[k - 1] if chain else base) for k, r in enumerate(routers) if k > 0}
    deltas = "".join(f"{means[r] - means[prev[r]]:+10.1f}" if r in means and prev[r] in means else f"{'-':>10}"
                     for r in routers[1:])
    print(f"{row[:58]:58}" + cells + deltas)
