#!/usr/bin/env python3
"""cisp_skew.py ZONES_DIR: start / end skew between neighbouring chips' sender kernels (the 823 zone CSV, profiler on),
slow against normal NX write iterations (user's question, 2026-11-05).
- Kernel zones: BRISC-KERNEL start/end per (chip, core); raw cycles per chip with a constant per-chip offset.
- Iterations: each chip's sync core (15,11) runs one kernel per iteration and spans its senders; sender kernels (x=15,
  y != 11) are assigned to the sync kernel whose start is nearest (all start within ~0.1 us of each other).
- Per (chip, iteration): first sender start, last sender end.
- Per neighbour pair (A, B): d(j) = t_A(j) - t_B(j), less its median over the CSV-normal NX write iterations (the
  constant offset and any fixed launch-order skew drop out); deviations in us at 1.35 GHz.
- Slow: the CSV row (chip 7 W) below 97% of the off median; each chip's low routers from the log's router telemetry
  (cisp_router_tally.py), so pairs are split by whether A sources a low-rate flow in that iteration."""
import collections, csv, os, re, statistics, sys

D = sys.argv[1]
sys.argv = ["x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cisp_router_tally as T  # noqa: E402

z = collections.defaultdict(list)  # (chip, x, y) -> [(type, t)]
with open(os.path.join(D, "zones_on_zones.csv")) as f:
    next(f); next(f)
    for line in f:
        p = [x.strip() for x in line.split(",")]
        if len(p) > 11 and p[10] == "BRISC-KERNEL":
            z[(int(p[0]), int(p[1]), int(p[2]))].append((p[11], int(p[5])))
kern = {}
for k, ev in z.items():
    ev.sort(key=lambda e: e[1])
    st = [t for ty, t in ev if ty == "ZONE_START"]; en = [t for ty, t in ev if ty == "ZONE_END"]
    kern[k] = list(zip(st, en))
chips = sorted({k[0] for k in kern})
# The zone CSV's chip is the PCIe slot; the router telemetry uses physical chip ids from the cluster descriptor.
# Map by the mesh: sync cores exist on every chip; assume slot = physical chip id (checked below against neighbour count).
it = {}  # (chip, j) -> (first sender start, last sender end)
for c in chips:
    sync = kern[(c, 15, 11)]
    starts = [s for s, e in sync]
    import bisect
    acc = collections.defaultdict(lambda: [None, None])
    for (cc, x, y), ks in kern.items():
        if cc != c or x != 15 or y == 11:
            continue
        for s, e in ks:
            j = bisect.bisect_left(starts, s)
            cand = [q for q in (j - 1, j) if 0 <= q < len(starts)]
            j = min(cand, key=lambda q: abs(starts[q] - s))
            if abs(starts[j] - s) > 2_000_000:  # 1.5 ms: not this iteration's
                continue
            a = acc[j]
            a[0] = s if a[0] is None else min(a[0], s)
            a[1] = e if a[1] is None else max(a[1], e)
    for j, (s, e) in acc.items():
        it[(c, j)] = (s, e)
n_it = len(kern[(chips[0], 15, 11)])
rows = list(csv.DictReader(open(os.path.join(D, "zones_on_bandwidth_results.csv"))))
offm = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(D, "zones_off_bandwidth_results.csv"))):
    offm[T.KEY.search(r["test_name"]).group(0)].append(float(r["bandwidth_GB_s"]))
offm = {k: statistics.median(v) for k, v in offm.items()}
tel = T.parse(os.path.join(D, "zones_on.log")); toff = T.parse(os.path.join(D, "zones_off.log"))
offrows = list(csv.DictReader(open(os.path.join(D, "zones_off_bandwidth_results.csv"))))
base = collections.defaultdict(list)
for (n, t), r in zip(toff, offrows):
    for rt, cyc in t.items():
        base[(T.rowkey(n, r), rt)].append(cyc)
base = {k: statistics.median(v) for k, v in base.items()}
assert len(rows) == n_it == len(tel), (len(rows), n_it, len(tel))
write = [j for j in range(n_it) if "unicast_write" in rows[j]["test_name"]]
slow = {j: float(rows[j]["bandwidth_GB_s"]) < 0.97 * offm[T.KEY.search(rows[j]["test_name"]).group(0)] for j in write}
lowsrc = {j: {rt[0] for rt, cyc in tel[j][1].items() if cyc / base[(T.rowkey(tel[j][0], rows[j]), rt)] - 1 > 0.02}
          for j in write}
# mesh neighbours only (the cluster descriptor also lists links the 2x4 mesh doesn't use)
pairs = sorted({tuple(sorted((a[0], b[0]))) for a, b in T.peer.items()
                if abs(T.chip2c[a[0]][0] - T.chip2c[b[0]][0]) + abs(T.chip2c[a[0]][1] - T.chip2c[b[0]][1]) == 1})
print(f"{n_it} iterations, {len(write)} NX write; CSV-slow {sum(slow.values())}; chips {chips}; neighbour pairs {pairs}")
res = collections.defaultdict(list)
for a, b in pairs:
    for w, idx in (("start", 0), ("end", 1)):
        d = {j: it[(a, j)][idx] - it[(b, j)][idx] for j in write if (a, j) in it and (b, j) in it}
        m = statistics.median(d[j] for j in d if not slow[j])
        for j, v in d.items():
            dev = (v - m) / 1350.0
            sa, sb = a in lowsrc[j], b in lowsrc[j]
            grp = ("both low" if sa and sb else "one low" if sa or sb else "neither low") + (" (CSV slow)" if slow[j] else "")
            res[(w, grp)].append(abs(dev))
            if sa != sb:  # orient: positive = the low-source chip is later than its neighbour
                res[(w, "low chip minus neighbour")].append(dev if sa else -dev)
print("|deviation| of neighbour-pair skew from its normal-iteration median, us (1.35 GHz): median / p90 / max, n")
for k in sorted(res):
    v = sorted(res[k])
    if k[1] == "low chip minus neighbour":
        continue
    print(f"   {k[0]:5s} {k[1]:28s} {statistics.median(v):7.2f} {v[int(0.9 * (len(v) - 1))]:7.2f} {v[-1]:7.2f}  n={len(v)}")
for w in ("start", "end"):
    v = sorted(res[(w, "low chip minus neighbour")])
    if v:
        print(f"   {w}: low-source chip minus its neighbour (signed, us): median {statistics.median(v):+.2f}, "
              f"p10 {v[int(0.1 * (len(v) - 1))]:+.2f}, p90 {v[int(0.9 * (len(v) - 1))]:+.2f}, n={len(v)}")
# per chip: sender kernel duration in slow vs normal iterations (own clock, no offset needed)
print("per chip, sender span (first start to last end), us: normal median / iterations where this chip sources a low "
      "router / where another chip does")
for c in chips:
    nrm = [(it[(c, j)][1] - it[(c, j)][0]) / 1350 for j in write if (c, j) in it and not lowsrc[j]]
    own = [(it[(c, j)][1] - it[(c, j)][0]) / 1350 for j in write if (c, j) in it and c in lowsrc[j]]
    oth = [(it[(c, j)][1] - it[(c, j)][0]) / 1350 for j in write if (c, j) in it and lowsrc[j] and c not in lowsrc[j]]
    f = lambda v: f"{statistics.median(v):8.1f} (n={len(v)})" if v else "      -"
    print(f"   chip {c}: {f(nrm)} {f(own)} {f(oth)}")
