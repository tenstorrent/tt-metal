#!/usr/bin/env python3
"""cisp_router_map.py RESULTS_DIR [--tests REGEX] [--thresh F]: per router and per test iteration, the telemetry cycles of
the profiler-on runs against the profiler-off run of the same pass (cisp_ab.sh logs sp<pass>_<arm>.log), with each
router's physical chip, eth channel, peer and link-sync role (ERISC0: transmitter when its chip id is below the peer's,
link_sync.cpp role_of; every link of the 2x4 P150 mesh is synced). Topology: fabric_exp/a09u32_cluster_desc.yaml and
the run's physical_chip_mesh_coordinate_mapping (both checked against the telemetry CSVs' link columns: 40 routers,
0 mismatches). Off baseline per router and test: the median of its off iterations. A hit: an on iteration more than
--thresh (default 0.005) above the baseline.

Prints, per test matching --tests, every router's per-iteration deltas (k cycles) per pass, then a per-router census over
all matching tests (hits / iterations, by role), and the hit sizes."""
import collections, glob, os, re, statistics, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CD = os.path.join(HERE, "..", "..", "a09u32_cluster_desc.yaml")
MAP = os.path.join(HERE, "a09u32_hw_mesh_mapping.yaml")  # the hardware runs' mapping (report (c)); generated/fabric is rewritten by every run, mock runs included
RUN = re.compile(r"Running Test: (\S+)")
TEL = re.compile(r"Telemetry from MeshCoordinate\(\[(\d+), (\d+)\]\) core 0-(\d+): .*cycles=(\d+), eth_words_sent=\d+, "
                 r"packets_sent=(\d+)")


def topology():
    cd = yaml.safe_load(open(CD))
    peer = {}
    for a, b in cd["ethernet_connections"]:
        peer[(a["chip"], a["chan"])] = (b["chip"], b["chan"])
        peer[(b["chip"], b["chan"])] = (a["chip"], a["chan"])
    m = yaml.safe_load(open(MAP))["chips"]
    return peer, {tuple(v): k for k, v in m.items()}


def iterations(log):
    """{test: [ {router: cycles} per iteration ]}; a new iteration at each 'Running Test' line."""
    out, cur = collections.defaultdict(list), None
    for line in open(log, errors="replace"):
        if m := RUN.search(line):
            cur = m.group(1)
            out[cur].append({})
        elif cur and (m := TEL.search(line)):
            r, c, ch, cyc, pk = map(int, m.groups())
            out[cur][-1][((r, c), ch)] = cyc
    return {t: [it for it in its if it] for t, its in out.items()}


def main():
    d = sys.argv[1]
    treg = re.compile(sys.argv[sys.argv.index("--tests") + 1]) if "--tests" in sys.argv else re.compile(".")
    thr = float(sys.argv[sys.argv.index("--thresh") + 1]) if "--thresh" in sys.argv else 0.005
    peer, c2chip = topology()
    runs = {}
    for f in glob.glob(os.path.join(d, "sp*_*.log")):
        m = re.match(r"sp([ab])_(\w+)\.log$", os.path.basename(f))
        if m:
            runs[(m.group(1), m.group(2))] = iterations(f)
    census = collections.defaultdict(lambda: [0, 0])  # router -> [hits, iterations]
    sizes = []
    lines = []
    for (p, arm) in sorted(runs):
        if arm == "off" or (p, "off") not in runs:
            continue
        off, on = runs[(p, "off")], runs[(p, arm)]
        for t in sorted(set(off) & set(on)):
            if t.startswith("Latency") or not treg.search(t):
                continue
            routers = sorted(set().union(*off[t]) & set().union(*on[t]))
            for rt in routers:
                base = statistics.median(it[rt] for it in off[t] if rt in it)
                deltas = [it[rt] - base for it in on[t] if rt in it]
                offd = [it[rt] - base for it in off[t] if rt in it]
                hits = [x for x in deltas if x > thr * base]
                census[rt][0] += len(hits)
                census[rt][1] += len(deltas)
                sizes += [(x, x / base) for x in hits]
                lines.append((t, p, arm, rt, base, deltas, offd))
    def name(rt):
        (r, c), ch = rt
        chip = c2chip[(r, c)]
        pc, pch = peer.get((chip, ch), (None, None))
        role = "-" if pc is None else ("T" if chip < pc else "R")
        return f"[{r},{c}] chip{chip} eth{ch:<2d} -> chip{pc}/eth{pch:<2d} {role}"
    cur = None
    for t, p, arm, rt, base, deltas, offd in sorted(lines, key=lambda x: (x[0], x[3], x[1], x[2])):
        if t != cur:
            cur = t
            print(f"\n== {t}  (k cycles vs the off median; off iterations in brackets)")
        print(f"  {name(rt)}  pass {p} {arm:6s} base {base / 1e6:7.3f}M  on " +
              " ".join(f"{x / 1e3:+6.0f}" for x in deltas) + "   [" + " ".join(f"{x / 1e3:+5.0f}" for x in offd) + "]")
    print(f"\n== census over matching tests: on iterations more than {thr:.1%} above the off median")
    by_role = collections.defaultdict(lambda: [0, 0])
    for rt in sorted(census, key=lambda r: (c2chip[r[0]], r[1])):
        h, n = census[rt]
        role = name(rt).split()[-1]
        by_role[role][0] += h
        by_role[role][1] += n
        print(f"  {name(rt)}  {h:3d} / {n:3d}")
    print("  by role: " + ", ".join(f"{k}: {h} / {n}" for k, (h, n) in sorted(by_role.items())))
    if sizes:
        ks = sorted(x for x, _ in sizes)
        print(f"  hit sizes: n {len(ks)}, k cycles min {ks[0] / 1e3:.0f} median {statistics.median(ks) / 1e3:.0f} "
              f"max {ks[-1] / 1e3:.0f}; fraction median {statistics.median(f for _, f in sizes):.2%}")


if __name__ == "__main__":
    main()
