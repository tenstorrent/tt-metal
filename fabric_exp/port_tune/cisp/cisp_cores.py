#!/usr/bin/env python3
"""cisp_cores.py TEST OFF_LOG ON_LOG: per router (mesh coordinate, eth core) telemetry cycles of TEST in two
cisp_ab.sh logs, off/on cycle ratio, and the spread."""
import re, sys, statistics, collections
T = sys.argv[1]
R = re.compile(r"Running Test: (\S+)"); C = re.compile(r"Telemetry from MeshCoordinate\(\[(\d+), (\d+)\]\) core (\S+): BW \(GB/s\)=([0-9.]+), pps=([0-9.]+), cycles=(\d+), eth_words_sent=(\d+), packets_sent=(\d+)")
def cores(log):
    d, cur = collections.defaultdict(list), None
    for l in open(log, errors="replace"):
        if m := R.search(l): cur = m.group(1)
        elif cur == T and (m := C.search(l)):
            d[(m.group(1), m.group(2), m.group(3))].append((int(m.group(6)), int(m.group(8))))
    return d
off, on = cores(sys.argv[2]), cores(sys.argv[3])
rs = []
for k in sorted(off):
    co = statistics.mean(c for c, _ in off[k]); cn = statistics.mean(c for c, _ in on.get(k, [(0,0)]))
    rs.append(co / cn if cn else 0)
    print(k, "off cyc %.0f on cyc %.0f off/on %.4f pk %s/%s" % (co, cn, co / cn if cn else 0, off[k][0][1], on.get(k, [(0, 0)])[0][1]))
print("n", len(rs), "min %.4f median %.4f max %.4f" % (min(rs), statistics.median(rs), max(rs)))
