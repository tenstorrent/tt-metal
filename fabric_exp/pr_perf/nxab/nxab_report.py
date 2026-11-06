#!/usr/bin/env python3
"""nxab_report.py RESULTS_DIR [MAIN_RESULTS_DIR]: the 200G NX A/B (frs_glx_nxab.sbatch). Per arm, cycles/packet
(1.35e9 / avg_packets_per_s, as the Galaxy records) for each pass at 3072 / 4096 / 4352 B, the mean, the change against
arm a, and port - main against MAIN_RESULTS_DIR's main (default the 179756 copy: {a,b}_main_nx.csv, 1 link)."""
import csv, os, statistics, sys

R = sys.argv[1]
M = sys.argv[2] if len(sys.argv) > 2 else "/home/aagarwal/tt-metal/fabric_exp/results/glx179756"
ARMS = ["a", "b10", "b30", "b60", "c16", "c64", "d64", "e60"]
SIZES = [3072, 4096, 4352]


def read(path):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if int(r.get("num_links") or 1) == 1 and int(r["packet_size"]) in SIZES:
            out[int(r["packet_size"])] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


main = {s: statistics.mean([d[s] for d in (read(f"{M}/a_main_nx.csv"), read(f"{M}/b_main_nx.csv")) if s in d])
        for s in SIZES}
data = {arm: {p: read(f"{R}/{p}_{arm}_nx.csv") for p in "ab"} for arm in ARMS}
base = {s: statistics.mean([data["a"][p][s] for p in "ab" if s in data["a"][p]]) for s in SIZES if any(
    s in data["a"][p] for p in "ab")}
print(f"main (179756, 1 link, mean of passes): " + ", ".join(f"{s} B {main[s]:.2f}" for s in SIZES))
print(f"{'arm':5s}" + "".join(f"{str(s) + ' B: pass a / b, mean':>34s}{'vs a':>8s}{'- main':>8s}" for s in SIZES))
for arm in ARMS:
    row = f"{arm:5s}"
    for s in SIZES:
        v = [data[arm][p].get(s) for p in "ab"]
        have = [x for x in v if x is not None]
        if not have:
            row += f"{'-':>34s}{'':>8s}{'':>8s}"
            continue
        m = statistics.mean(have)
        row += f"{' / '.join(f'{x:.2f}' if x is not None else '-' for x in v) + f', {m:.2f}':>34s}"
        row += f"{m - base[s]:+8.2f}" if s in base else f"{'':>8s}"
        row += f"{m - main[s]:+8.2f}"
    print(row)
