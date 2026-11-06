#!/usr/bin/env python3
"""cisp_report.py RESULTS_DIR: per bandwidth test of the streaming-profiler overhead runs (cisp_ab.sh), each arm's
ratio to the profiler-off run of the same pass (bandwidth parsed as test_streaming_profiler_sync.py does: per
"Running Test:" the mean of its "BW (GB/s)=" lines, latency tests left out), then per arm and pass the median and worst
ratio against the test's bounds (median 0.996, worst 0.989), and the NeighborExchange row's pass-to-pass spread."""
import glob, os, re, statistics, sys

RUNNING = re.compile(r"Running Test: (\S+)")
BANDWIDTH = re.compile(r"BW \(GB/s\)=([0-9.]+)")
MEDIAN_BOUND, WORST_BOUND = 0.996, 0.989
WATCHES = ["NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2",
           "NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1"]


def bandwidth(log):
    bw, running = {}, None
    for line in open(log, errors="replace"):
        if m := RUNNING.search(line):
            running = m.group(1)
        elif running and not running.startswith("Latency") and (m := BANDWIDTH.search(line)):
            bw.setdefault(running, []).append(float(m.group(1)))
    return {t: statistics.mean(v) for t, v in bw.items()}


d = sys.argv[1]
runs = {}
for f in glob.glob(os.path.join(d, "sp*_*.log")):
    m = re.match(r"sp([ab])_(\w+)\.log$", os.path.basename(f))
    if m:
        runs[(m.group(1), m.group(2))] = bandwidth(f)
arms = sorted({a for _, a in runs if a != "off"}, key=lambda a: ["on", "nohook"].index(a) if a in ("on", "nohook") else 9)
passes = sorted({p for p, _ in runs})
ratios = {}
for p in passes:
    off = runs.get((p, "off"))
    for a in arms:
        on = runs.get((p, a))
        if off and on:
            common = set(off) & set(on)
            ratios[(p, a)] = {t: on[t] / off[t] for t in common}
print("## Per arm and pass (ratio to the profiler-off run of the same pass)\n")
print("| arm | pass | tests | median | worst | worst test | median >= 0.996 | worst >= 0.989 |")
print("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
for (p, a), r in sorted(ratios.items(), key=lambda x: (x[0][1], x[0][0])):
    med = statistics.median(r.values()); wt = min(r, key=r.get)
    print(f"| {a} | {p} | {len(r)} | {med:.4f} | {r[wt]:.4f} | {wt} | {'yes' if med >= MEDIAN_BOUND else 'NO'} | "
          f"{'yes' if r[wt] >= WORST_BOUND else 'NO'} |")
for WATCH in WATCHES:
    print(f"\n## {WATCH}\n")
    for (p, a) in sorted(runs):
        if WATCH in runs[(p, a)]:
            print(f"- {a} pass {p}: {runs[(p, a)][WATCH]:.3f} GB/s" + (f", ratio {ratios[(p, a)][WATCH]:.4f}" if (p, a) in ratios and WATCH in ratios[(p, a)] else ""))
    vals = [runs[(p, 'off')][WATCH] for p in passes if (p, 'off') in runs and WATCH in runs[(p, 'off')]]
    if len(vals) == 2:
        print(f"- off pass-to-pass: {vals[1] / vals[0] - 1:+.4%}")
print("\n## Per test (ratio per pass)\n")
tests = sorted(set().union(*(set(r) for r in ratios.values()))) if ratios else []
head = "| test | " + " | ".join(f"{a} {p}" for a in arms for p in passes) + " | off b / off a |"
print(head)
print("| --- | " + " | ".join("---:" for _ in head.split("|")[2:-1]) + " |")
for t in tests:
    cells = [f"{ratios[(p, a)][t]:.4f}" if (p, a) in ratios and t in ratios[(p, a)] else "-" for a in arms for p in passes]
    oo = f"{runs[('b', 'off')][t] / runs[('a', 'off')][t]:.4f}" if ('a', 'off') in runs and ('b', 'off') in runs and t in runs[('a', 'off')] and t in runs[('b', 'off')] else "-"
    print(f"| {t} | " + " | ".join(cells) + f" | {oo} |")
