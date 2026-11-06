#!/usr/bin/env python3
"""cisp_sender.py RESULTS_DIR [RESULTS_DIR ...]: per cisp_ab.sh run (a tree), per bandwidth test, the profiler on/off
ratio of test_tt_fabric's own sender-side bandwidth (bandwidth_results_blackhole.csv: the mean of bandwidth_GB_s over
the test's sender rows, all iterations) next to the router-telemetry ratio (the "BW (GB/s)=" lines, as cisp_report.py
and the CI test parse them), per pass; then each metric's median and worst over the tests (the CI test's statistic)."""
import collections, csv, glob, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RUNNING = re.compile(r"Running Test: (\S+)")
BANDWIDTH = re.compile(r"BW \(GB/s\)=([0-9.]+)")
WATCH = re.compile(r"NeighborExchangeUnicast_size_4096_ntype_unicast_write|FullRingUnicast_size_4096_ntype_fused|SingleSender")


def telemetry(log):
    bw, running = collections.defaultdict(list), None
    for line in open(log, errors="replace"):
        if m := RUNNING.search(line):
            running = m.group(1)
        elif running and not running.startswith("Latency") and (m := BANDWIDTH.search(line)):
            bw[running].append(float(m.group(1)))
    return {t: statistics.mean(v) for t, v in bw.items()}


def sender(path):
    bw = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["test_name"].startswith("Latency"):
            continue
        bw[r["test_name"]].append(float(r["bandwidth_GB_s"]))
    return {t: statistics.mean(v) for t, v in bw.items()}


def short(t):
    return (t.replace("Unicast_size_", "/").replace("_ntype_unicast_", "/").replace("_ntype_", "/")
            .replace("_num_links_", "/L").replace("SingleSender", "SS").replace("AllDevices", ""))


for d in sys.argv[1:]:
    runs = {}
    for f in glob.glob(os.path.join(d, "sp*_*.log")):
        m = re.match(r"sp([ab])_(\w+)\.log$", os.path.basename(f))
        if not m:
            continue
        p, arm = m.groups()
        csvp = os.path.join(d, f"sp{p}_{arm}_bandwidth_results.csv")
        runs[(p, arm)] = (telemetry(f), sender(csvp) if os.path.exists(csvp) else {})
    print(f"\n## {d}")
    ratios = collections.defaultdict(dict)  # (metric, pass) -> test -> ratio
    for p in ("a", "b"):
        if (p, "off") not in runs or (p, "on") not in runs:
            continue
        (toff, soff), (ton, son) = runs[(p, "off")], runs[(p, "on")]
        for t in set(toff) & set(ton):
            ratios[("telemetry", p)][t] = ton[t] / toff[t]
        for t in set(soff) & set(son):
            ratios[("sender", p)][t] = son[t] / soff[t]
    print("| metric | pass | tests | median | worst | worst test |")
    print("| --- | --- | ---: | ---: | ---: | --- |")
    for (metric, p), r in sorted(ratios.items()):
        if not r:
            continue
        wt = min(r, key=r.get)
        print(f"| {metric} | {p} | {len(r)} | {statistics.median(r.values()):.4f} | {r[wt]:.4f} | {short(wt)} |")
    tests = sorted(set(ratios.get(("sender", "a"), {})) | set(ratios.get(("telemetry", "a"), {})))
    soff_a = runs.get(("a", "off"), ({}, {}))[1]
    print("\n| test | sender off GB/s (a) | sender on/off a | b | telemetry on/off a | b |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for t in tests:
        f = lambda m, p: (f"{ratios[(m, p)][t]:.4f}" if t in ratios.get((m, p), {}) else "-")
        mark = " **" if WATCH.search(t) else ""
        print(f"| {short(t)}{mark} | {soff_a.get(t, float('nan')):.3f} | {f('sender', 'a')} | {f('sender', 'b')} | "
              f"{f('telemetry', 'a')} | {f('telemetry', 'b')} |")
