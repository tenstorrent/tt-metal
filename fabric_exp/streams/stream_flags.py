#!/usr/bin/env python3
"""Per test: cycles/packet (1.35 GHz / packets per s), header-copy share, slot-wait share and the worker-bound flag,
read from a run's summary CSV and log. usage: stream_flags.py RESULTS_PREFIX ..."""
import csv, re, sys

for prefix in sys.argv[1:]:
    rows = {r["test_name"]: r for r in csv.DictReader(open(prefix + ".csv"))}
    info, test = {}, None
    for line in open(prefix + ".log"):
        m = re.search(r"Running Test: (\S+)", line)
        if m:
            test = m.group(1)
        m = re.search(
            r"(?:copied a header|wrote a header into their router slot) \(([\d.]+)%\), \d+ found the router's slots full \(([\d.]+)%\)",
            line,
        )
        if m and test:
            info.setdefault(test, []).append(f"copy {float(m.group(1)):5.1f}% wait {float(m.group(2)):5.1f}%")
        if "was worker-bound" in line and test:
            info.setdefault(test, []).append("WORKER-BOUND")
    print(f"== {prefix.split('/')[-1]}")
    for name, r in rows.items():
        print(f"  {name:22s} {1.35e9 / float(r['avg_packets_per_s']):7.1f} cyc/pkt  {'  '.join(info.get(name, []))}")
