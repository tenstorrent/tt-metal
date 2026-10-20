#!/usr/bin/env python3
"""Per test: cycles/packet (1.35 GHz / packets per s) and, for stream senders, header-copy share, slot-wait share,
worker busy cycles per packet and the worker-bound flag. usage: busy_table.py RESULTS_PREFIX ..."""
import csv, re, sys

for prefix in sys.argv[1:]:
    rows = {r["test_name"]: r for r in csv.DictReader(open(prefix + ".csv"))}
    info, flags, test = {}, set(), None
    for line in open(prefix + ".log"):
        m = re.search(r"Running Test: (\S+)", line)
        if m:
            test = m.group(1)
        m = re.search(
            r"(?:copied a header|wrote a header into their router slot) \(([\d.]+)%\).*full \(([\d.]+)%\), worker busy ([\d.]+) cycles/packet",
            line,
        )
        if m and test:
            info[test] = m.groups()
        if "was worker-bound" in line and test:
            flags.add(test)
    print(f"== {prefix.split('/')[-1]}")
    for name, r in rows.items():
        c = info.get(name)
        extra = f"  copy {c[0]:>5}%  wait {c[1]:>5}%  busy {c[2]:>6}" if c else ""
        print(
            f"  {name:20s} {1.35e9 / float(r['avg_packets_per_s']):6.1f} cyc/pkt{extra}{'  WORKER-BOUND' if name in flags else ''}"
        )
