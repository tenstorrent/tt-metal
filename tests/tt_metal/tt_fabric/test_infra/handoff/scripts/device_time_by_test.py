#!/usr/bin/env python3
"""Per-test device time comparison: device_time_by_test.py <label=log> [<label=log> ...]

For each test, the slowest receiver's loop time (ms) in each log, from the PERF_BD lines that
TT_FABRIC_TEST_PERF_BREAKDOWN=1 prints, and the SFPU coverage line of the first log.
"""

import collections
import re
import sys


def main():
    runs = [arg.split("=", 1) for arg in sys.argv[1:]]
    times = collections.OrderedDict()
    coverage = {}
    for k, (label, path) in enumerate(runs):
        test = None
        for line in open(path, errors="replace"):
            m = re.search(r"Running Test: (\S+)", line)
            if m:
                test = m.group(1)
            c = re.search(
                r"SFPU coverage: payloads generated for (\d+/\d+) sender configs, checked for (\d+/\d+)", line
            )
            if c and k == 0:
                coverage[test] = f"gen {c.group(1)}, verify {c.group(2)}"
            m = re.search(r"PERF_BD dev test=(\S+) role=rcv .* loop=(\d+)/(\d+)", line)
            if m:
                times.setdefault(m.group(1), {})[label] = int(m.group(3)) / 1e3
    labels = [label for label, _ in runs]
    print(
        f"{'test (slowest receiver loop, ms)':60s}" + "".join(f"{label:>12s}" for label in labels) + "   SFPU coverage"
    )
    for test, by_label in times.items():
        row = "".join(f"{by_label.get(label, float('nan')):12.1f}" for label in labels)
        print(f"{test[:60]:60s}{row}   {coverage.get(test, '')}")


if __name__ == "__main__":
    main()
