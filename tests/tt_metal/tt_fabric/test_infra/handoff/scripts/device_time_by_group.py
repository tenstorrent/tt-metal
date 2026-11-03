#!/usr/bin/env python3
"""Per-group device time comparison: device_time_by_group.py <label=log> [<label=log> ...]

Sums the slowest receiver's loop time (ms) per test group (ntype x ftype, over sizes and link counts) for each log,
and the per-group sender loop / slot wait and receiver validate time of the last log.
"""

import collections
import re
import sys


def load(path):
    groups = collections.OrderedDict()
    detail = collections.OrderedDict()
    for line in open(path, errors="replace"):
        m = re.search(r"PERF_BD dev test=(\S+) role=(snd|rcv) .* loop=(\d+)/(\d+)", line)
        if not m:
            continue
        test, role = m.group(1), m.group(2)
        group = (
            re.sub(r"_(mid|short|long)_size_\d+_num_links_\d+", "", test).replace("_ntype", "").replace("_ftype", "")
        )
        d = detail.setdefault(group, collections.Counter())
        if role == "rcv":
            groups[group] = groups.get(group, 0) + int(m.group(4)) / 1e3
            v = re.search(r"validate=(\d+)/(\d+)", line)
            d["rcv_validate_max"] += int(v.group(2)) / 1e3
        else:
            d["snd_loop_max"] += int(m.group(4)) / 1e3
            w = re.search(r"slot_wait=(\d+)/(\d+)", line)
            d["snd_wait_avg"] += int(w.group(1)) / 1e3
    return groups, detail


def main():
    runs = [arg.split("=", 1) for arg in sys.argv[1:]]
    data = [(label, *load(path)) for label, path in runs]
    names = list(collections.OrderedDict.fromkeys(g for _, groups, _ in data for g in groups))
    header = "".join(f"{label:>10s}" for label, _, _ in data)
    print(f"{'group (sum of slowest receiver, s)':52s}{header}   last: snd_loop snd_wait rcv_validate (s)")
    totals = [0.0] * len(data)
    for g in names:
        row = ""
        for k, (_, groups, _) in enumerate(data):
            v = groups.get(g, 0) / 1e3
            totals[k] += v
            row += f"{v:10.2f}"
        d = data[-1][2].get(g, {})
        print(
            f"{g:52s}{row}   {d.get('snd_loop_max', 0) / 1e3:8.2f} {d.get('snd_wait_avg', 0) / 1e3:8.2f} "
            f"{d.get('rcv_validate_max', 0) / 1e3:8.2f}"
        )
    print(f"{'TOTAL':52s}" + "".join(f"{t:10.2f}" for t in totals))


if __name__ == "__main__":
    main()
