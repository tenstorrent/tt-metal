#!/usr/bin/env python3
"""Where a test_tt_fabric run spends its time: log_phases.py [--stops] <log> [<log> ...]

From the log timestamps: fabric init on device (re)opens, program launch ("Launching programs" -> "Waiting for
programs", where kernels compile on a cold cache) and device wait ("Waiting for programs" -> "Finished"). Launches
over 0.3 s are counted as compile stops; --stops lists them.
"""

import datetime
import re
import sys


def timestamp(line):
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \|", line)
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() if m else None


def main():
    args = sys.argv[1:]
    list_stops = "--stops" in args
    for path in [a for a in args if a != "--stops"]:
        opens = launch = wait = 0.0
        stops = []
        first = last = t_open = t_launch = t_wait = None
        test = None
        for line in open(path, errors="replace"):
            t = timestamp(line)
            if t is None:
                continue
            first = first or t
            last = t
            m = re.search(r"Running Test: (\S+)", line)
            if m:
                test = m.group(1)
            if "Initializing Fabric" in line:
                t_open = t
            elif "Fabric initialized" in line and t_open:
                opens += t - t_open
                t_open = None
            elif "| Launching programs" in line:
                t_launch = t
            elif "| Waiting for programs" in line and t_launch:
                launch += t - t_launch
                if t - t_launch > 0.3:
                    stops.append((t - t_launch, test))
                t_launch = None
                t_wait = t
            elif " Finished. (" in line and t_wait:
                wait += t - t_wait
                t_wait = None
        print(
            f"{path}: span {last - first:.0f} s, fabric opens {opens:.1f} s, launch {launch:.1f} s "
            f"({len(stops)} compile stops, {sum(s for s, _ in stops):.1f} s), device wait {wait:.1f} s"
        )
        if list_stops:
            for seconds, name in stops:
                print(f"    {seconds:6.2f} s  {name}")


if __name__ == "__main__":
    main()
