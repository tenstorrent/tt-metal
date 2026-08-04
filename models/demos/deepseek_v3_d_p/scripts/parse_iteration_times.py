#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import statistics
import subprocess
from datetime import datetime
from glob import glob


def parse_timestamp(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")


def print_stats(title, values):
    print(f"=== {title} ===")
    if not values:
        print("No data")
        return
    first, rest = values[0], values[1:]
    print(f"First: {first:.3f}s")
    if rest:
        avg = statistics.mean(rest)
        median = statistics.median(rest)
        stddev = statistics.stdev(rest) if len(rest) > 1 else 0.0
        print(f"Rest ({len(rest)} samples): Avg: {avg:.3f}s | Median: {median:.3f}s | Stddev: {stddev:.3f}s")
        print(f"Min: {min(rest):.3f}s | Max: {max(rest):.3f}s")
    else:
        print("No additional samples")


def parse_file(path, show_sync=False):
    iter_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*Starting iteration: (\d+)")
    sync_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*Starting completion sync on iteration: (\d+)"
    )

    iterations = {}
    syncs = {}

    proc = subprocess.Popen(
        ["grep", "Starting", path],
        stdout=subprocess.PIPE,
        text=True,
    )

    for line in proc.stdout:
        m = iter_pattern.search(line)
        if m:
            ts, num = m.groups()
            iterations[int(num)] = parse_timestamp(ts)
            continue

        m = sync_pattern.search(line)
        if m:
            ts, num = m.groups()
            syncs[int(num)] = parse_timestamp(ts)

    proc.wait()

    iter_deltas = []
    sorted_iters = sorted(iterations.keys())
    for i in range(1, len(sorted_iters)):
        prev, curr = sorted_iters[i - 1], sorted_iters[i]
        if curr == prev + 1:
            delta = (iterations[curr] - iterations[prev]).total_seconds()
            iter_deltas.append(delta)

    print_stats("Iteration Period", iter_deltas)

    if show_sync:
        sync_deltas = []
        for sync_num in sorted(syncs.keys()):
            next_iter = sync_num + 1
            if next_iter in iterations:
                delta = (iterations[next_iter] - syncs[sync_num]).total_seconds()
                sync_deltas.append(delta)

        print()
        print_stats("Sync to Next Iteration", sync_deltas)


def main():
    parser = argparse.ArgumentParser(description="Parse iteration timing from log files")
    parser.add_argument("-f", "--file", required=True, help="Log file or directory containing log_* files")
    parser.add_argument(
        "-n", "--sync", action="store_true", help="Also show 'Sync to Next Iteration' stats (default: off)"
    )
    args = parser.parse_args()

    if os.path.isdir(args.file):
        log_files = sorted(glob(os.path.join(args.file, "log_*")))
        if not log_files:
            print(f"No log_* files found in {args.file}")
            return
        for i, path in enumerate(log_files):
            if i:
                print()
            print(f"########## {os.path.basename(path)} ##########")
            parse_file(path, show_sync=args.sync)
    else:
        parse_file(args.file, show_sync=args.sync)


if __name__ == "__main__":
    main()
