#!/usr/bin/env python3
"""Parse NAIVE_PROFILER timestamps from training stdout and print per-step and average timings."""

import re
import sys


def parse(lines):
    pattern = re.compile(r"\[NAIVE_PROFILER\]\s+(\S+)\s+timestamp_us=(\d+)")
    timestamps = {}  # marker -> timestamp_us for current step
    steps = []

    PHASE_PAIRS = [
        ("forward_ms", "dataloader_step_done", "forward_pass_done"),
        ("backward_ms", "forward_pass_done", "backward_pass_done"),
        ("gradient_sync_ms", "backward_pass_done", "gradient_sync_done"),
        ("optimizer_ms", "gradient_sync_done", "optimizer_step_done"),
    ]

    for line in lines:
        m = pattern.search(line)
        if not m:
            continue
        marker, ts = m.group(1), int(m.group(2))
        timestamps[marker] = ts

        if marker.startswith("iteration_"):
            row = {}
            for name, start, end in PHASE_PAIRS:
                if start in timestamps and end in timestamps:
                    row[name] = (timestamps[end] - timestamps[start]) / 1000.0
            if row:
                row["step"] = marker
                steps.append(row)
            timestamps.clear()

    return steps, [name for name, _, _ in PHASE_PAIRS]


def main():
    lines = (
        sys.stdin.readlines() if len(sys.argv) < 2 else open(sys.argv[1]).readlines()
    )
    steps, cols = parse(lines)
    if not steps:
        print("No profiler data found.")
        return

    header = f"{'step':<20}" + "".join(f"{c:>20}" for c in cols)
    print(header)
    print("-" * len(header))

    sums = {c: 0.0 for c in cols}
    for row in steps:
        parts = f"{row['step']:<20}"
        for c in cols:
            v = row.get(c, float("nan"))
            sums[c] += v
            parts += f"{v:>20.3f}"
        print(parts)

    n = len(steps)
    print("-" * len(header))
    avg = f"{'average':<20}" + "".join(f"{sums[c] / n:>20.3f}" for c in cols)
    print(avg)
    total_avg = sum(sums[c] for c in cols) / n
    print(f"\naverage total step: {total_avg:.3f} ms")


if __name__ == "__main__":
    main()
