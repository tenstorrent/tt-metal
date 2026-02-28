#!/usr/bin/env python3
"""Parse NAIVE_PROFILER timestamps from training stdout and print per-step and average timings."""

import math
import re
import sys

WARMUP_STEPS = 2

PHASE_PAIRS = [
    ("forward_ms", "dataloader_step_done", "forward_pass_done"),
    ("backward_ms", "forward_pass_done", "backward_pass_done"),
    ("gradient_sync_ms", "backward_pass_done", "gradient_sync_done"),
    ("optimizer_ms", "gradient_sync_done", "optimizer_step_done"),
]
PHASE_COLS = [name for name, _, _ in PHASE_PAIRS] + ["other_ms"]
COLS = PHASE_COLS + ["total_ms"]


def parse(lines):
    pattern = re.compile(r"\[NAIVE_PROFILER\]\s+(\S+)\s+timestamp_us=(\d+)")
    timestamps = {}  # marker -> timestamp_us for current step
    steps = []
    prev_optimizer_done_ts = None

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

            # "other" = everything between optimizer_step_done of the previous
            # iteration and dataloader_step_done of this iteration.
            if (
                prev_optimizer_done_ts is not None
                and "dataloader_step_done" in timestamps
            ):
                row["other_ms"] = (
                    timestamps["dataloader_step_done"] - prev_optimizer_done_ts
                ) / 1000.0

            if "optimizer_step_done" in timestamps:
                prev_optimizer_done_ts = timestamps["optimizer_step_done"]

            if row:
                row["total_ms"] = sum(row.get(c, 0.0) for c in PHASE_COLS)
                row["step"] = marker
                steps.append(row)
            timestamps.clear()

    return steps


def main():
    lines = (
        sys.stdin.readlines() if len(sys.argv) < 2 else open(sys.argv[1]).readlines()
    )
    steps = parse(lines)
    if not steps:
        print("No profiler data found.")
        return

    header = f"{'step':<20}" + "".join(f"{c:>20}" for c in COLS)
    print(header)
    print("-" * len(header))

    for row in steps:
        parts = f"{row['step']:<20}"
        for c in COLS:
            v = row.get(c, float("nan"))
            parts += f"{v:>20.3f}"
        print(parts)

    avg_steps = steps[WARMUP_STEPS:] if len(steps) > WARMUP_STEPS else steps

    sums = {c: 0.0 for c in COLS}
    counts = {c: 0 for c in COLS}
    for row in avg_steps:
        for c in COLS:
            v = row.get(c, float("nan"))
            if not math.isnan(v):
                sums[c] += v
                counts[c] += 1

    print("-" * len(header))
    label = f"avg (skip {WARMUP_STEPS})"
    avg_line = f"{label:<20}"
    for c in COLS:
        avg_line += f"{sums[c] / counts[c]:>20.3f}" if counts[c] > 0 else f"{'N/A':>20}"
    print(avg_line)


if __name__ == "__main__":
    main()
