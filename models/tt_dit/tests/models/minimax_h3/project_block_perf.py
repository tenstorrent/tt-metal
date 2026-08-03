# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Project one profiled MiniMax-H3 transformer block out to a denoise step and a whole video.

Reads the Tracy ops CSV produced by `test_transformer_block_perf_minimax_h3.py`, isolates the warm
iteration between the `start` and `stop` signposts, and multiplies out. Not a test -- the filename
deliberately does not start with `test_` so pytest leaves it alone.

    python project_block_perf.py 5s=path/to/ops_perf_results_A.csv 15s=.../B.csv
    python project_block_perf.py --drop Permute,Slice 15s=.../B.csv

`--drop` takes op-code substrings and reports what the projection becomes without them, which is how
to price a fusion that removes a graph of ops before writing it.

Two totals are always reported and they *bracket* the truth rather than one being correct:

  device only        sums kernel time, ignoring the host dispatch gaps between ops. An underestimate:
                     a real step cannot dispatch for free.
  device + op gap    adds the measured op-to-op latency. An overestimate at this scale: a single
                     profiled block cannot overlap host dispatch with a long stream of work, whereas
                     50 back-to-back layers can.

The bracket is uncalibrated -- nothing here has been checked against a profiled 50-layer run. Treat
the two numbers as bounds, and note the bracket is widest for short sequences, where the block is
short enough that dispatch dominates.

Per-op durations merge the mesh the same way tt-perf-report does: the slowest device for ordinary ops,
since a distributed step only advances as fast as its slowest participant, but the *mean* for
collectives, whose slowest device has peer-wait time folded into its recorded duration. With that rule
the totals here reproduce tt-perf-report's exactly, so these are its numbers multiplied out rather
than a second implementation that could drift from it. See `_per_op`.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

DEFAULT_LAYERS = 50  # MiniMax-H3 num_layers
DEFAULT_STEPS = 50  # denoising steps per video


def _warm_rows(path: str) -> tuple[list[dict[str, str]], bool]:
    """Rows strictly between the `start` and `stop` signposts, plus whether both were found."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    start = stop = None
    for i, row in enumerate(rows):
        if row.get("OP TYPE") != "signpost":
            continue
        if row["OP CODE"] == "start" and start is None:
            start = i
        elif row["OP CODE"] == "stop" and start is not None and stop is None:
            stop = i

    if start is None:
        return rows, False
    return rows[start + 1 : stop if stop is not None else len(rows)], stop is not None


def _per_op(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Aggregate to {op code: {device_ns, gap_ns, calls}}, merging devices by slowest per invocation.

    NOTE: `GLOBAL CALL COUNT` is per-device, not a shared invocation id, so it cannot be used to line
    invocations up across the mesh -- doing so silently counts every op 32 times over. Every device
    executes the same op sequence, so invocations are matched by position within each device's own
    ordered list instead.
    """

    # Collectives are averaged across devices, everything else takes the slowest device. A device that
    # reaches a collective early sits waiting for its peers and that wait is inside its recorded
    # duration, so the slowest device overstates the collective's real cost; the mean is the estimate
    # tt-perf-report uses. Verified against it: max everywhere with mean for AllGather and
    # ReduceScatter reproduces its per-op values and its totals to the nanosecond on all three CSVs.
    def merge(values, op: str) -> float:
        values = list(values)
        if "allgather" in op.lower() or "reducescatter" in op.lower():
            return sum(values) / len(values)
        return max(values)

    by_device: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for row in rows:
        if row.get("OP TYPE") == "signpost":
            continue

        def number(column: str) -> int:
            value = (row.get(column) or "").strip()
            return int(float(value)) if value else 0

        by_device[row["DEVICE ID"]].append(
            (row["OP CODE"], number("DEVICE KERNEL DURATION [ns]"), number("OP TO OP LATENCY [ns]"))
        )

    sequences = list(by_device.values())
    if not sequences:
        return {}
    length = min(len(seq) for seq in sequences)
    if len({len(seq) for seq in sequences}) > 1:
        print(
            f"    WARNING: devices recorded differing op counts {sorted({len(s) for s in sequences})}; using {length}"
        )

    out: dict[str, dict[str, float]] = defaultdict(lambda: {"device_ns": 0.0, "gap_ns": 0.0, "calls": 0})
    for i in range(length):
        op = sequences[0][i][0]
        entry = out[op]
        entry["device_ns"] += merge((seq[i][1] for seq in sequences), op)
        entry["gap_ns"] += merge((seq[i][2] for seq in sequences), op)
        entry["calls"] += 1
    return dict(out)


def _fmt_projection(block_ns: float, layers: int, steps: int) -> str:
    step_ms = block_ns * layers / 1e6
    video_s = step_ms * steps / 1e3
    return f"{block_ns / 1e6:9.2f} ms {step_ms / 1e3:11.2f} s {video_s:12.1f} s"


def report(label: str, path: str, layers: int, steps: int, drop: list[str]) -> None:
    rows, bounded = _warm_rows(path)
    per_op = _per_op(rows)
    if not per_op:
        print(f"\n### {label}: no ops found between signposts in {path}")
        return

    print(f"\n### {label}   ({path.split('/')[-1]})")
    if not bounded:
        print("    WARNING: no start/stop signpost pair; measuring the whole file, which will include")
        print("             compilation and the output readback. Re-profile with the bracketed test.")

    print(f"    {'op':46s} {'calls':>5s} {'block':>9s} {'/step':>13s} {'/video':>14s}")
    total_device = total_gap = 0.0
    dropped_device = dropped_gap = 0.0
    for op, value in sorted(per_op.items(), key=lambda kv: -kv[1]["device_ns"]):
        total_device += value["device_ns"]
        total_gap += value["gap_ns"]
        is_dropped = any(token.lower() in op.lower() for token in drop)
        if is_dropped:
            dropped_device += value["device_ns"]
            dropped_gap += value["gap_ns"]
        mark = " (dropped)" if is_dropped else ""
        print(
            f"    {op[:46]:46s} {int(value['calls']):5d} " f"{_fmt_projection(value['device_ns'], layers, steps)}{mark}"
        )

    print(f"    {'-' * 92}")
    print(f"    {'device only':46s} {'':5s} {_fmt_projection(total_device, layers, steps)}")
    print(f"    {'device + op gap':46s} {'':5s} {_fmt_projection(total_device + total_gap, layers, steps)}")
    if drop:
        kept_device = total_device - dropped_device
        kept_gap = total_gap - dropped_gap
        print(f"    {'-' * 92}")
        print(f"    {'device only, dropped ops removed':46s} {'':5s} {_fmt_projection(kept_device, layers, steps)}")
        print(
            f"    {'device + op gap, dropped ops removed':46s} {'':5s} "
            f"{_fmt_projection(kept_device + kept_gap, layers, steps)}"
        )
        saved = 100 * dropped_device / total_device if total_device else 0.0
        print(f"    -> removes {saved:.1f}% of block device time ({dropped_device / 1e6:.2f} ms/block)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", help="ops CSV paths, optionally 'label=path'")
    parser.add_argument(
        "--layers", type=int, default=DEFAULT_LAYERS, help=f"blocks per step (default {DEFAULT_LAYERS})"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"denoise steps (default {DEFAULT_STEPS})")
    parser.add_argument("--drop", default="", help="comma-separated op-code substrings to price out")
    args = parser.parse_args()

    drop = [token for token in (t.strip() for t in args.drop.split(",")) if token]
    print(f"projecting 1 block -> {args.layers} layers/step -> {args.steps} steps/video")
    print("(block stack only; the refiner, input projections, norm_out and output heads are excluded)")
    for entry in args.csvs:
        label, _, path = entry.partition("=")
        if not path:
            label, path = entry.split("/")[-1], entry
        report(label, path, args.layers, args.steps, drop)


if __name__ == "__main__":
    main()
