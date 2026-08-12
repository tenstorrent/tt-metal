# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Turn a Tracy ops CSV plus a probe's ``GROUP`` markers into a comparison table.

The device-time probes (``prefill_matmul_kblock_device*.py``,
``decode_sharded_out_probe.py``) print one ``GROUP <n> <label...>`` line per run
of ``<n>`` consecutive device ops, so the ops CSV can be sliced back into named
groups in emission order.  This script does that slicing and writes the
``logs/*_summary.txt`` files quoted in the README and work log, so those numbers
are reproducible from committed artifacts rather than from an ad-hoc shell
snippet.

Usage::

    python summarize_device_probe.py <probe.log> <ops_perf_results.csv> [--op-code SUBSTR]

``--op-code`` selects which op rows to count (default ``MinimalMatmul``); pass a
comma-separated list when a group contains more than one op kind, and each
group is then reported as the sum of the per-op-code medians (which is what the
"matmul + reshard against one sharded matmul" comparison needs).  A group whose
config the probe reported as ``BLOCKED`` contributes no rows and is skipped
automatically.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import re
import statistics
import sys


def open_text(path: str):
    """Open a probe log or ops CSV, transparently handling the committed ``.gz``.

    The two largest console logs and two of the ops CSVs are committed gzipped
    (the repo rejects files over 500 KB), so the summaries stay regenerable
    without a manual ``gunzip`` step.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, errors="ignore")


DURATION = "DEVICE KERNEL DURATION [ns]"


def parse_groups(log_path: str, default_reps: int = 8) -> list[tuple[int, str]]:
    """``[(row_count, label)]`` in emission order, with blocked groups zeroed."""
    with open_text(log_path) as handle:
        lines = handle.read().splitlines()
    # Two passes: a probe announces its group *before* it finds out whether the
    # op rejects the config, so the rejection line comes after the GROUP line.
    blocked = set()
    for line in lines:
        m = re.match(r"BLOCKED (\S+)\s+M(\d+) K(\d+) N(\d+)", line)
        if m:
            blocked.add((m.group(1), f"M{m.group(2)}_K{m.group(3)}_N{m.group(4)}"))
            continue
        m = re.match(r"BLOCKED (\S+)\s+(\S+?):", line)
        if m:
            blocked.add((m.group(1), m.group(2)))
    groups = []
    for line in lines:
        if not line.startswith("GROUP "):
            continue
        parts = line.split()
        # Round 1 of the k-block probe predates the explicit count field and
        # emits a fixed number of reps per group.
        if parts[1].isdigit():
            count, label = int(parts[1]), " ".join(parts[2:])
        else:
            count, label = default_reps, " ".join(parts[1:])
        tag = parts[-1]
        shape = parts[-2] if len(parts) > 3 else ""
        key = tag.split("_", 1)[1] if tag.startswith(("probe_", "blockedprobe_")) else tag
        if (shape, key) in blocked:
            count = 0
        groups.append((count, label))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("ops_csv")
    ap.add_argument("--op-code", default="MinimalMatmul")
    ap.add_argument("--reps", type=int, default=8, help="group size for logs without an explicit count")
    args = ap.parse_args()

    groups = parse_groups(args.log, args.reps)
    wanted = args.op_code.split(",")
    rows = [r for r in csv.DictReader(open_text(args.ops_csv)) if any(code in r["OP CODE"] for code in wanted)]
    durations = [(next(c for c in wanted if c in r["OP CODE"]), float(r[DURATION])) for r in rows]
    expected = sum(n for n, _ in groups)
    if len(durations) != expected:
        print(f"# MISMATCH: {len(durations)} rows in the CSV, {expected} announced by the log", file=sys.stderr)
        return 1

    print(f"# {args.ops_csv}\n# {len(durations)} '{args.op_code}' rows in {len(groups)} groups")
    per_label: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    index = 0
    for count, label in groups:
        window = durations[index : index + count]
        index += count
        if count >= 8:  # one-shot probe calls are not timing samples
            for code, value in window:
                per_label[label][code].append(value)

    def total(label):
        """Sum of the per-op-code medians in this group."""
        return sum(statistics.median(v) for v in per_label[label].values())

    def breakdown(label):
        parts = per_label[label]
        if len(parts) == 1:
            return ""
        return "  [" + " + ".join(f"{c}:{statistics.median(v)/1e3:.2f}" for c, v in parts.items()) + "]"

    shapes = []
    for label in per_label:
        shape = label.rsplit(" ", 1)[0]
        if shape not in shapes:
            shapes.append(shape)
    for shape in shapes:
        base_label = f"{shape} default"
        if base_label not in per_label:
            base_label = next(
                (
                    k
                    for k in per_label
                    if k.startswith(shape) and "default" in k or k.startswith(shape) and "shipped" in k
                ),
                None,
            )
            if base_label is None:
                continue
        base_med = total(base_label)
        reps = sum(len(v) for v in per_label[base_label].values())
        print(
            f"=== {shape}  {base_label.rsplit(' ', 1)[1]} median {base_med / 1e3:.1f} us "
            f"({reps} rows){breakdown(base_label)} ==="
        )
        for label in per_label:
            if not label.startswith(shape) or label == base_label:
                continue
            med = total(label)
            print(
                f"    {label.rsplit(' ', 1)[1]:26s} {med / 1e3:9.1f} us  "
                f"{(base_med / med - 1) * 100:+6.2f} %{breakdown(label)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
