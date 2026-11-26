# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-op device time of a one-row decode step against a B=4 batch step, from the device profiler's per-op report
(``<tracy output>/.logs/cpp_device_perf_report.csv``) of ``tests/decode/test_batch_decode_profile.py``.

Traced replays carry a trace id and a replay session per op. A one-row step alternates two parity traces, a batch
step replays one trace per variant; ``--list`` prints every trace's sessions in time order (on ``--device``), from
which the phases are picked, e.g. for the default positions 256 and 2048::

    python models/experimental/deepseek_v41_flash/tools/batch_profile_report.py .logs/cpp_device_perf_report.csv --list
    python models/experimental/deepseek_v41_flash/tools/batch_profile_report.py .logs/cpp_device_perf_report.csv \\
        --one-row 256=1:2-4,2:2-4 --batch 256=7:258-264 --one-row 2048=5:2-4,6:2-4 --batch 2048=9:1025-1031

(the first session of a phase is left out: it runs after the profiler read). ``RecvAsyncH2D`` is left out of the
totals: a synchronous one-row step's receive waits for the host's packet, which the batch step writes while it
dispatches, so it measures the host, not the device.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

_HOST_WAIT = "RecvAsyncH2DDeviceOperation"


def _sessions(spec: str) -> dict:
    """``"256=1:2-4,2:2-4"`` -> ``(256, {(trace, session), ...})``."""
    pos, parts = spec.split("=", 1)
    keys = set()
    for part in parts.split(","):
        trace, sessions = part.split(":")
        lo, _, hi = sessions.partition("-")
        keys.update((trace, str(s)) for s in range(int(lo), int(hi or lo) + 1))
    return int(pos), keys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--device", default="27", help="the stage chip to read (the TP4 chips run in lockstep)")
    p.add_argument("--list", action="store_true", help="list the traces' replay sessions in time order")
    p.add_argument("--one-row", action="append", default=[], metavar="POS=TRACE:S-S[,...]")
    p.add_argument("--batch", action="append", default=[], metavar="POS=TRACE:S-S[,...]")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()

    phases = {}
    for kind, specs in (("one_row", args.one_row), ("batch", args.batch)):
        for spec in specs:
            pos, keys = _sessions(spec)
            for key in keys:
                phases[key] = (kind, pos)
    ops: dict = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))  # (kind, pos, key) -> op -> [count, ns]
    starts: dict = {}
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["DEVICE ID"] != args.device or not row["METAL TRACE ID"]:
                continue
            key = (row["METAL TRACE ID"], row["METAL TRACE REPLAY SESSION ID"])
            start = int(row["DEVICE KERNEL START CYCLE"] or 0)
            starts[key] = min(starts.get(key, start), start)
            phase = phases.get(key)
            if phase is None and not args.list:
                continue
            entry = ops[(*(phase or ("-", 0)), key)][row["OP NAME"]]
            entry[0] += 1
            entry[1] += float(row["DEVICE KERNEL DURATION [ns]"] or 0)

    if args.list:
        previous = None
        for key in sorted(starts, key=starts.get):
            if key[0] != previous:
                n = sum(v[0] for k, d in ops.items() if k[2] == key for v in d.values())
                print(f"trace {key[0]} from session {key[1]} ({n} ops in that session)")
                previous = key[0]
        return

    per: dict = {}
    for kind, pos in set((kind, pos) for kind, pos, _ in ops):
        keys = [k for k in ops if k[0] == kind and k[1] == pos]
        mean = defaultdict(lambda: [0.0, 0.0])
        for k in keys:
            for op, (n, ns) in ops[k].items():
                mean[op][0] += n / len(keys)
                mean[op][1] += ns / 1e3 / len(keys)
        per[(kind, pos)] = mean
    for pos in sorted({pos for _, pos in per}):
        one, batch = per.get(("one_row", pos)), per.get(("batch", pos))
        if one is None or batch is None:
            continue
        t1 = sum(t for op, (_, t) in one.items() if op != _HOST_WAIT)
        t4 = sum(t for op, (_, t) in batch.items() if op != _HOST_WAIT)
        print(
            f"\nposition {pos}: device kernel time per step (without {_HOST_WAIT}) one-row {t1:.0f} us, "
            f"batch {t4:.0f} us ({t4 / t1:.2f}x)"
        )
        print(f"  {'op':<44}{'n 1row':>8}{'n B4':>8}{'us 1row':>10}{'us B4':>10}{'B4/1row':>9}{'+us':>8}")
        rows = sorted(
            ((op, *one.get(op, (0, 0)), *batch.get(op, (0, 0))) for op in set(one) | set(batch) if op != _HOST_WAIT),
            key=lambda r: -(r[4] - r[2]),
        )
        for op, n1, a, n4, b in rows[: args.top]:
            ratio = f"{b / a:.2f}" if a else "-"
            print(f"  {op[:44]:<44}{n1:8.1f}{n4:8.1f}{a:10.1f}{b:10.1f}{ratio:>9}{b - a:+8.0f}")


if __name__ == "__main__":
    main()
