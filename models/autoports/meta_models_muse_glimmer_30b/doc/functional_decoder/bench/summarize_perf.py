# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive the six perf windows from the committed tt-perf-report CSVs.

The README perf table and the ``functional_decoder.performance`` block of
``doc/context_contract.json`` quote per-iteration device time.  This script
recomputes those numbers from the filtered ``*_perf_report.csv`` files
(``Device Time`` column, microseconds) and re-runs the two capture-integrity
checks the stage relies on:

* the Tracy console log for the window must contain **no** ``markers were
  dropped`` warning (a dropped-marker capture silently under-counts ops);
* every ``OP Code`` row count in the filtered window must be an exact multiple of
  the replay count, which is read back out of the same Tracy log.

Usage::

    python summarize_perf.py            # rewrite logs/perf_summary.txt
    python summarize_perf.py --check    # exit 1 if it would change or a check fails
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/functional_decoder/
SUMMARY = ROOT / "logs/perf_summary.txt"

#: window id -> (filtered perf-report CSV, Tracy console log, raw ops CSV, signpost)
WINDOWS = (
    (
        "prefill 8192 tok, batch 1  [sliding]",
        "tracy/sliding/prefill_perf_report.csv",
        "logs/tracy_prefill_sliding.log",
        "tracy/sliding/prefill_ops.csv",
        "PERF_PREFILL",
    ),
    (
        "prefill 8192 tok, batch 1  [full]",
        "tracy/full/prefill_perf_report.csv",
        "logs/tracy_prefill_full.log",
        "tracy/full/prefill_ops.csv",
        "PERF_PREFILL",
    ),
    (
        "traced decode @2048        [sliding]",
        "tracy/sliding/decode_perf_report.csv",
        "logs/tracy_decode_sliding.log",
        "tracy/sliding/decode_ops.csv.gz",
        "PERF_DECODE",
    ),
    (
        "traced decode @131071      [sliding]",
        "tracy/sliding/decode_131071_perf_report.csv",
        "logs/tracy_decode_131071_sliding.log",
        "tracy/sliding/decode_131071_ops.csv.gz",
        "PERF_DECODE",
    ),
    (
        "traced decode @2048        [full]",
        "tracy/full/decode_perf_report.csv",
        "logs/tracy_decode_full.log",
        "tracy/full/decode_ops.csv.gz",
        "PERF_DECODE",
    ),
    (
        "traced decode @131071      [full]",
        "tracy/full/decode_131071_perf_report.csv",
        "logs/tracy_decode_131071_full.log",
        "tracy/full/decode_131071_ops.csv.gz",
        "PERF_DECODE",
    ),
)

#: the filtered report rounds each op's device time to 3 decimals of a
#: microsecond, so a whole window may disagree with the raw nanosecond sum by
#: half a millinanosecond per row.  Anything larger means the two files do not
#: describe the same capture.
RECONCILE_TOL_US = 0.5


def replay_count(log: pathlib.Path) -> int:
    """Replays inside the signposted window: ``iters=N`` for decode, 1 for prefill."""
    text = log.read_text(errors="ignore")
    hits = {int(m) for m in re.findall(r"decode perf window done:.*?iters=(\d+)", text)}
    if len(hits) > 1:
        raise SystemExit(f"{log} reports several replay counts {sorted(hits)}")
    if hits:
        return hits.pop()
    if "prefill perf window done:" not in text:
        raise SystemExit(f"{log} contains no perf-window completion line")
    return 1


def dropped_markers(log: pathlib.Path) -> int:
    return len(re.findall(r"markers were dropped", log.read_text(errors="ignore")))


def raw_window(raw_path: pathlib.Path, signpost: str) -> tuple[int, float]:
    """``(op rows, device kernel time us)`` between the signposts of a raw capture.

    Ties the committed ``*_perf_report.csv`` back to the raw Tracy ops CSV it was
    filtered from: the filtered report's ``Device Time`` is microseconds, the raw
    capture's ``DEVICE KERNEL DURATION [ns]`` is nanoseconds, and the two must
    describe the same set of ops.

    Rows are taken in **file order**, not sorted by ``HOST START TS``.  In a
    traced decode window every replayed op carries the host timestamp of trace
    *capture*, which is earlier than the ``PERF_DECODE`` signpost (measured:
    op timestamps end at 6491019686 while the start signpost is at 6504387979),
    so sorting by that column would move every op outside its own window.
    """
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", errors="ignore") as handle:
        rows = list(csv.DictReader(handle))
    inside, ops, total_ns = False, 0, 0
    for row in rows:
        code = (row["OP CODE"] or "").strip()
        if (row["OP TYPE"] or "").strip() == "signpost":
            if code == signpost:
                inside = True
            elif code == f"{signpost}_END":
                inside = False
            continue
        if not inside:
            continue
        duration = (row["DEVICE KERNEL DURATION [ns]"] or "").strip()
        if duration:
            ops += 1
            total_ns += int(duration)
    return ops, total_ns / 1000.0


def measure(csv_path: pathlib.Path, iters: int) -> dict[str, object]:
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit(f"{csv_path} has no op rows")
    device_us = sum(float(r["Device Time"]) for r in rows if r["Device Time"])
    gap_us = sum(float(r["Op-to-Op Gap"]) for r in rows if r["Op-to-Op Gap"])
    per_op = collections.Counter(r["OP Code"].split(" ")[0] for r in rows)
    ragged = {op: n for op, n in per_op.items() if n % iters}
    top = collections.Counter()
    for r in rows:
        if r["Device Time"]:
            top[r["OP Code"].split(" ")[0]] += float(r["Device Time"]) / iters
    return {
        "rows": len(rows),
        "device_us": device_us,
        "ops_per_iter": len(rows) // iters,
        "device_ms": device_us / iters / 1000.0,
        "with_gaps_ms": (device_us + gap_us) / iters / 1000.0,
        "ragged": ragged,
        "top": top.most_common(4),
    }


def render() -> str:
    out = [
        "# Functional decoder performance summary",
        "# generated from the committed tt-perf-report CSVs by bench/summarize_perf.py",
        "# 'Device Time' column, microseconds, divided by the replay count in the Tracy log",
        "",
        f"{'window':38s} {'reps':>4s} {'ops/iter':>8s} {'ms/iter':>9s} {'incl gaps':>9s}  integrity",
    ]
    for label, rel_csv, rel_log, rel_raw, signpost in WINDOWS:
        log = ROOT / rel_log
        iters = replay_count(log)
        drops = dropped_markers(log)
        m = measure(ROOT / rel_csv, iters)
        raw_ops, raw_us = raw_window(ROOT / rel_raw, signpost)
        drift = raw_us - float(m["device_us"])
        problems = []
        if drops:
            problems.append(f"drops={drops}")
        if m["ragged"]:
            problems.append(f"ragged={m['ragged']}")
        if raw_ops != m["rows"]:
            problems.append(f"raw_ops={raw_ops}!=filtered_rows={m['rows']}")
        if abs(drift) > RECONCILE_TOL_US:
            problems.append(f"raw_vs_filtered={drift:+.3f}us")
        integrity = "ok" if not problems else "FAIL " + " ".join(problems)
        out.append(
            f"{label:38s} {iters:4d} {m['ops_per_iter']:8d} "
            f"{m['device_ms']:9.3f} {m['with_gaps_ms']:9.3f}  {integrity}"
        )
    out.append("")
    out.append("# raw-capture reconciliation: DEVICE KERNEL DURATION [ns] summed between the")
    out.append("# signposts of the raw ops CSV vs the filtered report's Device Time [us]")
    out.append(f"{'window':38s} {'raw ops':>8s} {'raw us':>12s} {'filtered us':>12s} {'delta us':>9s}")
    for label, rel_csv, rel_log, rel_raw, signpost in WINDOWS:
        iters = replay_count(ROOT / rel_log)
        m = measure(ROOT / rel_csv, iters)
        raw_ops, raw_us = raw_window(ROOT / rel_raw, signpost)
        out.append(
            f"{label:38s} {raw_ops:8d} {raw_us:12.3f} "
            f"{float(m['device_us']):12.3f} {raw_us - float(m['device_us']):9.3f}"
        )
    out.append("")
    out.append("# per-iteration device time of the four heaviest op codes in each window")
    for label, rel_csv, rel_log, _rel_raw, _signpost in WINDOWS:
        iters = replay_count(ROOT / rel_log)
        m = measure(ROOT / rel_csv, iters)
        parts = ", ".join(f"{op}={us:.1f}us" for op, us in m["top"])
        out.append(f"{label:38s} {parts}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the summary is stale")
    args = ap.parse_args()

    text = render()
    failed = [line for line in text.splitlines() if "FAIL" in line]
    if failed:
        for line in failed:
            print(f"capture integrity failure: {line}", file=sys.stderr)
        return 2

    if args.check:
        if not SUMMARY.is_file() or SUMMARY.read_text() != text:
            print(f"{SUMMARY} is stale against the committed perf CSVs", file=sys.stderr)
            return 1
        print(f"{SUMMARY} matches the committed perf CSVs")
        return 0

    SUMMARY.write_text(text)
    print(f"wrote {SUMMARY} - {len(WINDOWS)} windows, all captures drop-free and op-count aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
