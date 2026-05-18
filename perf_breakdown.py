#!/usr/bin/env python3
"""Per-phase breakdown of a tt-perf ops CSV using signposts.

Usage:
    perf_breakdown.py <ops_perf_results.csv> [--top N]

Reads the CSV emitted by tracy (ops_perf_results_*.csv), groups ops between
signpost pairs (FWD_BEGIN/FWD_END, BWD_BEGIN/BWD_END, OPT_BEGIN/OPT_END, ...),
and prints a per-phase device-time breakdown.

Phase pairs are auto-discovered: any signpost whose label starts with
"<NAME>_BEGIN" is paired with the next "<NAME>_END" with a matching tail
(everything after the prefix), so multi-step runs work too.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

import pandas as pd

_BEGIN = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9]*)_BEGIN(?P<tail>.*)$")
_END = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9]*)_END(?P<tail>.*)$")


def discover_phases(signposts: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Pair BEGIN/END signposts and return [(label, begin_ts, end_ts), ...]."""
    open_begins: dict[tuple[str, str], float] = {}
    phases: list[tuple[str, float, float]] = []
    for _, row in signposts.iterrows():
        label = row["OP CODE"]
        ts = row["HOST START TS"]
        if (m := _BEGIN.match(label)) is not None:
            open_begins[(m["name"], m["tail"].strip())] = ts
            continue
        if (m := _END.match(label)) is not None:
            key = (m["name"], m["tail"].strip())
            if (begin_ts := open_begins.pop(key, None)) is not None:
                tag = m["name"] + (m["tail"].strip() and f" {m['tail'].strip()}")
                phases.append((tag, begin_ts, ts))
    return phases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--top", type=int, default=8, help="Top N ops per phase (default 8)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df["HOST START TS"] = pd.to_numeric(df["HOST START TS"], errors="coerce")

    signposts = df[df["OP TYPE"] == "signpost"].sort_values("HOST START TS")
    if signposts.empty:
        print("No signposts found in CSV.", file=sys.stderr)
        return 1

    phases = discover_phases(signposts)
    if not phases:
        print("No matched BEGIN/END signpost pairs found.", file=sys.stderr)
        return 1

    ops = df[df["OP TYPE"] != "signpost"].copy()
    for col in ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]", "HOST DURATION [ns]"):
        ops[col] = pd.to_numeric(ops[col], errors="coerce").fillna(0)

    # Aggregate identical-named phases (e.g. FWD across multiple steps).
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "ops": 0,
            "wall_ms": 0.0,
            "dev_k_ms": 0.0,
            "dev_fw_ms": 0.0,
            "host_ms": 0.0,
        }
    )
    op_totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for tag, begin_ts, end_ts in phases:
        # Strip trailing "step=N" so steps roll up under one tag.
        tag_root = re.sub(r"\s+step=\d+$", "", tag.strip()).strip() or tag.strip()
        sub = ops[(ops["HOST START TS"] >= begin_ts) & (ops["HOST START TS"] < end_ts)]
        agg = grouped[tag_root]
        agg["ops"] += len(sub)
        agg["wall_ms"] += (end_ts - begin_ts) / 1e6
        agg["dev_k_ms"] += sub["DEVICE KERNEL DURATION [ns]"].sum() / 1e6
        agg["dev_fw_ms"] += sub["DEVICE FW DURATION [ns]"].sum() / 1e6
        agg["host_ms"] += sub["HOST DURATION [ns]"].sum() / 1e6
        for code, dur in zip(sub["OP CODE"], sub["DEVICE KERNEL DURATION [ns]"]):
            op_totals[tag_root][code].append(dur / 1e6)

    print(f"\nPhases found ({len(phases)} pair(s), rolled up by name):\n")
    print(f"{'Phase':<22} {'Ops':>6} {'Wall':>12} {'Dev kernel':>14} {'Dev FW':>13} {'Host':>13}")
    print("-" * 86)
    totals = {"ops": 0, "wall_ms": 0.0, "dev_k_ms": 0.0, "dev_fw_ms": 0.0, "host_ms": 0.0}
    for tag, agg in grouped.items():
        print(
            f"{tag:<22} {int(agg['ops']):>6} "
            f"{agg['wall_ms']:>9.2f} ms "
            f"{agg['dev_k_ms']:>11.2f} ms "
            f"{agg['dev_fw_ms']:>10.2f} ms "
            f"{agg['host_ms']:>10.2f} ms"
        )
        for k in totals:
            totals[k] += agg[k]
    print("-" * 86)
    print(
        f"{'TOTAL':<22} {int(totals['ops']):>6} "
        f"{totals['wall_ms']:>9.2f} ms "
        f"{totals['dev_k_ms']:>11.2f} ms "
        f"{totals['dev_fw_ms']:>10.2f} ms "
        f"{totals['host_ms']:>10.2f} ms"
    )

    print(f"\n=== Top {args.top} device ops per phase (by device kernel time) ===")
    for tag, agg in grouped.items():
        per_op = op_totals[tag]
        rows = [(code, len(times), sum(times)) for code, times in per_op.items()]
        rows.sort(key=lambda r: r[2], reverse=True)
        total_k = agg["dev_k_ms"] or 1.0
        print(f"\n[{tag}]  kernel_total={agg['dev_k_ms']:.2f} ms")
        print(f"  {'count':>6} {'dev_ms':>10} {'pct':>7}  op")
        for code, count, ms in rows[: args.top]:
            print(f"  {count:>6} {ms:>10.2f} {100 * ms / total_k:>6.1f}%  {code}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
