#!/usr/bin/env python3
"""Per-phase / per-range breakdown of a tt-perf report.

Accepts either:
  - A tracy ops CSV (ops_perf_results_*.csv) — phase breakdown uses HOST START TS
    timestamps and DEVICE KERNEL DURATION [ns].
  - A tt-perf-report TEXT dump (the kind written to a .txt file) — phase / range
    breakdown uses the row IDs and the "Device Time" column (μs).

Usage:
    perf_breakdown.py <path> [--top N] [--ids X-Y]

Modes:
  - Auto-discovery (default): pairs BEGIN/END signposts and aggregates ops per
    phase (FWD, BWD, OPT, ...). Steps roll up by stripping `step=N`.
  - Range mode (--ids X-Y): sums device time for ops whose row ID is in [X, Y]
    (inclusive). Useful with TXT reports to query "what's the total cost of
    ops 3561..4446?".
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

_BEGIN = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9]*)_BEGIN(?P<tail>.*)$")
_END = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9]*)_END(?P<tail>.*)$")

# Ops to drop from totals — pure profiler instrumentation, no real work.
_IGNORED_OPS = {"ProfilerNoopOperation"}

# TXT parsing
# Op row example (note op code may contain spaces, e.g. "MatmulDeviceOperation 1024 x 4096 x 4096"):
#   "  857    0.0 %   FLOP  MatmulDeviceOperation 1024 x 4096 x 4096          0       346 μs      20,120 μs     96  ..."
# Signpost row example:
#   " 3559                  🪧 WARMUP_END after=1"
_TXT_OP_RE = re.compile(r"^\s*(\d+)\s+[\d.]+\s*%\s+(?:FLOP|SLOW)?\s*(.+?)\s+\d+\s+([\d,]+)\s*μs(?:\s+([\d,]+)\s*μs)?")
_TXT_SIGNPOST_RE = re.compile(r"^\s*(\d+)\s+🪧\s+(.+?)\s*$")


# ============================================================
# Phase pairing (shared between CSV and TXT modes)
# ============================================================


def pair_phases(signposts: list[tuple[int, str]]) -> list[tuple[str, int, int]]:
    """Pair BEGIN/END signposts; signposts is [(position, label), ...] ordered.

    Position is whatever scalar we use to delimit ranges — a HOST START TS for
    CSV mode, a row ID for TXT mode. Returns [(tag, begin_pos, end_pos), ...].
    """
    open_begins: dict[tuple[str, str], int] = {}
    phases: list[tuple[str, int, int]] = []
    for pos, label in signposts:
        if (m := _BEGIN.match(label)) is not None:
            open_begins[(m["name"], m["tail"].strip())] = pos
            continue
        if (m := _END.match(label)) is not None:
            key = (m["name"], m["tail"].strip())
            if (begin_pos := open_begins.pop(key, None)) is not None:
                tail = m["tail"].strip()
                tag = m["name"] + (f" {tail}" if tail else "")
                phases.append((tag, begin_pos, pos))
    return phases


def _phase_root(tag: str) -> str:
    """Strip trailing `step=N` so steps roll up under one tag."""
    return re.sub(r"\s+step=\d+$", "", tag.strip()).strip() or tag.strip()


# ============================================================
# CSV mode
# ============================================================

# Device-time columns collapsed by `min` across devices in multi-device mode.
_DEVICE_DURATION_COLS = ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]", "HOST DURATION [ns]")


def _collapse_spmd(ops, pd):
    """Collapse SPMD per-device op rows into one row per logical op.

    In a multi-device SPMD run the same program runs on every device, so each
    op appears once per device. The devices execute the op in lockstep, but a
    device that arrives early stalls waiting on collectives — that wait time is
    folded into its DEVICE KERNEL DURATION, producing huge outliers (e.g. a
    ~170 s "kernel" that is really idle spinning). Taking the *minimum* across
    devices recovers the true compute time of the fastest (non-stalled) device.

    Devices run identical op sequences, so we align ops by their per-device
    execution rank (k-th op on every device is the same logical op) rather than
    GLOBAL CALL COUNT, which is offset per device. Device-time columns and
    HOST START TS are reduced with `min`; OP CODE/OP TYPE take the first value.
    """
    ops = ops.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"]).copy()
    ops["__rank"] = ops.groupby("DEVICE ID").cumcount()

    # Sanity check: a rank should map to one op code across devices. If not, the
    # sequences diverged (not true SPMD) and rank-alignment is unsafe.
    mismatched = ops.groupby("__rank")["OP CODE"].nunique()
    n_bad = int((mismatched > 1).sum())
    if n_bad:
        print(
            f"WARNING: {n_bad} op position(s) have mismatched op codes across devices; "
            "SPMD alignment may be unreliable.",
            file=sys.stderr,
        )

    agg = {"OP CODE": "first", "OP TYPE": "first", "HOST START TS": "min"}
    agg.update({col: "min" for col in _DEVICE_DURATION_COLS})
    return ops.groupby("__rank", as_index=False).agg(agg)


def run_csv(path: str, top: int, per_device: bool) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("CSV mode requires pandas (pip install pandas).", file=sys.stderr)
        return 2

    df = pd.read_csv(path, low_memory=False)
    df["HOST START TS"] = pd.to_numeric(df["HOST START TS"], errors="coerce")

    sps_df = df[df["OP TYPE"] == "signpost"].sort_values("HOST START TS")
    if sps_df.empty:
        print("No signposts found in CSV.", file=sys.stderr)
        return 1

    signposts = [(int(ts), label) for ts, label in zip(sps_df["HOST START TS"], sps_df["OP CODE"])]
    phases = pair_phases(signposts)
    if not phases:
        print("No matched BEGIN/END signpost pairs found.", file=sys.stderr)
        return 1

    ops = df[(df["OP TYPE"] != "signpost") & (~df["OP CODE"].isin(_IGNORED_OPS))].copy()
    for col in _DEVICE_DURATION_COLS:
        ops[col] = pd.to_numeric(ops[col], errors="coerce").fillna(0)

    n_devices = int(ops["DEVICE ID"].nunique())
    if n_devices > 1 and not per_device:
        ops["GLOBAL CALL COUNT"] = pd.to_numeric(ops["GLOBAL CALL COUNT"], errors="coerce")
        ops = _collapse_spmd(ops, pd)
        print(
            f"Multi-device SPMD: {n_devices} devices collapsed to per-op "
            f"minimum device time ({len(ops)} ops). Use --per-device to disable."
        )

    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {"ops": 0, "wall_ms": 0.0, "dev_k_ms": 0.0, "dev_fw_ms": 0.0, "host_ms": 0.0}
    )
    op_totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for tag, begin_ts, end_ts in phases:
        tag_root = _phase_root(tag)
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

    print(f"\n=== Top {top} device ops per phase (by device kernel time) ===")
    for tag, agg in grouped.items():
        per_op = op_totals[tag]
        rows = [(code, len(times), sum(times)) for code, times in per_op.items()]
        rows.sort(key=lambda r: r[2], reverse=True)
        total_k = agg["dev_k_ms"] or 1.0
        print(f"\n[{tag}]  kernel_total={agg['dev_k_ms']:.2f} ms")
        print(f"  {'count':>6} {'dev_ms':>10} {'pct':>7}  op")
        for code, count, ms in rows[:top]:
            print(f"  {count:>6} {ms:>10.2f} {100 * ms / total_k:>6.1f}%  {code}")

    return 0


# ============================================================
# TXT mode
# ============================================================


def parse_text_report(
    path: str,
) -> tuple[list[tuple[int, str, float, float]], list[tuple[int, str]]]:
    """Parse a tt-perf-report TXT dump.

    tt-perf-report repeats the same op rows across several sections (main
    Performance Report, "High Op-to-Op Gap" advice, "Matmul Optimization"
    advice, etc.) — we dedupe by row ID so each op is counted exactly once.

    Returns (ops, signposts) where:
      ops       = [(row_id, op_code, device_us, op_to_op_gap_us), ...]
      signposts = [(row_id, label), ...]
    """
    seen_op_ids: set[int] = set()
    seen_sp_ids: set[int] = set()
    ops: list[tuple[int, str, float, float]] = []
    signposts: list[tuple[int, str]] = []
    with open(path) as f:
        for line in f:
            m = _TXT_SIGNPOST_RE.match(line)
            if m:
                row_id = int(m.group(1))
                if row_id in seen_sp_ids:
                    continue
                seen_sp_ids.add(row_id)
                signposts.append((row_id, m.group(2).strip()))
                continue
            m = _TXT_OP_RE.match(line)
            if m:
                row_id = int(m.group(1))
                if row_id in seen_op_ids:
                    continue
                op_code = m.group(2).strip()
                if op_code in _IGNORED_OPS:
                    seen_op_ids.add(row_id)
                    continue
                dev_us = float(m.group(3).replace(",", ""))
                gap_us = float(m.group(4).replace(",", "")) if m.group(4) else 0.0
                seen_op_ids.add(row_id)
                ops.append((row_id, op_code, dev_us, gap_us))
    return ops, signposts


def _print_topn(tag: str, ops_in_range: list[tuple[int, str, float, float]], top: int) -> None:
    """Top-N device ops by kernel time, matching the CSV-mode block style."""
    dev_total_ms = sum(d for _, _, d, _ in ops_in_range) / 1000.0
    per_op: dict[str, list[float]] = defaultdict(list)
    for _, code, d, _ in ops_in_range:
        per_op[code].append(d / 1000.0)
    rows = [(c, len(v), sum(v)) for c, v in per_op.items()]
    rows.sort(key=lambda r: r[2], reverse=True)
    denom = dev_total_ms or 1.0
    print(f"\n[{tag}]  kernel_total={dev_total_ms:.2f} ms")
    print(f"  {'count':>6} {'dev_ms':>10} {'pct':>7}  op")
    for code, count, ms in rows[:top]:
        print(f"  {count:>6} {ms:>10.2f} {100 * ms / denom:>6.1f}%  {code}")


def run_txt(
    path: str,
    top: int,
    id_range: tuple[int, int] | None,
    phase_filter: str | None,
) -> int:
    ops, signposts = parse_text_report(path)
    if not ops:
        print("No op rows parsed from text report.", file=sys.stderr)
        return 1

    if id_range is not None:
        lo, hi = id_range
        in_range = [r for r in ops if lo <= r[0] <= hi]
        _print_topn(f"IDs {lo}..{hi}", in_range, top)
        return 0

    if not signposts:
        print("No signposts found in TXT — pass --ids X-Y for a range query.", file=sys.stderr)
        return 1

    phases = pair_phases(signposts)
    if not phases:
        print("No matched BEGIN/END signpost pairs found.", file=sys.stderr)
        return 1

    grouped_ids: dict[str, list[tuple[int, str, float, float]]] = defaultdict(list)
    for tag, begin_id, end_id in phases:
        tag_root = _phase_root(tag)
        if phase_filter is not None and tag_root != phase_filter:
            continue
        for r in ops:
            if begin_id < r[0] < end_id:
                grouped_ids[tag_root].append(r)

    if phase_filter is not None and not grouped_ids:
        available = sorted({_phase_root(t) for t, _, _ in phases})
        print(
            f"Phase {phase_filter!r} not found in TXT. Available: {', '.join(available)}",
            file=sys.stderr,
        )
        return 1

    aggs: dict[str, dict[str, float]] = {}
    for tag, rows in grouped_ids.items():
        dev_ms = sum(d for _, _, d, _ in rows) / 1000.0
        gap_ms = sum(g for _, _, _, g in rows) / 1000.0
        aggs[tag] = {
            "ops": len(rows),
            "wall_ms": dev_ms + gap_ms,
            "dev_k_ms": dev_ms,
            "gap_ms": gap_ms,
        }

    header = f"{'Phase':<22} {'Ops':>6} {'Wall':>12} {'Dev kernel':>14} {'Gap':>14}"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)
    totals = {"ops": 0, "wall_ms": 0.0, "dev_k_ms": 0.0, "gap_ms": 0.0}
    for tag, agg in aggs.items():
        print(
            f"{tag:<22} {int(agg['ops']):>6} "
            f"{agg['wall_ms']:>9.2f} ms "
            f"{agg['dev_k_ms']:>11.2f} ms "
            f"{agg['gap_ms']:>11.2f} ms"
        )
        for k in totals:
            totals[k] += agg[k]
    print(sep)
    print(
        f"{'TOTAL':<22} {int(totals['ops']):>6} "
        f"{totals['wall_ms']:>9.2f} ms "
        f"{totals['dev_k_ms']:>11.2f} ms "
        f"{totals['gap_ms']:>11.2f} ms"
    )

    print(f"\n=== Top {top} device ops per phase (by device kernel time) ===")
    for tag, rows in grouped_ids.items():
        _print_topn(tag, rows, top)

    return 0


# ============================================================
# Entry
# ============================================================


def _parse_ids(s: str) -> tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError(f"--ids must look like 'X-Y', got {s!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to ops CSV or tt-perf-report TXT")
    parser.add_argument("--top", type=int, default=8, help="Top N ops per phase/range (default 8)")
    parser.add_argument(
        "--ids",
        type=_parse_ids,
        default=None,
        help="Range query 'X-Y' (TXT mode only): sum device time for ops with ID in [X, Y].",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Restrict TXT output to a single phase (e.g. FWD, BWD, OPT). Phases are auto-discovered from signposts.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "txt"),
        default="auto",
        help="Force input format (default: auto-detect by extension).",
    )
    parser.add_argument(
        "--per-device",
        action="store_true",
        help="CSV mode: keep every device's op row separately instead of collapsing "
        "SPMD ops to the per-op minimum device time across devices (multi-device runs).",
    )
    args = parser.parse_args()

    fmt = args.format
    if fmt == "auto":
        fmt = "csv" if args.path.lower().endswith(".csv") else "txt"

    if fmt == "csv":
        if args.ids is not None or args.phase is not None:
            print("--ids and --phase are only supported in TXT mode.", file=sys.stderr)
            return 2
        return run_csv(args.path, args.top, args.per_device)
    if args.ids is not None and args.phase is not None:
        print("--ids and --phase are mutually exclusive.", file=sys.stderr)
        return 2
    return run_txt(args.path, args.top, args.ids, args.phase)


if __name__ == "__main__":
    sys.exit(main())
