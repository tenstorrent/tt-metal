#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decompose one op_to_op_latency run into BW + latency components for charting.

Emits one chart-ready CSV row (per config = core count x barrier mode) with:
  agg_read_gbps        aggregate read BW across active cores (device-marker span)
  op_to_op_us          done/go dispatch hop (DISP_DONE_OBSERVED -> WORKER_GO_OBSERVED)
  math_to_math_us      last-math(A) -> first-math(B): FINISH_LAST_PUSH(k) -> TILE_IDX0(k+1)
  reader_to_writer_us  NCRISC_DONE -> BRISC_DONE (write-side tail after reads finish)
  write_tail_us        WRITE_BEFORE_BARRIER -> WRITE_AFTER_BARRIER (end barrier/flush)
  *_max                worst-core value (NoC-torus starvation tail)
  reader_done_spread_us / writer_done_spread_us
                       max-min completion time across cores per program (late-core skew)

All latencies are per-core medians over steady-state program transitions unless noted.
Run the test WITHOUT --read-only (writer must do real DRAM writes), with
TT_METAL_DEVICE_PROFILER=1 + TT_METAL_DEVICE_PROFILER_DISPATCH=1 and --use-device-profiler.
See [[goal-reduce-math-to-math-latency]].
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_op_to_op_profiler_csv import (  # noqa: E402
    DEFAULT_LOG,
    READ_RISC_TYPES,
    WRITE_RISC_TYPES,
    load_profiler_csv,
    parse_chip_freq_mhz,
    walk_barrier_markers,
    walk_core_markers,
    walk_done_to_go,
)

DEFAULT_TILE_BYTES = 2048


def walk_spans(df, risc_types, start_zone: str, end_zone: str, min_prog: int):
    """Per (chip, core, prog) start/end cycle span for a marker pair on given RISCs.
    Used for read (NCRISC: READ_BEFORE->READ_LAST) and write (BRISC: WRITE_BEFORE->WRITE_AFTER)."""
    zones = {"PROG_ID", start_zone, end_zone}
    m = df[df["zone name"].isin(zones) & df["RISC processor type"].isin(risc_types)].copy()
    m = m.sort_values(["PCIe slot", "core_x", "core_y", "RISC processor type", "time[cycles since reset]"])
    rows = []
    for (chip, cx, cy, _r), g in m.groupby(["PCIe slot", "core_x", "core_y", "RISC processor type"], sort=False):
        prog = None
        start = None
        for _, row in g.iterrows():
            z = row["zone name"]
            t = int(row["time[cycles since reset]"])
            if z == "PROG_ID":
                prog = int(row["data"])
                start = None
            elif prog is None:
                continue
            elif z == start_zone:
                start = t
            elif z == end_zone and start is not None:
                if prog >= min_prog:
                    rows.append(
                        {
                            "chip": chip,
                            "core_x": cx,
                            "core_y": cy,
                            "prog_id": prog,
                            "start_cycles": start,
                            "end_cycles": t,
                            "duration_cycles": t - start,
                        }
                    )
                start = None
    return pd.DataFrame(rows)


def _series(xs):
    return pd.Series([float(x) for x in xs if x is not None and x == x])


def median(xs):
    s = _series(xs)
    return float(s.median()) if len(s) else float("nan")


def vmax(xs):
    s = _series(xs)
    return float(s.max()) if len(s) else float("nan")


def aggregate_read_gbps(spans: pd.DataFrame, num_cores: int, bytes_per_core: int, freq_mhz: float) -> float:
    """Total bytes / union span (min start -> max end), per execution instance, median.

    Trace capture + replay run each prog id twice ~ms apart; split by start-time gaps so
    the union span is within one execution, not across both.
    """
    if spans.empty:
        return float("nan")
    freq_ghz = freq_mhz / 1000.0
    gap = max(5.0 * float(spans["duration_cycles"].median()), 1.0)
    s = spans.sort_values(["prog_id", "start_cycles"]).reset_index(drop=True)
    aggs = []
    cur = []
    prev_prog = prev_start = None
    for _, r in s.iterrows():
        prog, st = int(r["prog_id"]), int(r["start_cycles"])
        new_inst = (prog != prev_prog) or (prev_start is not None and (st - prev_start) > gap)
        if new_inst and cur:
            aggs.append(cur)
            cur = []
        cur.append(r)
        prev_prog, prev_start = prog, st
    if cur:
        aggs.append(cur)
    gbps = []
    for inst in aggs:
        ncores = len({(int(r["chip"]), int(r["core_x"]), int(r["core_y"])) for r in inst})
        union = max(int(r["end_cycles"]) for r in inst) - min(int(r["start_cycles"]) for r in inst)
        if union > 0:
            gbps.append(ncores * bytes_per_core / (union / freq_ghz))
    return float(pd.Series(gbps).median()) if gbps else float("nan")


def per_core_diff_us(end_d: dict, start_d: dict, freq_mhz: float, min_prog: int):
    """end_d[key] - start_d[key] in us, for keys (chip,cx,cy,prog) present in both, prog>=min."""
    out = []
    for k, e in end_d.items():
        if k[3] < min_prog:
            continue
        if k in start_d:
            out.append((int(e) - int(start_d[k])) / freq_mhz)
    return out


def completion_spread_us(done_d: dict, freq_mhz: float, min_prog: int):
    """Per program: (max - min) completion time across cores -> late-core skew."""
    by_prog: dict = {}
    for (chip, cx, cy, prog), t in done_d.items():
        if prog < min_prog:
            continue
        by_prog.setdefault(prog, []).append(int(t))
    return [(max(v) - min(v)) / freq_mhz for v in by_prog.values() if len(v) > 1]


def global_bubbles_us(pack: dict, unp0: dict, freq_mhz: float, min_prog: int):
    """True inter-op math-to-math: last core to finish pack(k) -> first core to start
    unpack(k+1). The window where NO core does math. Excludes intra-op cross-core skew
    (that is execution time of the gating core, not inter-op latency). Consecutive progs
    only (skips the trace capture->replay boundary)."""
    progs = sorted({k[3] for k in pack if k[3] >= min_prog})
    out = []
    for a, b in zip(progs, progs[1:]):
        if b != a + 1:
            continue
        pk = [v for k, v in pack.items() if k[3] == a]
        un = [v for k, v in unp0.items() if k[3] == b]
        if pk and un:
            bub = (min(un) - max(pk)) / freq_mhz
            if bub > 0:
                out.append(bub)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Decompose one op_to_op run into BW + latency columns.")
    ap.add_argument("--input-file", type=Path, default=None, help=f"default $TT_METAL_HOME/{DEFAULT_LOG}")
    ap.add_argument("--pages-per-core", type=int, required=True)
    ap.add_argument("--tile-bytes", type=int, default=DEFAULT_TILE_BYTES)
    ap.add_argument("--num-cores", type=int, required=True)
    ap.add_argument("--min-prog-id", type=int, default=1)
    ap.add_argument("--csv-out", type=Path, default=None, help="append one row here (writes header if new)")
    ap.add_argument(
        "--per-core-csv",
        type=Path,
        default=None,
        help="write per-core read BW vs position (core_x,core_y,read_gbps,frac_of_max) for plotting starvation",
    )
    ap.add_argument("--label", default="", help="leading k=v;k=v fields to record (e.g. cores=8;mode=full;in_cb=16)")
    args = ap.parse_args()

    log = (args.input_file or Path(os.environ.get("TT_METAL_HOME", ".")) / DEFAULT_LOG).resolve()
    if not log.is_file():
        print(f"Input not found: {log}", file=sys.stderr)
        return 1
    freq = parse_chip_freq_mhz(log)
    df = load_profiler_csv(log)

    pack_finish, unpack0, _ = walk_core_markers(df)
    barriers = walk_barrier_markers(df)
    dg = walk_done_to_go(df, freq, args.min_prog_id)
    spans = walk_spans(df, READ_RISC_TYPES, "READ_BEFORE_BARRIER", "READ_LAST_BARRIER", args.min_prog_id)
    wspans = walk_spans(df, WRITE_RISC_TYPES, "WRITE_BEFORE_BARRIER", "WRITE_AFTER_BARRIER", args.min_prog_id)

    bytes_per_core = args.pages_per_core * args.tile_bytes
    freq_ghz = freq / 1000.0
    agg_bw = aggregate_read_gbps(spans, args.num_cores, bytes_per_core, freq)
    agg_wbw = aggregate_read_gbps(wspans, args.num_cores, bytes_per_core, freq)
    agg_total = (agg_bw + agg_wbw) if (agg_bw == agg_bw and agg_wbw == agg_wbw) else float("nan")

    # Per-core read-BW distribution -> quantifies reader starvation/serialization.
    pcbw: dict = {}
    if not spans.empty:
        for _, r in spans.iterrows():
            k = (int(r["chip"]), int(r["core_x"]), int(r["core_y"]))
            dur = int(r["duration_cycles"])
            if dur > 0:
                pcbw.setdefault(k, []).append(bytes_per_core / (dur / freq_ghz))
    per_core_bw = [float(pd.Series(v).median()) for v in pcbw.values()]
    pc_min = min(per_core_bw) if per_core_bw else float("nan")
    pc_max = max(per_core_bw) if per_core_bw else float("nan")
    # starvation_ratio = fastest-core BW / slowest-core BW (1.0 = perfectly fair, >1 worse)
    starv = (pc_max / pc_min) if (per_core_bw and pc_min > 0) else float("nan")

    if args.per_core_csv is not None:
        args.per_core_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(
            ((cx, cy, float(pd.Series(v).median())) for (chip, cx, cy), v in pcbw.items()),
            key=lambda r: (r[1], r[0]),  # by row (y) then column (x)
        )
        with args.per_core_csv.open("w") as fh:
            fh.write("core_x,core_y,read_gbps,frac_of_max\n")
            for cx, cy, g in rows:
                fh.write(f"{cx},{cy},{g:.3f},{(g / pc_max if pc_max else float('nan')):.3f}\n")
        print(f"wrote per-core BW-vs-position to {args.per_core_csv} ({len(rows)} cores)")

    # math-to-math = global inter-op bubble (true latency); NOT the per-core gap (which
    # counts fast cores idling at done/go = op execution skew, reported separately).
    m2m = global_bubbles_us(pack_finish, unpack0, freq, args.min_prog_id)
    # read-fill head: first read issued (READ_BEFORE_BARRIER) -> compute gets tile 0
    # (TILE_IDX[0]). This is the first per-trid BATCH (N=trid_in_flight reads) landing +
    # barrier + push, i.e. the reader latency that defers math start (grows with N, not CB depth).
    read_fill = per_core_diff_us(unpack0, barriers["read_before"], freq, args.min_prog_id)
    r2w = per_core_diff_us(barriers["brisc_done"], barriers["ncrisc_done"], freq, args.min_prog_id)
    wtail = per_core_diff_us(barriers["write_after"], barriers["write_before"], freq, args.min_prog_id)
    rd_spread = completion_spread_us(barriers["ncrisc_done"], freq, args.min_prog_id)
    wr_spread = completion_spread_us(barriers["brisc_done"], freq, args.min_prog_id)
    o2o_med = float(dg["dg_median_ns"].median()) / 1000.0 if not dg.empty else float("nan")
    o2o_last = float(dg["dg_last_ns"].median()) / 1000.0 if not dg.empty else float("nan")
    # Per-core op duration (reader start NCRISC_GO -> writer done BRISC_DONE) to normalize
    # the spread: absolute spread (us) scales with per-core work, so report it as a % of
    # the typical core's op time for comparability across work sizes / NOP counts.
    op_durs = per_core_diff_us(barriers["brisc_done"], barriers["ncrisc_go"], freq, args.min_prog_id)
    op_dur_med = median(op_durs)
    wr_spread_med = median(wr_spread)
    wr_spread_pct = (
        (wr_spread_med / op_dur_med * 100.0) if (op_dur_med == op_dur_med and op_dur_med > 0) else float("nan")
    )

    row = {
        "agg_total_gbps": round(agg_total, 3),
        "agg_read_gbps": round(agg_bw, 3),
        "agg_write_gbps": round(agg_wbw, 3),
        "per_core_gbps_min": round(pc_min, 3),
        "per_core_gbps_median": round(median(per_core_bw), 3),
        "per_core_gbps_max": round(pc_max, 3),
        "starvation_ratio": round(starv, 3),
        "op_to_op_us": round(o2o_med, 4),
        "op_to_op_last_us": round(o2o_last, 4),
        "math_to_math_us": round(median(m2m), 4),
        "math_to_math_max_us": round(vmax(m2m), 4),
        "read_fill_head_us": round(median(read_fill), 4),
        "reader_to_writer_us": round(median(r2w), 4),
        "reader_to_writer_max_us": round(vmax(r2w), 4),
        "write_tail_us": round(median(wtail), 4),
        "write_tail_max_us": round(vmax(wtail), 4),
        # op execution skew (NoC starvation), distinct from inter-op latency above:
        "reader_done_spread_us": round(median(rd_spread), 4),
        "writer_done_spread_us": round(median(wr_spread), 4),
        "op_duration_us": round(op_dur_med, 3),
        # normalized spread: writer done-spread as a % of the per-core op duration (work-size
        # independent, so comparable across core counts and NOP counts).
        "writer_spread_pct": round(wr_spread_pct, 2),
    }

    label = {}
    for kv in args.label.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            label[k.strip()] = v.strip()

    print(f"freq={freq:.1f}MHz  cores={args.num_cores}  bytes/core={bytes_per_core}")
    for k, v in row.items():
        print(f"  {k:26s} {v}")

    # Columns kept in stdout (driver greps agg_read/agg_total during tuning) but dropped
    # from the chart CSV as noise -- per-core BW detail and the read/write BW split;
    # agg_total + starvation_ratio (+ peak_read_gbps from the label) carry the signal.
    csv_drop = {"agg_read_gbps", "agg_write_gbps", "per_core_gbps_min", "per_core_gbps_median", "per_core_gbps_max"}
    if args.csv_out is not None:
        cols = [c for c in (list(label.keys()) + list(row.keys())) if c not in csv_drop]
        new = not args.csv_out.exists()
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("a") as fh:
            if new:
                fh.write(",".join(cols) + "\n")
            fh.write(",".join(str(label.get(c, row.get(c, ""))) for c in cols) + "\n")
        print(f"appended row to {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
