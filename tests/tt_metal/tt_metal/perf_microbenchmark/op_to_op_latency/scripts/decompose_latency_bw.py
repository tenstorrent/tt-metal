#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decompose one op_to_op_latency run into BW + latency components for charting.

Emits one chart-ready CSV row (per config = core count x barrier mode) with:
  agg_read_gbps        aggregate read BW across active cores (device-marker span)
  official_op2op_us    op2op via official tools/tracy methodology (standard KERNEL zones):
                       {risc}-KERNEL ZONE_END(k) -> DM-KERNEL ZONE_START(k+1); '_min' == tracy Min
  device_kernel_dur_us first KERNEL start -> last KERNEL end across riscs (op kernel span)
  m2m_*                math-to-math brackets: bubble / skew_env / within / finish+start skew
  read_fill_head_us    READ_BEFORE_BARRIER -> TILE_IDX[0] (reader latency before first math)
  reader_to_writer_us  NCRISC_DONE -> BRISC_DONE (write-side tail after reads finish)
  write_tail_us        WRITE_BEFORE_BARRIER -> WRITE_AFTER_BARRIER (end barrier/flush)
  *_max                worst-core value (NoC-torus starvation tail)
  reader_done_spread_us / writer_done_spread_us
                       max-min completion time across cores per program (late-core skew)

All metrics derive from standard profiler zones + test-side kernel markers only -- no custom
FW/dispatch markers (those were reverted out of tt_metal/). For absolute (unpolluted) op2op use
the realtime profiler (--use-realtime-profiler -> profile_log_device_rt.csv gap_to_next_go).
All latencies are per-core medians over steady-state program transitions unless noted.
Run the test WITHOUT --read-only (writer must do real DRAM writes), with
TT_METAL_DEVICE_PROFILER=1 and --use-device-profiler. See [[goal-reduce-math-to-math-latency]].
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


KERNEL_ZONES = ("BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL")
DM_KERNEL_ZONES = ("BRISC-KERNEL", "NCRISC-KERNEL")


def official_kernel_metrics(df, freq_mhz: float, max_gap_us: float = 50.0):
    """Reproduce the tools/tracy methodology (device_post_proc_config.py) from the standard
    auto-emitted KERNEL zones -- no custom markers, comparable to process_device_log.py / Tracy.

      op2op              : per core, adjacent ops -> first DM-KERNEL start(k+1) - last KERNEL end(k)
                           (device_post_proc_config 'op2op'). Steady-state value; the max_gap_us cap
                           drops trace-instance boundaries that inflate the tool's Average/Max.
      device_kernel_dur  : per op, last KERNEL end - first KERNEL start ON A SINGLE CORE, then
                           median across cores (device_post_proc_config 'device_kernel_duration').
                           This is a PER-CORE number, NOT the run's go->done -- see envelope below.
      envelope           : per op, last-core KERNEL end - first-core KERNEL start ACROSS ALL cores
                           (aligned by op index). This IS the whole-op go->done wall-clock; it is
                           set by the SLOWEST core, so it exceeds the per-core median duration by the
                           straggler tail. (8c: env ~= per-core dur; 56c: env >> per-core dur because
                           cores finish spread over tens of us -- half the grid idles on stragglers.)

    Segments ops by the start->end->start transition per core (no PROG_ID needed, so robust to
    trace capture/replay duplicate program ids). Returns (op2op_us list, kernel_dur_us list,
    env dict with envelope_us / kdur_fast_us / kdur_slow_us / end_spread_us / start_spread_us,
    each a median over op indices)."""
    z = df[df["zone name"].isin(KERNEL_ZONES) & df["type"].isin(["ZONE_START", "ZONE_END"])]
    op2op, kdur = [], []
    ops_by_core: dict = {}  # (slot,x,y) -> list of fully-formed ops, aligned by index across cores
    cap = max_gap_us * freq_mhz
    for _key, g in z.groupby(["PCIe slot", "core_x", "core_y"], sort=False):
        evs = sorted(
            (
                int(r["time[cycles since reset]"]),
                r["type"] == "ZONE_START",
                r["zone name"] in DM_KERNEL_ZONES,
            )
            for _, r in g.iterrows()
        )
        ops = []  # each: {start, dm, end}
        cur = None
        prev_was_end = False
        for t, is_start, is_dm in evs:
            if is_start:
                if cur is None or prev_was_end:  # first start after a run of ends = new op
                    if cur is not None:
                        ops.append(cur)
                    cur = {"start": t, "dm": t if is_dm else None, "end": None}
                elif is_dm and cur["dm"] is None:
                    cur["dm"] = t
                prev_was_end = False
            else:
                if cur is not None:
                    cur["end"] = t
                prev_was_end = True
        if cur is not None:
            ops.append(cur)
        done = [o for o in ops if o["end"] is not None]
        ops_by_core[_key] = done
        for o in done:
            kdur.append((o["end"] - o["start"]) / freq_mhz)
        for a, b in zip(ops, ops[1:]):
            if a["end"] is not None and b["dm"] is not None:
                gap = b["dm"] - a["end"]
                if 0 < gap < cap:
                    op2op.append(gap / freq_mhz)

    # Whole-op envelope: align ops by index across cores (all cores run the same op sequence, so op
    # index i is the same program instance). Per op index: envelope = max(end) - min(start) across
    # cores (= go->done for the run), plus fastest/slowest single-core duration and the end/start
    # spreads. Report the median over op indices, matching device_kernel_dur's median-over-ops.
    env = {k: float("nan") for k in ("envelope_us", "kdur_fast_us", "kdur_slow_us", "end_spread_us", "start_spread_us")}
    if ops_by_core:
        n = min(len(v) for v in ops_by_core.values())
        envs, ends_sp, starts_sp, dmins, dmaxs = [], [], [], [], []
        for i in range(n):
            starts = [ops_by_core[c][i]["start"] for c in ops_by_core]
            ends = [ops_by_core[c][i]["end"] for c in ops_by_core]
            durs = [(e - s) / freq_mhz for s, e in zip(starts, ends)]
            envs.append((max(ends) - min(starts)) / freq_mhz)
            ends_sp.append((max(ends) - min(ends)) / freq_mhz)
            starts_sp.append((max(starts) - min(starts)) / freq_mhz)
            dmins.append(min(durs))
            dmaxs.append(max(durs))
        if envs:
            env = {
                "envelope_us": float(pd.Series(envs).median()),
                "kdur_fast_us": float(pd.Series(dmins).median()),
                "kdur_slow_us": float(pd.Series(dmaxs).median()),
                "end_spread_us": float(pd.Series(ends_sp).median()),
                "start_spread_us": float(pd.Series(starts_sp).median()),
            }
    return op2op, kdur, env


def m2m_brackets(pack: dict, unp0: dict, freq_mhz: float, min_prog: int):
    """Decompose the math-to-math interval per consecutive op pair (k -> k+1).

    Four cross-core fenceposts: F=pack_finish(k) (last-math), S=unpack0(k+1) (first-math).
      F_min/F_max = first/last core to FINISH op k     S_min/S_max = first/last core to START op k+1
    Returns per-pair lists (us):
      bubble     = S_min - F_max   global idle window: NO core doing math (raw, may be <0 if ops overlap)
      skew_env   = S_max - F_min   first core to finish -> last core to start (full cross-core envelope)
      finish_skew= F_max - F_min   spread of op-k completion across cores
      start_skew = S_max - S_min   spread of op-(k+1) start across cores
    Identity: skew_env = finish_skew + bubble + start_skew.
    Plus within-core gap (same core's unpack0(k+1) - pack_finish(k)) across all core/pair samples.
    Consecutive progs only (skips the trace capture->replay boundary)."""
    progs = sorted({k[3] for k in pack if k[3] >= min_prog})
    bubble, skew_env, finish_skew, start_skew, within = [], [], [], [], []
    for a, b in zip(progs, progs[1:]):
        if b != a + 1:
            continue
        fin = {k[:3]: v for k, v in pack.items() if k[3] == a}
        sta = {k[:3]: v for k, v in unp0.items() if k[3] == b}
        if not fin or not sta:
            continue
        F, S = list(fin.values()), list(sta.values())
        bubble.append((min(S) - max(F)) / freq_mhz)
        skew_env.append((max(S) - min(F)) / freq_mhz)
        finish_skew.append((max(F) - min(F)) / freq_mhz)
        start_skew.append((max(S) - min(S)) / freq_mhz)
        for c in fin.keys() & sta.keys():
            within.append((sta[c] - fin[c]) / freq_mhz)
    return {
        "bubble": bubble,
        "skew_env": skew_env,
        "finish_skew": finish_skew,
        "start_skew": start_skew,
        "within": within,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Decompose one op_to_op run into BW + latency columns.")
    ap.add_argument("--input-file", type=Path, default=None, help=f"default $TT_METAL_HOME/{DEFAULT_LOG}")
    ap.add_argument("--pages-per-core", type=int, required=True)
    ap.add_argument("--tile-bytes", type=int, default=DEFAULT_TILE_BYTES)
    ap.add_argument("--num-cores", type=int, required=True)
    ap.add_argument(
        "--directions",
        type=int,
        default=0,
        help="bytes multiplier for overall_bw_gbps: 1=read-only, 2=read+write, 0=auto (2 if writes present)",
    )
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

    # math-to-math brackets: global bubble + cross-core skew envelope + within-core gap.
    # See m2m_brackets(): skew_env = finish_skew + bubble + start_skew.
    mb = m2m_brackets(pack_finish, unpack0, freq, args.min_prog_id)
    # read-fill head: first read issued (READ_BEFORE_BARRIER) -> compute gets tile 0
    # (TILE_IDX[0]). This is the first per-trid BATCH (N=trid_in_flight reads) landing +
    # barrier + push, i.e. the reader latency that defers math start (grows with N, not CB depth).
    read_fill = per_core_diff_us(unpack0, barriers["read_before"], freq, args.min_prog_id)
    r2w = per_core_diff_us(barriers["brisc_done"], barriers["ncrisc_done"], freq, args.min_prog_id)
    wtail = per_core_diff_us(barriers["write_after"], barriers["write_before"], freq, args.min_prog_id)
    # write_drain = LAST write issued -> write barrier complete (WRITE_LAST_ISSUED -> WRITE_AFTER_BARRIER).
    # This is the un-hideable write latency paid at op end: the final flush+barrier draining the
    # writes still outstanding when the loop exits. With per-CB flushing ~(cb_page_slots-1) writes
    # are in flight at the end, so it rises ~linearly with output CB depth. Distinct from write_tail
    # (WRITE_BEFORE->WRITE_AFTER = whole write phase ~= bytes/BW, ~flat in CB depth).
    wdrain = per_core_diff_us(barriers["write_after"], barriers["write_last_issued"], freq, args.min_prog_id)
    rd_spread = completion_spread_us(barriers["ncrisc_done"], freq, args.min_prog_id)
    wr_spread = completion_spread_us(barriers["brisc_done"], freq, args.min_prog_id)
    # Op2op + kernel duration via the OFFICIAL tools/tracy methodology from the standard
    # auto-emitted KERNEL zones (comparable to process_device_log.py / Tracy). No custom FW or
    # dispatch markers -- those were reverted out of tt_metal/; the realtime profiler
    # (--use-realtime-profiler -> profile_log_device_rt.csv) gives the unpolluted absolute op2op.
    off_op2op, off_kdur, off_env = official_kernel_metrics(df, freq, max_gap_us=50.0)

    # OVERALL BW = total bytes moved / whole-op KERNEL ENVELOPE (first-core start -> last-core end).
    # This is the honest end-to-end throughput INCLUDING the straggler tail -- unlike agg_*_gbps, which
    # divide each direction's bytes by that direction's union span (a steady/peak rate that ignores the
    # ragged completion tail). At high core counts overall_bw < agg_total because the envelope carries
    # the write-drain tail. directions: 1=read-only, 2=read+write; 0=auto (2 iff write markers present).
    dirn = args.directions if args.directions else (2 if (agg_wbw == agg_wbw) else 1)
    env_us = off_env["envelope_us"]
    overall_bw_gbps = (
        dirn * args.num_cores * bytes_per_core / (env_us * 1e3) if (env_us == env_us and env_us > 0) else float("nan")
    )
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
        "overall_bw_gbps": round(overall_bw_gbps, 3),  # total bytes / kernel envelope (incl. straggler tail)
        "agg_total_gbps": round(agg_total, 3),
        "agg_read_gbps": round(agg_bw, 3),
        "agg_write_gbps": round(agg_wbw, 3),
        "per_core_gbps_min": round(pc_min, 3),
        "per_core_gbps_median": round(median(per_core_bw), 3),
        "per_core_gbps_max": round(pc_max, 3),
        "starvation_ratio": round(starv, 3),
        # OFFICIAL tools/tracy methodology (standard KERNEL zones; Tracy / process_device_log.py
        # comparable) -- canonical op2op + kernel duration, no custom markers:
        "official_op2op_us": round(median(off_op2op), 4),  # steady-state median
        "official_op2op_min_us": round(min(off_op2op) if off_op2op else float("nan"), 4),  # == tracy 'Min'
        "device_kernel_dur_us": round(median(off_kdur), 4),  # PER-CORE median (one core's busy time)
        # Whole-op ENVELOPE = the run's go->done (first-core start -> last-core end, across all
        # cores). Set by the slowest core, so it exceeds device_kernel_dur_us by the straggler tail.
        # This -- not device_kernel_dur_us -- is what the timeline chart's op span reflects.
        "kernel_envelope_us": round(off_env["envelope_us"], 4),
        "kernel_dur_fast_us": round(off_env["kdur_fast_us"], 4),  # fastest single core (finishes early, idles)
        "kernel_dur_slow_us": round(off_env["kdur_slow_us"], 4),  # slowest single core (sets the envelope)
        "kernel_end_spread_us": round(off_env["end_spread_us"], 4),  # last-core end - first-core end (straggler tail)
        # math-to-math, three brackets (all per-pair medians):
        "m2m_bubble_us": round(median(mb["bubble"]), 4),  # global idle window (== old math_to_math_us)
        "m2m_skew_env_us": round(median(mb["skew_env"]), 4),  # first-finish -> last-start (cross-core)
        "m2m_within_med_us": round(median(mb["within"]), 4),  # same-core gap, typical
        "m2m_within_max_us": round(vmax(mb["within"]), 4),  # same-core gap, worst core
        "finish_skew_us": round(median(mb["finish_skew"]), 4),  # op-k completion spread
        "start_skew_us": round(median(mb["start_skew"]), 4),  # op-(k+1) start spread
        "read_fill_head_us": round(median(read_fill), 4),
        "reader_to_writer_us": round(median(r2w), 4),
        "reader_to_writer_max_us": round(vmax(r2w), 4),
        "write_tail_us": round(median(wtail), 4),
        "write_tail_max_us": round(vmax(wtail), 4),
        "write_drain_us": round(median(wdrain), 4),  # last write issued -> barrier done (the real write tail)
        "write_drain_max_us": round(vmax(wdrain), 4),
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
