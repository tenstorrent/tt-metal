# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-process tracy profile_log_device.csv for TEN-4679.

All percentages reported are means across active SDPA cores (cores where
FPU_COUNTER value > 0), each value normalised by that core's ref_cnt.

Counters used (all already exposed in tt_metal/tools/profiler/perf_counters.hpp):

  Q1 (FPU% / SFPU%):
    FPU_COUNTER, SFPU_COUNTER, MATH_COUNTER

  Q2 (FPU stalled by dest / vice-versa):
    Direct (small on BH for SDPA because kernel uses tile_regs_*, not STALLWAIT):
      WAITING_FOR_SFPU_IDLE_1, WAITING_FOR_MATH_IDLE_1
    Derived from req-vs-grant counter pairs (these capture the implicit
    tile_regs_* dest contention that the WAIT counters miss):
      scoreboard_stall  = MATH_INSTRN_AVAILABLE - AVAILABLE_MATH
      dest_wr_port_stall = MATH_INSTRN_AVAILABLE - MATH_NOT_STALLED_DEST_WR_PORT
      fidelity_stall    = MATH_FIDELITY_STALL  (direct)
      d2a_hazard_stall  = MATH_INSTRN_AVAILABLE - DATA_HAZARD_STALLS_MOVD2A
        (the enum name is misleading - sel 1 actually counts cycles NOT stalled by D2A)
      issue_eff         = MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE

  Q4 (NoC/L1 idle):
    Math-thread perspective: WAITING_FOR_NONZERO_SEM_1
    L1 bandwidth perspective (req minus grant per port, from l1_0 group):
      L1_0_NOC_RING0_INCOMING_{0,1}, L1_0_NOC_RING0_OUTGOING_{0,1},
      L1_0_UNPACKER_0, L1_0_TDMA_BUNDLE_{0,1}

Full-32-bit ref_cnt recovery:
  perf_counters.hpp PerfCounter packs ref_cnt in 24 bits but the BH hardware
  counter is 32. If the kernel runs longer than ~12 ms at 1.35 GHz the
  truncated value wraps. We look for paired profiler_id=9091 records (emitted
  alongside each counter readout in read_single_group on the perf_counters
  branch) and use those to recover the full 32-bit ref_cnt by matching low 24
  bits. Falls back to the truncated 24-bit value if no 9091 record is present
  (graceful degradation, useful for kernels under 12 ms).

Usage:
    python analysis/post_process.py /path/to/profile_log_device.csv [--seq-lens 1024 ...]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PERF_COUNTER_PROFILER_ID = 9090
PERF_COUNTER_REF_CNT_FULL_PROFILER_ID = 9091


def parse_meta(s: str) -> dict:
    s = s.strip()
    if not s.startswith("{"):
        return {}
    try:
        return json.loads(s.replace(";", ","))
    except json.JSONDecodeError:
        return {}


def load_counters(path: Path) -> pd.DataFrame:
    """Load counter records; recover full 32-bit ref_cnt from 9091 aux records if present."""
    full_ref_by_key: Dict = {}
    raw_rows = []

    with path.open() as f:
        f.readline()
        header = f.readline().strip().split(", ")
        col_x = header.index("core_x")
        col_y = header.index("core_y")
        col_risc = header.index("RISC processor type")
        col_timer = header.index("timer_id")
        col_data = header.index("data")
        col_runid = header.index("run host ID")
        col_meta = header.index("meta data")

        for line in f:
            parts = line.rstrip("\n").split(",", len(header) - 1)
            if len(parts) <= col_meta:
                continue
            try:
                tid = int(parts[col_timer])
            except ValueError:
                continue
            try:
                run_id = int(parts[col_runid]) if parts[col_runid] else -1
            except ValueError:
                run_id = -1

            if tid == PERF_COUNTER_REF_CNT_FULL_PROFILER_ID:
                try:
                    raw = int(parts[col_data])
                except ValueError:
                    continue
                ref_cnt_full = raw & 0xFFFFFFFF
                ctype_int = (raw >> 32) & 0xFF
                key = (run_id, int(parts[col_x]), int(parts[col_y]), parts[col_risc], ctype_int)
                full_ref_by_key[key] = ref_cnt_full
            elif tid == PERF_COUNTER_PROFILER_ID:
                meta = parse_meta(parts[col_meta])
                counter = meta.get("counter type")
                if not counter:
                    continue
                try:
                    value = float(meta.get("value", 0))
                    ref = float(meta.get("ref cnt", 0))
                except (TypeError, ValueError):
                    continue
                raw_rows.append(
                    {
                        "run_id": run_id,
                        "core_x": int(parts[col_x]),
                        "core_y": int(parts[col_y]),
                        "risc": parts[col_risc],
                        "counter": counter,
                        "value": value,
                        "ref_cnt_24bit": ref,
                    }
                )

    rows = []
    for r in raw_rows:
        matched_full = None
        target = int(r["ref_cnt_24bit"]) & 0xFFFFFF
        for key, full in full_ref_by_key.items():
            kr, kx, ky, krisc, _ctype = key
            if kr == r["run_id"] and kx == r["core_x"] and ky == r["core_y"] and krisc == r["risc"]:
                if (full & 0xFFFFFF) == target:
                    matched_full = full
                    break
        rows.append(
            {
                "run_id": r["run_id"],
                "core_x": r["core_x"],
                "core_y": r["core_y"],
                "risc": r["risc"],
                "counter": r["counter"],
                "value": r["value"],
                "ref_cnt": float(matched_full) if matched_full is not None else r["ref_cnt_24bit"],
                "ref_cnt_24bit": r["ref_cnt_24bit"],
                "ref_cnt_full_recovered": matched_full is not None,
            }
        )
    return pd.DataFrame(rows)


def util_pct(df, counter: str) -> float:
    """% of ref_cnt (mean across cores). NaN if counter absent."""
    sub = df[df["counter"] == counter]
    if sub.empty:
        return float("nan")
    u = (sub["value"] / sub["ref_cnt"].replace(0, pd.NA)) * 100
    u = u.dropna()
    return float(u.mean()) if len(u) else float("nan")


def summarise_one(df_op: pd.DataFrame, label: str) -> Dict[str, float]:
    print()
    print("=" * 90)
    print(f"  {label}")
    print("=" * 90)

    fpu_active_cores = set(
        zip(
            df_op[(df_op["counter"] == "FPU_COUNTER") & (df_op["value"] > 0)]["core_x"],
            df_op[(df_op["counter"] == "FPU_COUNTER") & (df_op["value"] > 0)]["core_y"],
        )
    )
    print(f"Active cores: {len(fpu_active_cores)}")
    df = df_op[df_op.apply(lambda r: (r["core_x"], r["core_y"]) in fpu_active_cores, axis=1)]

    # Q1
    fpu = util_pct(df, "FPU_COUNTER")
    sfpu = util_pct(df, "SFPU_COUNTER")
    math = util_pct(df, "MATH_COUNTER")

    # Q2 direct
    wait_sfpu_t1 = util_pct(df, "WAITING_FOR_SFPU_IDLE_1")
    wait_math_t1 = util_pct(df, "WAITING_FOR_MATH_IDLE_1")

    # Q2 derived
    math_avail = util_pct(df, "MATH_INSTRN_AVAILABLE")
    avail_math = util_pct(df, "AVAILABLE_MATH")
    not_stalled_wr_port = util_pct(df, "MATH_NOT_STALLED_DEST_WR_PORT")
    not_stalled_d2a = util_pct(df, "DATA_HAZARD_STALLS_MOVD2A")
    src_data_ready = util_pct(df, "MATH_SRC_DATA_READY")
    fidelity_stall = util_pct(df, "MATH_FIDELITY_STALL")
    math_started = util_pct(df, "MATH_INSTRN_STARTED")

    def safe_diff(a, b):
        if pd.isna(a) or pd.isna(b):
            return float("nan")
        return max(0.0, a - b)

    scoreboard_stall = safe_diff(math_avail, avail_math)
    wr_port_stall = safe_diff(math_avail, not_stalled_wr_port)
    d2a_stall = safe_diff(math_avail, not_stalled_d2a)
    src_data_stall = safe_diff(math_avail, src_data_ready)

    # Q4 math-thread perspective
    sem_t1 = util_pct(df, "WAITING_FOR_NONZERO_SEM_1")
    sem_t2 = util_pct(df, "WAITING_FOR_NONZERO_SEM_2")

    # Q4 L1 bandwidth perspective (req vs grant, only if l1_0 group was enabled)
    noc_in0_req = util_pct(df, "L1_0_NOC_RING0_INCOMING_0")
    noc_in0_grant = util_pct(df, "L1_0_NOC_RING0_INCOMING_0_GRANT")
    noc_in1_req = util_pct(df, "L1_0_NOC_RING0_INCOMING_1")
    noc_in1_grant = util_pct(df, "L1_0_NOC_RING0_INCOMING_1_GRANT")
    noc_out0_req = util_pct(df, "L1_0_NOC_RING0_OUTGOING_0")
    noc_out0_grant = util_pct(df, "L1_0_NOC_RING0_OUTGOING_0_GRANT")
    noc_out1_req = util_pct(df, "L1_0_NOC_RING0_OUTGOING_1")
    noc_out1_grant = util_pct(df, "L1_0_NOC_RING0_OUTGOING_1_GRANT")
    unp0_req = util_pct(df, "L1_0_UNPACKER_0")
    unp0_grant = util_pct(df, "L1_0_UNPACKER_0_GRANT")
    tdma0_req = util_pct(df, "L1_0_TDMA_BUNDLE_0_RISC")
    tdma0_grant = util_pct(df, "L1_0_TDMA_BUNDLE_0_GRANT")
    tdma1_req = util_pct(df, "L1_0_TDMA_BUNDLE_1_TRISC")
    tdma1_grant = util_pct(df, "L1_0_TDMA_BUNDLE_1_GRANT")

    print(f"  Q1 FPU active                            : {fpu:6.2f} %")
    print(f"  Q1 SFPU active                           : {sfpu:6.2f} %")
    print(f"  Q1 MATH active (= FPU + SFPU)            : {math:6.2f} %")
    print(f"     math-idle residual (100 - MATH)       : {max(0, 100 - math):6.2f} %")
    print()
    print(f"  Q2 direct WAIT counters (small on BH):")
    print(f"     WAIT_SFPU on math thread (T1)         : {wait_sfpu_t1:6.2f} %")
    print(f"     WAIT_MATH on math thread (T1)         : {wait_math_t1:6.2f} %")
    print()
    print(f"  Q2 derived dest contention (the real numbers):")
    print(f"     MATH_INSTRN_AVAILABLE                  : {math_avail:6.2f} %  (math op queued)")
    print(f"     MATH_INSTRN_STARTED                    : {math_started:6.2f} %  (math op issued)")
    print(
        f"     issue efficiency (started/available)   : {(math_started/math_avail*100) if math_avail else float('nan'):6.2f} %"
    )
    print(f"     scoreboard stall (avail - AVAILABLE_M) : {scoreboard_stall:6.2f} %  *** dest dependency ***")
    print(f"     dest write-port stall                  : {wr_port_stall:6.2f} %")
    print(f"     fidelity stall (HiFi2 phase ongoing)   : {fidelity_stall:6.2f} %")
    print(f"     D2A hazard stall                       : {d2a_stall:6.2f} %")
    print(f"     src-data-not-ready stall (ALU only)    : {src_data_stall:6.2f} %")
    print()
    print(f"  Q4 NoC/L1 - math-thread perspective (cb_wait_front):")
    print(f"     T1 (math) sem wait                     : {sem_t1:6.2f} %")
    print(f"     T2 (pack) sem wait                     : {sem_t2:6.2f} %  (pack waits for math output)")
    print(f"  Q4 NoC/L1 - bandwidth perspective (L1 bank 0, req vs grant):")
    print(
        f"     NOC R0 in0    req={noc_in0_req:6.2f}%  grant={noc_in0_grant:6.2f}%  stall={max(0,(noc_in0_req or 0)-(noc_in0_grant or 0)):6.2f}%"
    )
    print(
        f"     NOC R0 in1    req={noc_in1_req:6.2f}%  grant={noc_in1_grant:6.2f}%  stall={max(0,(noc_in1_req or 0)-(noc_in1_grant or 0)):6.2f}%"
    )
    print(
        f"     NOC R0 out0   req={noc_out0_req:6.2f}%  grant={noc_out0_grant:6.2f}%  stall={max(0,(noc_out0_req or 0)-(noc_out0_grant or 0)):6.2f}%"
    )
    print(
        f"     NOC R0 out1   req={noc_out1_req:6.2f}%  grant={noc_out1_grant:6.2f}%  stall={max(0,(noc_out1_req or 0)-(noc_out1_grant or 0)):6.2f}%"
    )
    print(
        f"     Unpacker 0    req={unp0_req:6.2f}%  grant={unp0_grant:6.2f}%  stall={max(0,(unp0_req or 0)-(unp0_grant or 0)):6.2f}%"
    )
    print(
        f"     TDMA bundle 0 req={tdma0_req:6.2f}%  grant={tdma0_grant:6.2f}%  stall={max(0,(tdma0_req or 0)-(tdma0_grant or 0)):6.2f}%"
    )
    print(
        f"     TDMA bundle 1 req={tdma1_req:6.2f}%  grant={tdma1_grant:6.2f}%  stall={max(0,(tdma1_req or 0)-(tdma1_grant or 0)):6.2f}%"
    )

    return {
        "label": label,
        "active_cores": len(fpu_active_cores),
        "fpu_pct": fpu,
        "sfpu_pct": sfpu,
        "math_pct": math,
        "math_idle_residual_pct": max(0.0, 100.0 - math),
        "wait_sfpu_t1": wait_sfpu_t1,
        "wait_math_t1": wait_math_t1,
        "math_avail": math_avail,
        "math_started": math_started,
        "issue_eff_pct": (math_started / math_avail * 100) if math_avail else float("nan"),
        "scoreboard_stall_pct": scoreboard_stall,
        "wr_port_stall_pct": wr_port_stall,
        "fidelity_stall_pct": fidelity_stall,
        "d2a_stall_pct": d2a_stall,
        "src_data_stall_pct": src_data_stall,
        "sem_t1_pct": sem_t1,
        "sem_t2_pct": sem_t2,
        "noc_r0_in0_req_pct": noc_in0_req,
        "noc_r0_in0_grant_pct": noc_in0_grant,
        "noc_r0_in0_stall_pct": max(0, (noc_in0_req or 0) - (noc_in0_grant or 0))
        if not pd.isna(noc_in0_req)
        else float("nan"),
        "noc_r0_out0_req_pct": noc_out0_req,
        "noc_r0_out0_stall_pct": max(0, (noc_out0_req or 0) - (noc_out0_grant or 0))
        if not pd.isna(noc_out0_req)
        else float("nan"),
        "unp0_req_pct": unp0_req,
        "unp0_stall_pct": max(0, (unp0_req or 0) - (unp0_grant or 0)) if not pd.isna(unp0_req) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--seq-lens", nargs="+", type=int)
    ap.add_argument("--out", type=Path, default=Path("analysis/ten4679_results.csv"))
    args = ap.parse_args()

    df = load_counters(args.csv)
    if df.empty:
        print("No counter data.")
        return

    run_ids = sorted(df["run_id"].unique())
    labels = []
    for i, rid in enumerate(run_ids):
        labels.append(f"S={args.seq_lens[i]}" if args.seq_lens and i < len(args.seq_lens) else f"run_id={rid}")

    summaries = [summarise_one(df[df["run_id"] == rid], lab) for rid, lab in zip(run_ids, labels)]

    if len(summaries) > 1:
        print()
        print("=" * 110)
        print("  SWEEP SUMMARY (all percentages of math-thread cycles)")
        print("=" * 110)
        cols = [
            "S",
            "cores",
            "FPU%",
            "SFPU%",
            "MATH%",
            "idle%",
            "scbrd%",
            "wr_port%",
            "fidlty%",
            "D2A%",
            "sem_T1",
            "sem_T2",
        ]
        widths = [10, 6, 7, 7, 7, 7, 8, 9, 8, 7, 7, 7]
        print("".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
        for s in summaries:
            row = [
                s["label"],
                s["active_cores"],
                s["fpu_pct"],
                s["sfpu_pct"],
                s["math_pct"],
                s["math_idle_residual_pct"],
                s["scoreboard_stall_pct"],
                s["wr_port_stall_pct"],
                s["fidelity_stall_pct"],
                s["d2a_stall_pct"],
                s["sem_t1_pct"],
                s["sem_t2_pct"],
            ]
            print(f"{row[0]:>10}{row[1]:>6d}" + "".join(f"{v:>{widths[i+2]}.2f}" for i, v in enumerate(row[2:])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "seq_len": (args.seq_lens[i] if args.seq_lens and i < len(args.seq_lens) else f"run_{run_ids[i]}"),
            "active_cores": s["active_cores"],
            "fpu_pct": round(s["fpu_pct"], 2),
            "sfpu_pct": round(s["sfpu_pct"], 2),
            "math_pct": round(s["math_pct"], 2),
            "math_idle_residual_pct": round(s["math_idle_residual_pct"], 2),
            "scoreboard_stall_pct": round(s["scoreboard_stall_pct"], 2),
            "wr_port_stall_pct": round(s["wr_port_stall_pct"], 2),
            "fidelity_stall_pct": round(s["fidelity_stall_pct"], 2),
            "d2a_stall_pct": round(s["d2a_stall_pct"], 2),
            "src_data_stall_pct": round(s["src_data_stall_pct"], 2),
            "issue_eff_pct": round(s["issue_eff_pct"], 2),
            "wait_sfpu_t1_pct": round(s["wait_sfpu_t1"], 2),
            "wait_math_t1_pct": round(s["wait_math_t1"], 2),
            "sem_t1_pct": round(s["sem_t1_pct"], 2),
            "sem_t2_pct": round(s["sem_t2_pct"], 2),
        }
        for i, s in enumerate(summaries)
    ]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
