#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op event timeline from device-profiler markers, as delta cycle counts between events.

Reconstructs one steady-state op's life across all worker cores and prints the ordered timeline:

  go                         worker GO / kernels start (NCRISC_GO) -- assumed ~synchronized
  read issued                first NoC read issued (READ_BEFORE_BARRIER)
  first read return          earliest core's first read back (min READ_AFTER_BARRIER)
  read skew                  spread of first-read-return across cores (max-min READ_AFTER_BARRIER)
  first core reads complete  earliest core done with ALL its reads (min READ_LAST_BARRIER)
  last core reads complete   latest core done with ALL its reads (max READ_LAST_BARRIER); the
                             delta first->last = INTER-core read-complete skew (exploitable stagger).
                             Separately, INTRA-core span = per-core (READ_LAST - READ_AFTER), median.
  first core last wr issued  earliest core's last write issued (min WRITE_LAST_ISSUED)
  first core last wr flushed  min WRITE_LAST_FLUSHED
  first core barrier done    min WRITE_AFTER_BARRIER
  last core last wr issued   latest core's last write issued (max WRITE_LAST_ISSUED)  [done-skew begins]
  last core last wr flushed  max WRITE_LAST_FLUSHED
  last core barrier done     max WRITE_AFTER_BARRIER  (= op done)
  go received                next op's go (min NCRISC_GO of op k+1); op-to-op = this - last barrier done

Each row shows delta cycles from the prior row (cum = cycles since 'go'). Aggregated as the median
over steady-state op transitions (trace-instance boundaries dropped via a gap cap). Run with the
device profiler (TT_METAL_DEVICE_PROFILER=1, --use-device-profiler), NOT --read-only.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_op_to_op_profiler_csv import load_profiler_csv, parse_chip_freq_mhz  # noqa: E402

T = "time[cycles since reset]"


def per_core_ops(df):
    """For each worker core, segment markers into op instances delimited by NCRISC_GO.
    Returns {(chip,cx,cy): [ {marker: time, ...}, ... ]} (one dict per op instance, in time order)."""
    want = {
        "NCRISC_GO",
        "READ_BEFORE_BARRIER",
        "READ_AFTER_BARRIER",
        "READ_LAST_BARRIER",
        "WRITE_LAST_ISSUED",
        "WRITE_LAST_FLUSHED",
        "WRITE_AFTER_BARRIER",
    }
    m = df[df["zone name"].isin(want)]
    out = {}
    for (chip, cx, cy), g in m.groupby(["PCIe slot", "core_x", "core_y"], sort=False):
        evs = sorted((int(r[T]), r["zone name"]) for _, r in g.iterrows())
        ops, cur = [], None
        for t, name in evs:
            if name == "NCRISC_GO":
                if cur is not None:
                    cur["go_received"] = t  # next op's go
                    ops.append(cur)
                cur = {"go": t}
            elif cur is not None and name not in cur:  # keep FIRST occurrence per op
                cur[name] = t
        if cur is not None:
            ops.append(cur)
        out[(int(chip), int(cx), int(cy))] = ops
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, default=None)
    ap.add_argument(
        "--max-op-us", type=float, default=5000.0, help="drop op instances longer than this (trace boundary)"
    )
    ap.add_argument("--csv-out", type=Path, default=None, help="append the timeline (one row per event) to this CSV")
    ap.add_argument("--label", type=str, default="", help="label column value for --csv-out rows (e.g. config id)")
    args = ap.parse_args()
    log = (args.input_file or Path("generated/profiler/.logs/profile_log_device.csv")).resolve()
    freq = parse_chip_freq_mhz(log)  # MHz == cycles/us
    df = load_profiler_csv(log)
    core_ops = per_core_ops(df)
    ncores = len(core_ops)
    n = min(len(v) for v in core_ops.values()) if core_ops else 0
    if not ncores or n < 2:
        print("not enough data (need >=2 op instances on every worker core)")
        return 1

    # Align op instances by index across cores; for each op, aggregate across cores.
    REQ = [
        "go",
        "READ_BEFORE_BARRIER",
        "READ_AFTER_BARRIER",
        "READ_LAST_BARRIER",
        "WRITE_LAST_ISSUED",
        "WRITE_LAST_FLUSHED",
        "WRITE_AFTER_BARRIER",
        "go_received",
    ]
    # Timeline event order (the CSV/print walk uses exactly these, in this order). Extra per-op
    # scalars (e.g. intra_read_cyc) are stored on the row but kept OUT of this list.
    ORDER = [
        "go",
        "read issued",
        "first read return",
        "read skew (last core 1st return)",
        "first core reads complete",
        "last core reads complete",
        "first core last write issued",
        "first core last write flushed",
        "first core barrier done",
        "last core last write issued",
        "last core last write flushed",
        "last core barrier done",
        "go received",
    ]
    rows = []  # one per aligned op instance: dict of timeline timestamps
    for i in range(n):
        ops_i = [ops[i] for ops in core_ops.values()]
        if any(any(k not in o for k in REQ) for o in ops_i):
            continue
        col = lambda k: np.array([o[k] for o in ops_i], dtype="int64")
        go = col("go")
        rb = col("READ_BEFORE_BARRIER")
        ra = col("READ_AFTER_BARRIER")
        rl = col("READ_LAST_BARRIER")
        wi = col("WRITE_LAST_ISSUED")
        wf = col("WRITE_LAST_FLUSHED")
        wb = col("WRITE_AFTER_BARRIER")
        gr = col("go_received")
        rows.append(
            {
                "go": int(go.min()),
                "read issued": int(rb.min()),
                "first read return": int(ra.min()),
                "read skew (last core 1st return)": int(ra.max()),
                # first/last CORE to finish ALL its reads (min/max of READ_LAST across cores);
                # the delta between them (below) is the INTER-core read-complete skew.
                "first core reads complete": int(rl.min()),
                "last core reads complete": int(rl.max()),
                "first core last write issued": int(wi.min()),
                "first core last write flushed": int(wf.min()),
                "first core barrier done": int(wb.min()),
                "last core last write issued": int(wi.max()),
                "last core last write flushed": int(wf.max()),
                "last core barrier done": int(wb.max()),
                "go received": int(gr.min()),
                # INTRA-core read span (per core: last-read-complete - first-read-return), median
                # across cores. NOT a timeline event -- kept off ORDER.
                "intra_read_cyc": float(np.median(rl - ra)),
            }
        )
    cap = args.max_op_us * freq
    steady = [r for r in rows if 0 < (r["go received"] - r["go"]) < cap]
    if not steady:
        print(f"no steady-state op instances under {args.max_op_us} us (got {len(rows)} aligned)")
        return 1

    order = ORDER
    # median delta (from prior event) across steady ops, in cycles
    deltas = {}
    for j, ev in enumerate(order):
        if j == 0:
            deltas[ev] = 0.0
        else:
            d = [r[ev] - r[order[j - 1]] for r in steady]
            deltas[ev] = float(np.median(d))

    print(f"freq={freq:.0f} MHz  cores={ncores}  steady op transitions={len(steady)}  (cycles; 1 cyc = 1 ns @ 1GHz)")
    print(f"{'event':<34} {'Δcyc (from prior)':>18} {'cum cyc (from go)':>18}")
    cum = 0.0
    for ev in order:
        cum += deltas[ev]
        print(f"{ev:<34} {deltas[ev]:>18,.0f} {cum:>18,.0f}")
    print(f"\nop-to-op (last barrier done -> go received) = {deltas['go received']:,.0f} cyc")
    print(f"read skew (first->last core first-read-return) = {deltas['read skew (last core 1st return)']:,.0f} cyc")
    done_skew = float(np.median([r["last core barrier done"] - r["first core barrier done"] for r in steady]))
    print(f"done skew (first->last core barrier done) = {done_skew:,.0f} cyc")
    # Read-completion skew decomposition (the point of this run):
    #   INTER-core = spread across cores of when each finishes ALL its reads (exploitable stagger).
    #   INTRA-core = a single core's own read span (first-return -> last-return), median across cores
    #                (the floor: even one core's reads aren't instantaneous).
    inter_read = float(np.median([r["last core reads complete"] - r["first core reads complete"] for r in steady]))
    intra_read = float(np.median([r["intra_read_cyc"] for r in steady]))
    print(f"read INTER-core skew (first->last core reads-complete) = {inter_read:,.0f} cyc")
    print(f"read INTRA-core span (per-core last-return - first-return, median) = {intra_read:,.0f} cyc")
    print(f"  inter/intra ratio = {inter_read/intra_read:.2f}" if intra_read > 0 else "  intra=0")

    # Persist the timeline: one row per event with delta-from-prior and cumulative-from-go, in
    # both cycles and us (1 cyc = 1/freq us). Appends so a sweep can collect many configs (one
    # --label per config) into a single CSV. done_skew is derivable as
    # cum_us(last core barrier done) - cum_us(first core barrier done).
    if args.csv_out:
        import csv

        new_file = not args.csv_out.exists()
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["label", "cores", "event", "delta_cyc", "cum_cyc", "delta_us", "cum_us"])
            cum = 0.0
            for ev in order:
                cum += deltas[ev]
                w.writerow(
                    [
                        args.label,
                        ncores,
                        ev,
                        f"{deltas[ev]:.0f}",
                        f"{cum:.0f}",
                        f"{deltas[ev]/freq:.4f}",
                        f"{cum/freq:.4f}",
                    ]
                )
            # Trailing derived scalar: intra-core read span (median per core). delta==cum here; it's
            # not part of the cumulative timeline (inter-core read skew is the delta of the
            # "last core reads complete" event above).
            w.writerow(
                [
                    args.label,
                    ncores,
                    "intra_core_read_span",
                    f"{intra_read:.0f}",
                    f"{intra_read:.0f}",
                    f"{intra_read/freq:.4f}",
                    f"{intra_read/freq:.4f}",
                ]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
