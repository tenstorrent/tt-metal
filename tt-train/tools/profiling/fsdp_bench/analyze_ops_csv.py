# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Aggregate a tt-metal profiler ops_perf_results_*.csv into a per-op-code device-time table.

Splits the run into steps using the tt-train profiler_marker noops
("dataloader_step_done", "forward_pass_done", ...), drops warm-up steps, and
reports per step (averaged over the remaining steps, per device):
  * device kernel time per OP CODE and per phase
  * op counts
  * device idle time (sum of OP TO OP LATENCY), i.e. time the device waited for the host

Usage: python analyze_ops_csv.py <ops_perf_results.csv> [--warmup 2] [--device 0] [--top 25]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

PHASE_MARKERS = [
    "dataloader_step_done",
    "forward_pass_done",
    "backward_pass_done",
    "gradient_sync_done",
    "optimizer_step_done",
]
PHASE_NAME = {
    "dataloader_step_done": "forward",
    "forward_pass_done": "backward",
    "backward_pass_done": "gradsync",
    "gradient_sync_done": "optimizer",
    "optimizer_step_done": "other",
}


def classify(op: str) -> str:
    o = op.lower()
    if "all_gather" in o or "allgather" in o:
        return "CCL all_gather"
    if "reduce_scatter" in o or "reducescatter" in o:
        return "CCL reduce_scatter"
    if "all_reduce" in o or "allreduce" in o:
        return "CCL all_reduce"
    if "matmul" in o or "linear" in o:
        return "matmul"
    if "sdpa" in o or "attention" in o:
        return "attention"
    if "embedding" in o:
        return "embedding"
    if "moreh" in o or "adam" in o:
        return "optimizer"
    if "rmsnorm" in o or "layernorm" in o or "norm" in o:
        return "norm"
    if "softmax" in o or "cross_entropy" in o or "nll" in o:
        return "loss/softmax"
    if any(
        k in o
        for k in (
            "binary",
            "unary",
            "eltwise",
            "multiply",
            "add",
            "subtract",
            "mul",
            "silu",
            "gelu",
            "exp",
            "rsqrt",
            "sqrt",
            "where",
            "clamp",
            "typecast",
            "fill",
            "sum",
            "reduce",
            "mean",
            "concat",
            "slice",
            "transpose",
            "permute",
            "reshape",
            "repeat",
            "pad",
            "untilize",
            "tilize",
        )
    ):
        return "eltwise/data-movement"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--device", type=int, default=None, help="restrict to one DEVICE ID (default: average over devices)"
    )
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    need = ["OP CODE", "DEVICE ID", "DEVICE KERNEL DURATION [ns]", "GLOBAL CALL COUNT"]
    for c in need:
        if c not in df.columns:
            sys.exit(f"missing column {c}; have {list(df.columns)[:20]}")
    if args.device is not None:
        df = df[df["DEVICE ID"] == args.device]
    ndev = df["DEVICE ID"].nunique()
    df = df.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"])

    # Assign step / phase per device by scanning the marker noops in call order.
    step_col, phase_col = [], []
    attr_col = "ATTRIBUTES" if "ATTRIBUTES" in df.columns else None
    for dev, g in df.groupby("DEVICE ID", sort=False):
        step, phase, last_marker = 0, "other", None
        for op, attr in zip(g["OP CODE"].astype(str), g[attr_col].astype(str) if attr_col else [""] * len(g)):
            # tt-train markers are ProfilerNoopOperation rows whose identifier lives in ATTRIBUTES
            key = attr if "ProfilerNoop" in op else op
            marker = next((m for m in PHASE_MARKERS if m in key), None)
            # each profiler_marker emits several identical noop rows; count a transition once
            if marker and marker != last_marker:
                phase = PHASE_NAME[marker]
                if marker == "optimizer_step_done":
                    step += 1
            if marker:
                last_marker = marker
            step_col.append(step)
            phase_col.append(phase)
    df["step"] = step_col
    df["phase"] = phase_col
    df["kernel_ms"] = df["DEVICE KERNEL DURATION [ns]"] / 1e6
    lat = "OP TO OP LATENCY [ns]"
    df["gap_ms"] = df[lat] / 1e6 if lat in df.columns else 0.0
    df["class"] = df["OP CODE"].astype(str).map(classify)
    df = df[~df["OP CODE"].astype(str).str.contains("ProfilerNoop")]

    nsteps = df["step"].max()
    steady = df[(df["step"] >= args.warmup) & (df["step"] < nsteps)]
    n = steady["step"].nunique()
    if n == 0:
        sys.exit(f"not enough steps ({nsteps}) for warmup {args.warmup}")
    scale = 1.0 / (n * ndev)

    print(f"file: {args.csv}\ndevices: {ndev}  total steps: {nsteps}  steady steps averaged: {n}\n")
    tot_k = steady["kernel_ms"].sum() * scale
    tot_gap = steady["gap_ms"].sum() * scale
    print(
        f"per step, per device:  kernel busy {tot_k:9.1f} ms   op-to-op gaps (device idle) {tot_gap:9.1f} ms   ops {len(steady) * scale:8.0f}\n"
    )

    print("== by phase ==")
    ph = steady.groupby("phase").agg(
        kernel_ms=("kernel_ms", "sum"), gap_ms=("gap_ms", "sum"), ops=("kernel_ms", "size")
    )
    ph = ph * scale
    print(ph.round(2).to_string())

    print("\n== by class ==")
    cl = (
        steady.groupby("class").agg(kernel_ms=("kernel_ms", "sum"), gap_ms=("gap_ms", "sum"), ops=("kernel_ms", "size"))
        * scale
    )
    cl["kernel_%"] = 100 * cl["kernel_ms"] / tot_k
    print(cl.sort_values("kernel_ms", ascending=False).round(2).to_string())

    print("\n== by class x phase (kernel ms) ==")
    print(
        (steady.pivot_table(index="class", columns="phase", values="kernel_ms", aggfunc="sum", fill_value=0) * scale)
        .round(2)
        .to_string()
    )

    print(f"\n== top {args.top} op codes ==")
    top = steady.groupby("OP CODE").agg(
        kernel_ms=("kernel_ms", "sum"),
        gap_ms=("gap_ms", "sum"),
        ops=("kernel_ms", "size"),
        avg_us=("kernel_ms", "mean"),
    )
    top[["kernel_ms", "gap_ms", "ops"]] *= scale
    top["avg_us"] *= 1e3
    top["kernel_%"] = 100 * top["kernel_ms"] / tot_k
    print(top.sort_values("kernel_ms", ascending=False).head(args.top).round(2).to_string())

    # Overlap accounting: per step and device, wall = last kernel end - first kernel start; busy
    # time split into CCL vs compute. overlap = compute_busy + ccl_busy - wall (0 = serialized).
    if "DEVICE FW START CYCLE" in steady.columns and "DEVICE FW END CYCLE" in steady.columns:
        rows = []
        for (dev, st), g in steady.groupby(["DEVICE ID", "step"]):
            is_ccl = g["class"].str.startswith("CCL")
            span_cycles = g["DEVICE FW END CYCLE"].max() - g["DEVICE FW START CYCLE"].min()
            # derive cycles->ms from this group's own kernel durations
            cyc = (g["DEVICE FW END CYCLE"] - g["DEVICE FW START CYCLE"]).sum()
            ns = (
                g["DEVICE FW DURATION [ns]"].sum()
                if "DEVICE FW DURATION [ns]" in g
                else g["DEVICE KERNEL DURATION [ns]"].sum()
            )
            ns_per_cycle = ns / cyc if cyc else 1.0
            wall = span_cycles * ns_per_cycle / 1e6
            rows.append((dev, st, wall, g.loc[~is_ccl, "kernel_ms"].sum(), g.loc[is_ccl, "kernel_ms"].sum()))
        ov = pd.DataFrame(rows, columns=["dev", "step", "wall_ms", "compute_ms", "ccl_ms"])
        m = ov.groupby("step")[["wall_ms", "compute_ms", "ccl_ms"]].mean()
        m["hidden_ccl_ms"] = (m["compute_ms"] + m["ccl_ms"] - m["wall_ms"]).clip(lower=0)
        print("\n== overlap accounting (per step, mean over devices) ==")
        print(m.round(1).to_string())

    ccl = steady[steady["class"].str.startswith("CCL")]
    if len(ccl):
        print("\n== CCL ops: per-call device kernel time distribution (us) ==")
        shape_cols = [c for c in ccl.columns if c.startswith("INPUT_0_") and "PAD[LOGICAL]" in c]
        keys = ["OP CODE", "phase"] + shape_cols
        g = ccl.groupby(keys).agg(
            calls=("kernel_ms", "size"),
            mean_us=("kernel_ms", "mean"),
            max_us=("kernel_ms", "max"),
            gap_us=("gap_ms", "mean"),
        )
        g["calls"] = g["calls"] * scale
        for c in ("mean_us", "max_us", "gap_us"):
            g[c] *= 1e3
        print(g.round(1).to_string())


if __name__ == "__main__":
    main()
