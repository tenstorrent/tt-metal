# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Summarize (and diff) tracy ``ops_perf_results_*.csv`` from a gemma4 prefill run.

The profiler emits a 37 MB CSV per run and a ~300 MB .tracy. The CSV is the part worth
keeping: this turns it into a per-op, per-layer-invocation table, and diffs two of them
so a perf change is measured rather than eyeballed in the GUI.

Why the numbers are computed the way they are:

* **Per device, not summed across devices.** All 32 devices run concurrently, so latency
  is what one device does. Summing across the mesh inflates everything 32x.
* **Warmup invocations dropped.** A traced run is warmup + capture + warm replay before
  the measured replays; the first three are compile- and dispatch-contaminated.
* **DEVICE KERNEL DURATION, not OP TO OP LATENCY.** The latter sums to ~126 ms/layer in a
  traced replay because it also counts the host-side gaps between invocations and
  profiler stalls. Device kernel/FW durations come from hardware counters and are sound.
* A few percent of rows have a NaN duration and are skipped; the count is reported so a
  silent undercount cannot be mistaken for a speedup.

Usage::

    python -m models.demos.gemma4.tests.analyze_ops_perf REPORT.csv
    python -m models.demos.gemma4.tests.analyze_ops_perf BASELINE.csv CANDIDATE.csv
"""

from __future__ import annotations

import sys

import pandas as pd

# Invocations that precede the measured one in test_prefill_layer_perf: the eager
# compile pass, the capture, and GEMMA4_PERF_WARMUP_ITERS warm replays (default 5).
# Override with the env var if you changed it. Only the LAST invocation is the measured
# one, so this analyzer keeps just that; when a report was produced with signposts,
# prefer tt-perf-report, which slices the region properly.
WARMUP_INVOCATIONS = 2 + int(__import__("os").environ.get("GEMMA4_PERF_WARMUP_ITERS", "5"))
# Gemma4-31B layer mix, for the whole-model extrapolation.
N_SLIDING, N_GLOBAL = 50, 10


def load(path, device_id=0, warmup=WARMUP_INVOCATIONS):
    """Per-op stats for one device, averaged over the measured invocations."""
    df = pd.read_csv(path, low_memory=False)
    df["dur"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce")
    d = df[df["DEVICE ID"] == device_id].copy()
    if d.empty:
        raise SystemExit(f"{path}: no rows for device {device_id}")
    # Ops repeat in the same order every invocation, so cumcount within an op code is
    # that op's invocation index.
    d["slot"] = d.groupby("OP CODE").cumcount()
    # Count invocations from an op that runs EXACTLY once per layer. Taking the min over
    # all ops is wrong: ops that only appear during warmup (Tilize, for one) would cap the
    # count at their own row total and silently rescale every per-layer number.
    if (d["OP CODE"] == "SDPAOperation").any():
        n_inv = int((d["OP CODE"] == "SDPAOperation").sum())
    else:
        n_inv = int(d.groupby("OP CODE")["slot"].max().max()) + 1
    meas = d[d["slot"] >= warmup]
    n_meas = max(1, n_inv - warmup)

    g = meas.groupby("OP CODE")["dur"].agg(["count", "sum", "median"])
    g["us_per_layer"] = g["sum"] / 1000 / n_meas
    g["calls_per_layer"] = g["count"] / n_meas
    g["nan_rows"] = meas.groupby("OP CODE")["dur"].apply(lambda s: int(s.isna().sum()))
    return g.sort_values("us_per_layer", ascending=False), n_inv, n_meas


def _shape_of(path, op="SDPAOperation", device_id=0):
    """Q shape of the first SDPA, which identifies the layer type (head_dim 512 = global)."""
    df = pd.read_csv(path, low_memory=False)
    rows = df[(df["DEVICE ID"] == device_id) & (df["OP CODE"] == op)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    keys = [f"INPUT_0_{a}_PAD[LOGICAL]" for a in ("W", "Z", "Y", "X")]
    return [r[k] for k in keys if k in rows.columns]


def summarize(path):
    g, n_inv, n_meas = load(path)
    total = g["us_per_layer"].sum()
    shape = _shape_of(path)
    print(f"\n{path}")
    print(f"  invocations={n_inv} (measured {n_meas}, dropped {WARMUP_INVOCATIONS} warmup)  device 0")
    if shape:
        print(f"  SDPA Q shape {shape}  -> {'global' if '512' in str(shape[-1]) else 'sliding'} layer")
    nan_total = int(g["nan_rows"].sum())
    if nan_total:
        print(f"  NOTE {nan_total} rows had no device duration and are excluded (undercount)")
    print(f"\n  {'OP':32s} {'calls':>6s} {'us/layer':>9s} {'%':>6s} {'us/call':>8s}")
    for op, r in g.iterrows():
        print(
            f"  {op:32s} {r['calls_per_layer']:6.0f} {r['us_per_layer']:9.1f} "
            f"{100 * r['us_per_layer'] / total:5.1f}% {r['us_per_layer'] / max(r['calls_per_layer'], 1):8.1f}"
        )
    print(f"  {'TOTAL':32s} {'':6s} {total:9.1f} {100.0:5.1f}%")
    ccl = g.index.str.contains("AllGather|ReduceScatter")
    print(
        f"\n  CCL (AllGather + ReduceScatter): {g[ccl]['us_per_layer'].sum():.1f} us/layer "
        f"({100 * g[ccl]['us_per_layer'].sum() / total:.1f}%)"
    )
    print(
        f"  whole-model estimate if this is the only layer type: "
        f"x{N_GLOBAL} = {N_GLOBAL * total / 1000:.1f} ms, x{N_SLIDING} = {N_SLIDING * total / 1000:.1f} ms"
    )
    return g


def diff(base_path, cand_path):
    b, _, _ = load(base_path)
    c, _, _ = load(cand_path)
    ops = sorted(set(b.index) | set(c.index))
    bt, ct = b["us_per_layer"].sum(), c["us_per_layer"].sum()
    print(f"\nbaseline  {base_path}\ncandidate {cand_path}\n")
    print(f"  {'OP':32s} {'base us':>9s} {'cand us':>9s} {'delta':>9s} {'%':>8s}")
    for op in ops:
        bv = float(b["us_per_layer"].get(op, 0.0))
        cv = float(c["us_per_layer"].get(op, 0.0))
        pct = (cv - bv) / bv * 100 if bv else float("inf")
        flag = "" if abs(pct) < 5 else ("  <-- FASTER" if pct < 0 else "  <-- SLOWER")
        print(f"  {op:32s} {bv:9.1f} {cv:9.1f} {cv - bv:+9.1f} {pct:+7.1f}%{flag}")
    print(f"  {'TOTAL':32s} {bt:9.1f} {ct:9.1f} {ct - bt:+9.1f} {(ct - bt) / bt * 100:+7.1f}%")
    print("\n  Run-to-run noise on a shared machine is a few percent; treat anything under")
    print("  ~5% as unproven and re-run before believing it.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        summarize(sys.argv[1])
    elif len(sys.argv) == 3:
        diff(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(__doc__)
