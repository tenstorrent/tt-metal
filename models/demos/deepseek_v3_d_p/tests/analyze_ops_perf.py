# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Summarize (and diff) tracy ``ops_perf_results_*.csv`` from a deepseek_v3_d_p prefill run.

Adapted from ``models/demos/gemma4/tests/analyze_ops_perf.py``. The accounting rules are that
file's and they matter more than the code:

* **Per device, not summed across the mesh.** All 32 devices run concurrently, so latency is what
  ONE device does. Summing across the mesh inflates every number 32x.
* **DEVICE KERNEL DURATION, not OP TO OP LATENCY.** The latter also counts host-side gaps between
  invocations and profiler stalls, which on a traced replay sums to something like 30x the truth.
  Kernel durations come from hardware counters and survive host noise and program compilation.
* **NaN-duration rows are reported, not silently dropped**, so an undercount cannot read as a
  speedup.
* **Under ~5% is noise** on a shared box. Re-run before believing a diff.

Difference from the gemma4 version: there, one *layer* was invoked repeatedly, so an op's
invocation count was the number of measured layers. Here one *forward* runs all 36 layers, so each
per-layer op appears 36 times per forward. This reports **per-forward** totals (which is exactly the
demo's per-token latency, since the demo re-prefills per token) and derives per-layer by dividing.

Usage::

    python models/demos/deepseek_v3_d_p/tests/analyze_ops_perf.py REPORT.csv
    python models/demos/deepseek_v3_d_p/tests/analyze_ops_perf.py BASELINE.csv CANDIDATE.csv
"""

import os
import sys

import pandas as pd

NUM_LAYERS = int(os.environ.get("ANALYZE_NUM_LAYERS", 36))


def load(path, device_id=0):
    """Per-op device-time totals for one device, for the ONE signposted forward in the report.

    ``profile_prefill.py::test_ops`` runs WARMUP forwards (compile- and dispatch-contaminated) and
    then brackets exactly one forward in ``signpost("start")`` / ``signpost("stop")``. Those
    signposts land in this CSV as rows with OP CODE ``start`` / ``stop``, so the measured region is
    simply the rows between them — no invocation arithmetic, and nothing to get wrong.

    That matters here: an earlier version assumed each op runs once per layer and kept the trailing
    ``NUM_LAYERS`` rows, which is wrong for the many ops that run several times per layer (Matmul
    runs ~10x per layer, so it has ~361 rows per forward, not 36) and would have silently reported a
    tenth of the real matmul time.
    """
    df = pd.read_csv(path, low_memory=False)
    if "DEVICE KERNEL DURATION [ns]" not in df.columns:
        raise SystemExit(f"{path}: no DEVICE KERNEL DURATION column — was TT_METAL_DEVICE_PROFILER=1 set?")

    codes = df["OP CODE"].astype(str)
    starts, stops = df.index[codes == "start"], df.index[codes == "stop"]
    if len(starts) and len(stops):
        lo = starts[-1]
        hi = stops[stops > lo][-1] if (stops > lo).any() else df.index[-1]
        df = df.loc[lo:hi]
        region = f"signposted region rows {lo}..{hi}"
    else:
        region = "WHOLE REPORT (no start/stop signposts found — includes warmup, treat with suspicion)"

    df = df.copy()
    df["dur"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce")
    d = df[df["DEVICE ID"] == device_id]
    if d.empty:
        raise SystemExit(f"{path}: no rows for device {device_id}")

    g = d.groupby("OP CODE")["dur"].agg(["count", "sum", "median"])
    g["us_per_forward"] = g["sum"] / 1000.0
    g["calls_per_forward"] = g["count"]
    g["us_per_call"] = g["us_per_forward"] / g["calls_per_forward"].clip(lower=1)
    g["nan_rows"] = d.groupby("OP CODE")["dur"].apply(lambda s: int(s.isna().sum()))
    g = g[g["calls_per_forward"] > 0]
    return g.sort_values("us_per_forward", ascending=False), region


def summarize(path):
    g, region = load(path)
    total = g["us_per_forward"].sum()
    print(f"\n{path}")
    print(f"  device 0 only; {region}")
    nan_total = int(g["nan_rows"].sum())
    if nan_total:
        print(f"  NOTE {nan_total} rows had no device duration and are excluded (this is an UNDERCOUNT)")
    print(f"\n  {'OP':38s} {'calls':>6s} {'us/fwd':>10s} {'%':>7s} {'us/call':>9s}")
    for op, r in g.iterrows():
        print(
            f"  {op:38s} {r['calls_per_forward']:6.0f} {r['us_per_forward']:10.1f} "
            f"{100 * r['us_per_forward'] / total:6.1f}% {r['us_per_call']:9.1f}"
        )
    print(f"  {'TOTAL (device kernel time, 1 forward)':38s} {'':6s} {total:10.1f} {100.0:6.1f}%")
    print(f"\n  => device busy {total/1000:.1f} ms per forward = per generated token")
    ccl = g.index.str.contains("AllGather|ReduceScatter|AllReduce|Broadcast", case=False, regex=True)
    if ccl.any():
        c = g[ccl]["us_per_forward"].sum()
        print(f"  CCL total: {c/1000:.1f} ms ({100*c/total:.1f}%)")
    print(
        "\n  Compare this against the WALL CLOCK from test_walltime (run without the profiler):\n"
        "    wall_clock - this = host share. If the host share is large, trace replay can remove\n"
        "    most of it; if it is small, the device is the wall and only decode changes the answer."
    )
    return g


def diff(base_path, cand_path):
    b, _ = load(base_path)
    c, _ = load(cand_path)
    bt, ct = b["us_per_forward"].sum(), c["us_per_forward"].sum()
    print(f"\nbaseline  {base_path}\ncandidate {cand_path}\n")
    print(f"  {'OP':38s} {'base us':>10s} {'cand us':>10s} {'delta':>10s} {'%':>8s}")
    for op in sorted(set(b.index) | set(c.index)):
        bv = float(b["us_per_forward"].get(op, 0.0))
        cv = float(c["us_per_forward"].get(op, 0.0))
        pct = (cv - bv) / bv * 100 if bv else float("inf")
        flag = "" if abs(pct) < 5 else ("  <-- FASTER" if pct < 0 else "  <-- SLOWER")
        print(f"  {op:38s} {bv:10.1f} {cv:10.1f} {cv - bv:+10.1f} {pct:+7.1f}%{flag}")
    print(f"  {'TOTAL':38s} {bt:10.1f} {ct:10.1f} {ct - bt:+10.1f} {(ct - bt) / bt * 100:+7.1f}%")
    print("\n  Under ~5% is noise on a shared machine. Re-run before believing it.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        summarize(sys.argv[1])
    elif len(sys.argv) == 3:
        diff(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(__doc__)
