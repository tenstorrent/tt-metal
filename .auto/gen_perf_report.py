#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate .auto/tracy_perf_report.log from a tracy profiler CSV.

Usage:
    python .auto/gen_perf_report.py [ops_perf_results.csv]

If no CSV is given, the newest generated/profiler/reports/*/ops_perf_results_*.csv
is used. Runs `tt-perf-report` to normalize the CSV, then writes an aggregated
summary + optimization candidates + raw tt-perf-report output to
.auto/tracy_perf_report.log. This file is meant to be committed/shared so another
agent can pick up the optimization work.

Fill in perf numbers via env vars (optional):
    BEST_MS, AVG_MS, PCC  — e.g. BEST_MS=1560.6 AVG_MS=1612 PCC=0.9530
"""

import csv
import glob
import os
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, ".auto", "tracy_perf_report.log")
NORM = "/tmp/ttperf_norm.csv"


def find_latest_csv():
    pat = os.path.join(REPO, "generated/profiler/reports/*/ops_perf_results_*.csv")
    files = glob.glob(pat)
    if not files:
        sys.exit(f"No profiler CSV found under {pat}")
    return max(files, key=os.path.getmtime)


def f(r, k):
    try:
        return float(r.get(k, "") or 0)
    except ValueError:
        return 0.0


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO).stdout.strip()


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_csv()
    # Normalize via tt-perf-report (adds FLOP%/DRAM%/Bound columns).
    subprocess.run(
        [
            "tt-perf-report",
            csv_path,
            "--ignore-signposts",
            "--no-color",
            "--no-host-ops",
            "--no-merge-devices",
            "--csv",
            NORM,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(open(NORM)))
    d0 = [r for r in rows if r.get("Device") in ("0", "0.0")]
    tot = sum(f(r, "Device Time") for r in d0 if f(r, "Device Time") > 0)

    agg = defaultdict(lambda: [0.0, 0, 0.0, 0.0])  # time, n, flop%sum, dram%sum
    first = {}
    for r in d0:
        dt = f(r, "Device Time")
        if dt <= 0:
            continue
        op = r["OP Code"]
        agg[op][0] += dt
        agg[op][1] += 1
        agg[op][2] += f(r, "FLOPs %")
        agg[op][3] += f(r, "DRAM %")
        first.setdefault(op, r)

    head = sh(["git", "log", "-1", "--oneline"])
    date = sh(["date", "-u", "+%Y-%m-%d %H:%M:%S UTC"])
    best = os.environ.get("BEST_MS", "?")
    avg = os.environ.get("AVG_MS", "?")
    pcc = os.environ.get("PCC", "?")

    out = []
    P = out.append
    P("=" * 100)
    P("BGE-M3 B12/S8192 TP=2 SEQUENCE-PARALLEL (1x N300 = 2 chips) — tt-perf-report SUMMARY")
    P("Regenerated on every tracy run via: python .auto/gen_perf_report.py [csv]")
    P("=" * 100)
    P(f"Generated : {date}")
    P(f"Commit    : {head}")
    P(
        f"Perf      : traced best (measure.sh) = {best}ms  |  tracy-unrolled avg = {avg}ms  |  baseline single-chip = 3213ms"
    )
    P(f"PCC gate  : 0.93 (current {pcc})")
    P(f"CSV       : {os.path.relpath(csv_path, REPO)}")
    P("")
    P("ARCHITECTURE: sequence-parallel TP2. Activations sharded on seq dim (S/2=4096 per chip),")
    P("weights replicated. MLP/LN/embed/output-proj token-local (zero comm). Attention all-gathers")
    P("K,V only (Q stays local): local Sq=4096 queries attend to full gathered Sk=8192 keys.")
    P("")
    P("=" * 100)
    P("AGGREGATE BY OP TYPE (device 0, sorted by total device time)")
    P("=" * 100)
    P(f"{'total_ms':>9} {'%':>5} {'n':>4} {'avg_us':>8} {'FLOP%':>6} {'DRAM%':>6} {'cores':>5} {'bound':>5}  OP")
    for op, (s, n, fl, dr) in sorted(agg.items(), key=lambda x: -x[1][0]):
        r = first[op]
        P(
            f"{s/1000:9.1f} {100*s/tot:5.1f} {n:4d} {s/n:8.0f} {fl/n:6.1f} {dr/n:6.1f} "
            f"{r.get('Cores',''):>5} {r.get('Bound',''):>5}  {op[:60]}"
        )
    P("")
    P(f"device0 total device-time = {tot/1000:.1f}ms across {sum(v[1] for v in agg.values())} ops (x2 tracy passes)")
    P("")
    P("=" * 100)
    P("OPTIMIZATION CANDIDATES (ranked by opportunity) — see .auto/prompt.md for full lever inventory")
    P("=" * 100)
    P(CANDIDATES)
    P("=" * 100)
    P("RAW tt-perf-report OUTPUT (native tool, ops >=0.5%, first ~2 layers)")
    P("cmd: tt-perf-report <csv> --ignore-signposts --no-color --no-host-ops --no-merge-devices --min-percentage 0.5")
    P("=" * 100)
    raw = subprocess.run(
        [
            "tt-perf-report",
            csv_path,
            "--ignore-signposts",
            "--no-color",
            "--no-host-ops",
            "--no-merge-devices",
            "--min-percentage",
            "0.5",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    out.extend(raw[:80])

    with open(LOG, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {LOG} ({len(out)} lines) from {os.path.relpath(csv_path, REPO)}")


CANDIDATES = """
1. SDPA — ~58% of runtime, ~33ms/op. BIGGEST LEVER.
   - Chunk config q512/k512 BOUNDED (q1024 crashes=kernel limit, k1024 crashes=L1).
   - Low FLOP util (~22% of 156 TFLOP/s peak) => softmax/SFPU-reduction bound, not matmul bound.
   - TODO: SWEEP SDPA program_config grid for the Sq=4096/Sk=8192 shape in-model.

2. AllGather (K/V) — ~22% of runtime, ~7ms/op, ONLY 9 CORES.
   - USER HINT: give the CCL more cores (8x8, 8x4) via sub_core_grids / worker grid.
   - N300 = 1 ethernet link/axis (HW cap) but the gather worker core count IS tunable.
   - Best idea: OVERLAP K/V gather with QKV/MLP compute via 2 command queues.

3. Matmuls (SWEEP block sizes, don't one-knob — current configs tuned for M=8192 single-chip, now M=4096):
   - MLPwo b12x4096x1024x4096 ~4.8%, ~52% FLOP
   - QKV   b12x4096x1024x3072 ~3.8%, ~49% FLOP (k8; k16 crashes L1)
   - MLPwi b12x4096x4096x1024 ~3.6%, ~67% FLOP (best)
   - AttnOut b12x4096x1024x1024 ~1.1%, ~54% FLOP

4. LayerNorm ~3.4% bandwidth-bound; sharding impossible at scale. GenericOp (residual) ~2%.

METHOD: tracy perf -> pick candidate -> SWEEP the config space -> apply winner
-> measure.sh keep/revert -> re-run tracy -> re-run this script.
"""

if __name__ == "__main__":
    main()
