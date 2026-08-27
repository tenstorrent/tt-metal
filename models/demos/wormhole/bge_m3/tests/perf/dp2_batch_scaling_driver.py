# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parent driver for the DP=2 batch-scaling sweep (fail-forward).

Runs each (local_batch, variant) as a FRESH subprocess with an external timeout,
parses the worker's RESULT_JSON line, and appends a row to results.csv (and
failures.csv for non-PASS). A timeout or clean L1 OOM is a failed data point:
recorded, then the campaign continues to the next independent case.

Usage:
  python dp2_batch_scaling_driver.py \
      --cases 1:stock_dram 6:stock_dram 2:stock_dram ... \
      --timeout 900

Each case is "<local_batch>:<variant>[:<l1_handoff>]". The l1_handoff (for
jit_l1_* variants) is exported as BGE_L1_HANDOFF to the worker.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

RESULTS = ".auto/batch_scaling/results.csv"
FAILURES = ".auto/batch_scaling/failures.csv"
RAW_DIR = ".auto/batch_scaling/raw"

COLUMNS = [
    "sha",
    "local_batch",
    "global_batch",
    "variant",
    "sdpa_impl",
    "l1_tensors",
    "status",
    "pcc",
    "iterations",
    "min_ms",
    "p50_ms",
    "mean_ms",
    "p95_ms",
    "std_ms",
    "input_copy_ms",
    "device_ms",
    "global_tokens_per_s",
    "per_chip_sequences_per_s",
    "profiler_artifact",
    "notes",
]

WORKER = "models/demos/wormhole/bge_m3/tests/perf/dp2_batch_scaling.py"


def _ensure_csv(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(columns)


def _append(path, columns, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([row.get(c, "") for c in columns])


def run_case(local_batch, variant, l1_handoff, seed, timeout, screen_iters, final_iters):
    env = dict(os.environ)
    env["TT_VISIBLE_DEVICES"] = env.get("TT_VISIBLE_DEVICES", "0")
    if l1_handoff:
        env["BGE_L1_HANDOFF"] = l1_handoff
    cmd = [
        sys.executable,
        WORKER,
        "--local-batch",
        str(local_batch),
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--screen-iters",
        str(screen_iters),
        "--final-iters",
        str(final_iters),
    ]
    print(f"\n=== CASE local_batch={local_batch} variant={variant} l1={l1_handoff or '-'} ===", flush=True)
    t0 = time.perf_counter()
    result_json = None
    status = "FAIL"
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + "\n" + proc.stderr
        for line in out.splitlines():
            if line.startswith("RESULT_JSON="):
                result_json = json.loads(line[len("RESULT_JSON=") :])
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        exit_code = -9
        result_json = {
            "local_batch": local_batch,
            "global_batch": 2 * local_batch,
            "variant": variant,
            "status": "FAIL",
            "notes": f"TIMEOUT after {timeout}s",
        }
    elapsed = time.perf_counter() - t0

    if result_json is None:
        result_json = {
            "local_batch": local_batch,
            "global_batch": 2 * local_batch,
            "variant": variant,
            "status": "FAIL",
            "notes": f"no RESULT_JSON (exit={exit_code})",
        }
    status = result_json.get("status", "FAIL")
    result_json.setdefault("l1_tensors", l1_handoff)
    print(
        f"    -> status={status} exit={exit_code} elapsed={elapsed:.1f}s notes={result_json.get('notes','')}",
        flush=True,
    )

    # Persist raw samples if present.
    if "raw_samples" in result_json:
        os.makedirs(RAW_DIR, exist_ok=True)
        rawf = f"{RAW_DIR}/b{local_batch}_{variant}_s{seed}.json"
        with open(rawf, "w") as f:
            json.dump(result_json["raw_samples"], f)
        result_json["profiler_artifact"] = result_json.get("profiler_artifact", "") or rawf
        del result_json["raw_samples"]

    _append(RESULTS, COLUMNS, result_json)
    if status != "PASS":
        _ensure_csv(FAILURES, COLUMNS)
        _append(FAILURES, COLUMNS, result_json)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True, help="each '<lb>:<variant>[:<l1_handoff>]'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--screen-iters", type=int, default=10)
    ap.add_argument("--final-iters", type=int, default=30)
    a = ap.parse_args()

    _ensure_csv(RESULTS, COLUMNS)
    summary = []
    for case in a.cases:
        parts = case.split(":")
        lb = int(parts[0])
        variant = parts[1]
        l1_handoff = parts[2] if len(parts) > 2 else ""
        st = run_case(lb, variant, l1_handoff, a.seed, a.timeout, a.screen_iters, a.final_iters)
        summary.append((case, st))

    print("\n=== SWEEP SUMMARY ===", flush=True)
    for case, st in summary:
        print(f"  {case:<30} {st}", flush=True)


if __name__ == "__main__":
    main()
