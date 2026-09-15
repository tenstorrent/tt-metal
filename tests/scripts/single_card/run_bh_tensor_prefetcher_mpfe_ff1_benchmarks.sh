#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Run the complete MPFE policy matrix on the receiver-contiguous 3B FF1
# Tensor-Prefetcher-to-matmul benchmark. The benchmark detects the available
# Blackhole DRAM banks and sizes its receiver ring accordingly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
MPFE_MATMUL_SHAPE="${MPFE_MATMUL_SHAPE:-3B_FF1}"
OUTPUT_DIR="${OUTPUT_DIR:-$TT_METAL_HOME/generated/mpfe-ff1-benchmark-$(date -u +%Y%m%dT%H%M%SZ)}"

export TT_METAL_HOME
export OUTPUT_DIR
export BENCH_DUAL_SENDERS="${BENCH_DUAL_SENDERS:-1}"
export TEST="tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bench.py::test_bench_dram_core_repeats_recv_contig"

"$SCRIPT_DIR/run_bh_tensor_prefetcher_mpfe_benchmarks.sh" \
    -k "$MPFE_MATMUL_SHAPE and shard_contiguous" \
    "$@"

RESULTS_JSONL="$OUTPUT_DIR/results.jsonl"
COMPARISON_CSV="$OUTPUT_DIR/relative-to-static-000.csv"
"${PYTHON:-python3}" - "$RESULTS_JSONL" "$COMPARISON_CSV" <<'PY'
import csv
import json
import sys

input_path, output_path = sys.argv[1:]
with open(input_path, encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
if not rows:
    raise SystemExit(f"no benchmark records found in {input_path}")

baselines = {
    row["suite_iteration"]: row["tflops"]
    for row in rows
    if row.get("run_label") == "static-000"
}
missing_baselines = sorted({row["suite_iteration"] for row in rows} - baselines.keys())
if missing_baselines:
    raise SystemExit(f"missing static-000 baseline for iterations {missing_baselines}")

with open(output_path, "w", encoding="utf-8", newline="") as output:
    writer = csv.DictWriter(
        output,
        fieldnames=["suite_iteration", "run_label", "tflops", "vs_static_000_percent"],
    )
    writer.writeheader()
    for row in rows:
        iteration = row["suite_iteration"]
        baseline = baselines[iteration]
        writer.writerow(
            {
                "suite_iteration": iteration,
                "run_label": row["run_label"],
                "tflops": f'{row["tflops"]:.6f}',
                "vs_static_000_percent": f'{(row["tflops"] / baseline - 1) * 100:.3f}',
            }
        )
PY

printf 'Static-000 comparison: %s\n' "$COMPARISON_CSV"
