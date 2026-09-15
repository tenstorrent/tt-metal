#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Repeat the complete Device-side Tensor Prefetcher MPFE benchmark matrix.
# Every configuration runs in a fresh process and appends human-readable and
# machine-readable results to one timestamped output directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PYTHON="${PYTHON:-python3}"
BENCH_SUITE_ITERATIONS="${BENCH_SUITE_ITERATIONS:-3}"
BENCH_TRACE_REPEATS="${BENCH_TRACE_REPEATS:-50}"
MPFE_HIGH_WEIGHTS="${MPFE_HIGH_WEIGHTS:-3 5 7}"
MPFE_MEDIUM_WEIGHTS="${MPFE_MEDIUM_WEIGHTS:-1 3 5}"
MPFE_ACTIVE_WEIGHTS="${MPFE_ACTIVE_WEIGHTS:-1 3 5 7}"
OUTPUT_DIR="${OUTPUT_DIR:-$TT_METAL_HOME/generated/mpfe-benchmark-$(date -u +%Y%m%dT%H%M%SZ)}"
TEST="${TEST:-tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention}"
PYTEST_ARGS=("$@")

mkdir -p "$OUTPUT_DIR"
TEXT_LOG="$OUTPUT_DIR/benchmark.log"
JSONL_LOG="$OUTPUT_DIR/results.jsonl"
touch "$TEXT_LOG" "$JSONL_LOG"
cd "$TT_METAL_HOME"

read -r -a high_weights <<< "$MPFE_HIGH_WEIGHTS"
read -r -a medium_weights <<< "$MPFE_MEDIUM_WEIGHTS"
read -r -a active_weights <<< "$MPFE_ACTIVE_WEIGHTS"

printf 'MPFE benchmark output: %s\n' "$OUTPUT_DIR" | tee -a "$TEXT_LOG"
printf 'iterations=%s trace_repeats=%s high={%s} medium={%s} active={%s}\n' \
    "$BENCH_SUITE_ITERATIONS" "$BENCH_TRACE_REPEATS" \
    "$MPFE_HIGH_WEIGHTS" "$MPFE_MEDIUM_WEIGHTS" "$MPFE_ACTIVE_WEIGHTS" | tee -a "$TEXT_LOG"

failures=()
run_case() {
    local iteration="$1" name="$2" policy="$3" high="$4" medium="$5" active="$6" force_sync="$7"
    printf '\n== iteration %s: %s ==\n' "$iteration" "$name" | tee -a "$TEXT_LOG"

    local -a env_args=(
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_MEDIUM_WEIGHT
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ACTIVE_WEIGHT
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_IDLE_WEIGHTS
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ACTIVE_WEIGHTS
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FORCE_REQUEST_SYNC
        -u TT_METAL_SLOW_DISPATCH_MODE
        "ARCH_NAME=blackhole"
        "BENCH_TRACE_REPEATS=$BENCH_TRACE_REPEATS"
        "PYTHONPATH=$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
        "TT_METAL_BENCHMARK_RESULT_JSONL=$JSONL_LOG"
        "TT_METAL_BENCHMARK_RUN_LABEL=$name"
        "TT_METAL_BENCHMARK_SUITE_ITERATION=$iteration"
    )
    [[ "$policy" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY=$policy")
    [[ "$high" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT=$high")
    [[ "$medium" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_MEDIUM_WEIGHT=$medium")
    [[ "$active" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ACTIVE_WEIGHT=$active")
    [[ "$force_sync" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FORCE_REQUEST_SYNC=$force_sync")

    if env "${env_args[@]}" "$PYTHON" -m pytest -q -s "$TEST" "${PYTEST_ARGS[@]}" 2>&1 | tee -a "$TEXT_LOG"; then
        printf 'PASS: iteration %s %s\n' "$iteration" "$name" | tee -a "$TEXT_LOG"
    else
        failures+=("iteration-$iteration:$name")
        printf 'FAIL: iteration %s %s\n' "$iteration" "$name" | tee -a "$TEXT_LOG"
    fi
}

for ((iteration = 1; iteration <= BENCH_SUITE_ITERATIONS; ++iteration)); do
    # Run the neutral hardware-default policy first so every suite iteration has
    # an explicit baseline before any priority-weighted configuration.
    run_case "$iteration" "static-000" "static-000" "-" "-" "-" "-"
    run_case "$iteration" "production-default" "-" "-" "-" "-" "-"

    for high in "${high_weights[@]}"; do
        run_case "$iteration" "static-777-high-$high" "static-777" "$high" "-" "-" "-"
        run_case "$iteration" "static-007-high-$high" "static-007" "$high" "-" "-" "-"
        run_case "$iteration" "static-770-high-$high" "static-770" "$high" "-" "-" "-"
        run_case "$iteration" "dynamic-007-high-$high" "dynamic-007" "$high" "-" "0" "-"
        run_case "$iteration" "dynamic-000-high-$high" "dynamic-000" "$high" "-" "-" "-"

        for medium in "${medium_weights[@]}"; do
            if ((medium <= high)); then
                run_case \
                    "$iteration" "static-0${medium}${high}" "static-037" "$high" "$medium" "-" "-"
            fi
        done
    done

    for active in "${active_weights[@]}"; do
        run_case "$iteration" "dynamic-007-active-$active" "dynamic-007" "7" "-" "$active" "-"
    done

    # Compare this with static-007-high-7 to isolate request-barrier overhead.
    run_case "$iteration" "static-007-high-7-forced-sync" "static-007" "7" "-" "-" "1"
done

printf '\n== MPFE benchmark summary ==\n' | tee -a "$TEXT_LOG"
printf 'Text log: %s\nJSONL results: %s\n' "$TEXT_LOG" "$JSONL_LOG" | tee -a "$TEXT_LOG"
if ((${#failures[@]})); then
    printf 'FAILED (%s): %s\n' "${#failures[@]}" "${failures[*]}" | tee -a "$TEXT_LOG" >&2
    exit 1
fi

printf 'PASSED: all benchmark configurations\n' | tee -a "$TEXT_LOG"
