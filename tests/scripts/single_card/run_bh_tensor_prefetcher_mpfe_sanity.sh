#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Exercise representative static Device-side Tensor Prefetcher MPFE weights.
# The contention test byte-validates the first and final prefetched layers and
# stops the prefetcher, which also checks restoration to hardware defaults.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PYTHON="${PYTHON:-python3}"
BENCH_TRACE_REPEATS="${BENCH_TRACE_REPEATS:-2}"
TEST="tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention"

cd "$TT_METAL_HOME"

# name|free sender|NOC1 sender|ordinary;
# "-" leaves that setting unset.
CASES=(
    "production-default|-|-|-"
    "static-000|0|0|0"
    "static-014|0|1|4"
    "static-037|0|3|7"
    "static-777|7|7|7"
)

failures=()
for spec in "${CASES[@]}"; do
    IFS="|" read -r name free_weight noc1_weight ordinary_weight <<< "$spec"
    printf '\n== MPFE sanity: %s ==\n' "$name"

    env_args=(
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FREE_SENDER_WEIGHT
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_NOC1_SENDER_WEIGHT
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ORDINARY_WEIGHT
        -u TT_METAL_BENCHMARK_RESULT_JSONL
        -u TT_METAL_SLOW_DISPATCH_MODE
        "ARCH_NAME=blackhole"
        "BENCH_TRACE_REPEATS=$BENCH_TRACE_REPEATS"
        "PYTHONPATH=$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
    )
    [[ "$free_weight" == "-" ]] ||
        env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_FREE_SENDER_WEIGHT=$free_weight")
    [[ "$noc1_weight" == "-" ]] ||
        env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_NOC1_SENDER_WEIGHT=$noc1_weight")
    [[ "$ordinary_weight" == "-" ]] ||
        env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ORDINARY_WEIGHT=$ordinary_weight")

    if ! env "${env_args[@]}" "$PYTHON" -m pytest -sv "$TEST" "$@"; then
        failures+=("$name")
    fi
done

printf '\n== MPFE sanity summary ==\n'
if ((${#failures[@]})); then
    printf 'FAILED: %s\n' "${failures[*]}" >&2
    exit 1
fi

printf 'PASSED: %s configurations\n' "${#CASES[@]}"
