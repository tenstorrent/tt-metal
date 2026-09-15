#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Exercise every Device-side Tensor Prefetcher MPFE policy in a fresh process.
# The contention test byte-validates the first and final prefetched layers and
# stops the prefetcher, which also checks restoration to hardware defaults.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PYTHON="${PYTHON:-python3}"
BENCH_TRACE_REPEATS="${BENCH_TRACE_REPEATS:-2}"
TEST="tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py::test_mpfe_priority_contention"

cd "$TT_METAL_HOME"

# name|policy|high|medium|active; "-" leaves that setting unset.
# Besides every named policy, cover the no-environment production default and
# all three independently configurable weight inputs.
CASES=(
    "production-default|-|-|-|-"
    "dynamic-007-custom|dynamic-007|5|-|2"
    "dynamic-000|dynamic-000|5|-|-"
    "static-000|static-000|7|-|-"
    "static-777|static-777|5|-|-"
    "static-007|static-007|5|-|-"
    "static-037|static-037|5|3|-"
    "static-770|static-770|5|-|-"
)

failures=()
for spec in "${CASES[@]}"; do
    IFS="|" read -r name policy high medium active <<< "$spec"
    printf '\n== MPFE sanity: %s ==\n' "$name"

    env_args=(
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_MEDIUM_WEIGHT
        -u TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ACTIVE_WEIGHT
        -u TT_METAL_SLOW_DISPATCH_MODE
        "ARCH_NAME=blackhole"
        "BENCH_TRACE_REPEATS=$BENCH_TRACE_REPEATS"
        "PYTHONPATH=$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
    )
    [[ "$policy" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_PRIORITY_POLICY=$policy")
    [[ "$high" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_HIGH_WEIGHT=$high")
    [[ "$medium" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_MEDIUM_WEIGHT=$medium")
    [[ "$active" == "-" ]] || env_args+=("TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ACTIVE_WEIGHT=$active")

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
