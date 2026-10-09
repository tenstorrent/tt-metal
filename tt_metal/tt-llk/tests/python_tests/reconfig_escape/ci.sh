#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Script for the weekly reconfig sweep that runs in CI.
# It gathers a catalog of possible config space states that a kernel may leave behind, dedups,
# then sweeps across the testsuite by injecting state before kernel init to find failures.
# If a failure is found, it tries to reproduce it by running the potentially problematic pair
# back-to-back, and reports if that run fails. False positives are recorded but don't fail CI.
#
# It additionally synthesizes possible states from the gathered catalog as follows.
# Suppose you run kernels A, B and C. A's init will observe a clean slate, B's init will see
# A's residual, and C's init will see B U (A \ B).
# The depth to which this is done is configurable.
#
# Due to time constraints, only a subset of tests are run on a given week.
#
# Usage:
#   ./ci.sh --arch blackhole [--report-dir DIR] [--sample-per-test N] [--jobs N] [--timeout SECS]
#           [--splits N --group G] [--chain-depth N] [--chains-per-machine N]
#
# Exit codes: 0 = no escapes, 1 = at least one escape found, 2 = the sweep itself errored out.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../../.." && pwd)"   # tests/python_tests/reconfig_escape -> tt-llk root

ARCH=""
REPORT_DIR="$HERE/reports"
SAMPLE_PER_TEST=15
JOBS=8
TIMEOUT=90
SPLITS=""
GROUP=""
CHAIN_DEPTH=2
CHAINS_PER_MACHINE=40

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --sample-per-test) SAMPLE_PER_TEST="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --chain-depth) CHAIN_DEPTH="$2"; shift 2 ;;
        --chains-per-machine) CHAINS_PER_MACHINE="$2"; shift 2 ;;
        *) echo "reconfig_escape/ci.sh: unknown option $1" >&2; exit 4 ;;
    esac
done

if [[ -z "$ARCH" ]]; then
    echo "reconfig_escape/ci.sh: --arch blackhole is required" >&2
    exit 4
fi

(cd "$WORKTREE/tests" && ./setup_testing_env.sh)

rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR/catalog"

SPLIT_ARGS=()
[[ -n "$SPLITS" ]] && SPLIT_ARGS+=(--splits "$SPLITS" --group "${GROUP:-1}")
[[ -z "$SPLITS" ]] || echo ">> splits=${SPLITS} group=${GROUP:-1}"

TOTAL_CHAINS=$(( CHAINS_PER_MACHINE * ${SPLITS:-1} ))

echo ">> [1/4] catalog discovery"
python3 "$HERE/discover_catalog.py" \
    --worktree "$WORKTREE" --arch "$ARCH" \
    --out-dir "$REPORT_DIR/catalog" --manifest "$REPORT_DIR/manifest.json" \
    --sample-per-test "$SAMPLE_PER_TEST" --jobs "$JOBS" --compile-jobs "$JOBS" \
    --timeout "$TIMEOUT" \
    || { echo ">> catalog discovery did not complete" >&2; exit 2; }

echo ">> [2/4] pair sweep (depth 1, exhaustive)"
python3 "$HERE/pair_sweep.py" \
    --worktree "$WORKTREE" --arch "$ARCH" --manifest "$REPORT_DIR/manifest.json" \
    --out "$REPORT_DIR/findings_depth1.jsonl" --jobs "$JOBS" --timeout "$TIMEOUT" --skip-compile \
    "${SPLIT_ARGS[@]}" \
    || { echo ">> depth-1 pair sweep did not complete" >&2; exit 2; }

echo ">> [3/4] pair sweep (depth $CHAIN_DEPTH, $TOTAL_CHAINS chain(s) total across $((${SPLITS:-1})) machine(s))"
python3 "$HERE/pair_sweep.py" \
    --worktree "$WORKTREE" --arch "$ARCH" --manifest "$REPORT_DIR/manifest.json" \
    --out "$REPORT_DIR/findings_depth${CHAIN_DEPTH}.jsonl" --jobs "$JOBS" --timeout "$TIMEOUT" \
    --skip-compile --depth "$CHAIN_DEPTH" --chains "$TOTAL_CHAINS" \
    "${SPLIT_ARGS[@]}" \
    || { echo ">> depth-$CHAIN_DEPTH chain sweep did not complete" >&2; exit 2; }

cat "$REPORT_DIR/findings_depth1.jsonl" "$REPORT_DIR/findings_depth${CHAIN_DEPTH}.jsonl" \
    > "$REPORT_DIR/findings.jsonl"

echo ">> [4/4] report"
status=0
python3 "$HERE/report.py" \
    --jsonl "$REPORT_DIR/findings.jsonl" \
    --report-md "$REPORT_DIR/report.md" --junit "$REPORT_DIR/junit.xml" \
    || status=$?

exit "$status"
