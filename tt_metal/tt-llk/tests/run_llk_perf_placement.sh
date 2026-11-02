#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Stage 2 of the LLK perf gate: measure the tests in a node list with the TRISC code at 5 positions.
# Each position's reports go to perf_data/runs/p<shift>/. One compile session and one measuring session cover
# all positions: each test compiles its objects once, links them per position, and runs at every position in turn.
#
# Usage: LLK_PERF_NODE_LIST=<file> [LLK_STOP_AT=<epoch s>] run_llk_perf_placement.sh
# With LLK_STOP_AT, no test starts after that time; the tests it skips show as not measured.
set -euo pipefail

NODE_LIST="${LLK_PERF_NODE_LIST:?LLK_PERF_NODE_LIST must name the file of test node ids}"
NODE_LIST="$(realpath "$NODE_LIST")"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"
export LLK_PERF_RUN_TYPES="${LLK_PERF_RUN_TYPES:-L1_TO_L1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

if [[ ! -s "$NODE_LIST" ]]; then
  echo "The node list is empty; nothing to measure."
  exit 0
fi

TEMP_ROOT="${RUNNER_TEMP:-$(mktemp -d)}"
export LLK_OBJ_CACHE="$TEMP_ROOT/placement-objects"
mkdir -p "$LLK_OBJ_CACHE"
# Collect only the files of the listed tests: collecting the whole suite is most of a session's time.
mapfile -t FILES < <(cut -d: -f1 "$NODE_LIST" | sort -u)
PYTEST_EXTRA=(-q --override-ini=log_cli=false -m "perf and not accuracy" --timeout=120 --node-list="$NODE_LIST")
STATUS=0
export LLK_CODE_SHIFTS="0 4 20 36 52"
export RUNNER_TEMP="$TEMP_ROOT/placement"
# The first position's reports go to perf_data/runs/p0; PerfConfig writes the others to p<shift>.
export PERF_RUN_TAG="p0"
export LLK_WINDOW_HASH=1
STOP_AT="${LLK_STOP_AT:-}"
if [[ -n "$STOP_AT" ]]; then
  # Compiling takes about twice as long as measuring; it gets 60 % of what is left.
  export LLK_STOP_AT=$(( $(date +%s) + (STOP_AT - $(date +%s)) * 3 / 5 ))
fi
echo "stage2-time: compile start $(date +%s)"
pytest "${PYTEST_EXTRA[@]}" --compile-producer -n 10 --junitxml="pytest-report-placement-1-compile.xml" "${FILES[@]}" || STATUS=1
echo "stage2-time: compile end $(date +%s)"
[[ -n "$STOP_AT" ]] && export LLK_STOP_AT="$STOP_AT"
pytest "${PYTEST_EXTRA[@]}" --compile-consumer --dist loadgroup -n 15 --junitxml="pytest-report-placement-1-run.xml" "${FILES[@]}" || STATUS=1
echo "stage2-time: run end $(date +%s)"

# A test that failed at one position is reported as unclear; the comparison still runs.
exit "$STATUS"
