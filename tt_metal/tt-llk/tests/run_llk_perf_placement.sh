#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Stage 2 of the LLK perf gate: measure the tests in a node list with the TRISC code at 5 positions.
# Each position's reports go to perf_data/runs/p<shift>/. Objects are compiled once and relinked per position.
#
# Usage: LLK_PERF_NODE_LIST=<file> run_llk_perf_placement.sh
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
PYTEST_EXTRA=(-q --override-ini=log_cli=false -m "perf and not accuracy" --timeout=60 --node-list "$NODE_LIST")
STATUS=0

for SHIFT in 0 4 20 36 52; do
  export LLK_CODE_SHIFT="$SHIFT"
  export RUNNER_TEMP="$TEMP_ROOT/placement-p$SHIFT"
  export PERF_RUN_TAG="p$SHIFT"
  if [[ "$SHIFT" == 0 ]]; then export LLK_WINDOW_HASH=1; else unset LLK_WINDOW_HASH; fi
  pytest "${PYTEST_EXTRA[@]}" --compile-producer -n 10 --junitxml="pytest-report-placement-p${SHIFT}-1-compile.xml" "${FILES[@]}" || STATUS=1
  pytest "${PYTEST_EXTRA[@]}" --compile-consumer --dist loadgroup -n 15 --junitxml="pytest-report-placement-p${SHIFT}-1-run.xml" "${FILES[@]}" || STATUS=1
done

# A test that failed at one position is reported as unclear; the comparison still runs.
exit "$STATUS"
