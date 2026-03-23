#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Purpose: Run fault tolerance tests for tt_metal

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow test overrides for shell-level regression coverage.
MPIRUN="${MPIRUN:-$SCRIPT_DIR/mpirun_wrapper.sh}"

TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)}"
TEST_BIN="${TEST_BIN:-$TT_METAL_HOME/build/test/tt_metal/fault_tolerance_tests}"

if [ ! -x "$TEST_BIN" ]; then
    echo "ERROR: fault_tolerance_tests binary not found at $TEST_BIN" >&2
    echo "Build with: cmake --build build --target fault_tolerance_tests" >&2
    exit 1
fi

fail=0

_run_ulfm_test() {
    local expected_test="$1"
    shift

    # mpirun exits non-zero when ranks are intentionally killed (e.g. ShrinkAfterRankFailure,
    # DisableBrokenBlock). Treat those as pass only when the expected GTest markers show the
    # test really ran to completion; launcher failures or truncated output must still fail.
    local tmpout
    local cmd_status=0
    tmpout=$(mktemp)

    if "$@" 2>&1 | tee "$tmpout"; then
        cmd_status=0
    else
        cmd_status=${PIPESTATUS[0]}
    fi

    if [[ ! -s "$tmpout" ]]; then
        echo "ERROR: no output captured for ${expected_test}; launcher exited ${cmd_status}" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    if ! grep -Fq "[ RUN      ] ${expected_test}" "$tmpout"; then
        echo "ERROR: missing GTest start marker for ${expected_test}; launcher exited ${cmd_status}" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    if grep -q '\[  FAILED  \]' "$tmpout"; then
        echo "ERROR: GTest failures detected in ${expected_test}; launcher exited ${cmd_status}" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    if ! grep -Fq "[       OK ] ${expected_test}" "$tmpout" && ! grep -Fq "[  PASSED  ]" "$tmpout"; then
        echo "ERROR: missing successful GTest completion markers for ${expected_test}; launcher exited ${cmd_status}" >&2
        fail=$((fail + 1))
    elif [[ $cmd_status -ne 0 ]]; then
        echo "INFO: ${expected_test} passed (GTest markers OK) despite mpirun exit ${cmd_status} (expected for rank-kill tests)" >&2
    fi

    rm -f "$tmpout"
}

_run_ulfm_test "FaultTolerance.ShrinkAfterRankFailure" \
    "$MPIRUN" --with-ft ulfm -np 8 "$TEST_BIN" --gtest_filter=FaultTolerance.ShrinkAfterRankFailure
_run_ulfm_test "FaultTolerance.DisableBrokenBlock" \
    "$MPIRUN" --with-ft ulfm -np 8 "$TEST_BIN" --gtest_filter=FaultTolerance.DisableBrokenBlock
_run_ulfm_test "FaultTolerance.AgreeConsensus" \
    "$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.AgreeConsensus
_run_ulfm_test "FaultTolerance.FailurePolicySwitching" \
    "$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.FailurePolicySwitching

exit $fail
