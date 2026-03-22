#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Purpose: Run fault tolerance tests for tt_metal

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the wrapper script
MPIRUN="$SCRIPT_DIR/mpirun_wrapper.sh"

TT_METAL_HOME=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
TEST_BIN="$TT_METAL_HOME/build/test/tt_metal/fault_tolerance_tests"

if [ ! -x "$TEST_BIN" ]; then
    echo "ERROR: fault_tolerance_tests binary not found at $TEST_BIN" >&2
    echo "Build with: cmake --build build --target fault_tolerance_tests" >&2
    exit 1
fi

fail=0

_run_ulfm_test() {
    # mpirun exits non-zero when ranks are intentionally killed (e.g. ShrinkAfterRankFailure,
    # DisableBrokenBlock). Use GTest [  FAILED  ] lines to determine pass/fail rather than
    # relying on mpirun's exit code, which is a false positive for rank-kill tests.
    local tmpout
    tmpout=$(mktemp)
    { "$@" || true; } 2>&1 | tee "$tmpout"
    if grep -q '\[  FAILED  \]' "$tmpout"; then
        echo "ERROR: GTest failures detected in: $*" >&2
        fail=$((fail + 1))
    fi
    rm -f "$tmpout"
}

_run_ulfm_test "$MPIRUN" --with-ft ulfm -np 8 "$TEST_BIN" --gtest_filter=FaultTolerance.ShrinkAfterRankFailure
_run_ulfm_test "$MPIRUN" --with-ft ulfm -np 8 "$TEST_BIN" --gtest_filter=FaultTolerance.DisableBrokenBlock
_run_ulfm_test "$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.AgreeConsensus
_run_ulfm_test "$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.FailurePolicySwitching

exit $fail
