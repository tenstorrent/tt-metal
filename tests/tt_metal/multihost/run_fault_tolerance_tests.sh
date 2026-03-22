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

"$MPIRUN" --with-ft ulfm -np 8 "$TEST_BIN" --gtest_filter=FaultTolerance.ShrinkAfterRankFailure || fail=$((fail + 1))
"$MPIRUN" --with-ft ulfm -np 8 "$TEST_BIN" --gtest_filter=FaultTolerance.DisableBrokenBlock || fail=$((fail + 1))
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.AgreeConsensus || fail=$((fail + 1))
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.FailurePolicySwitching || fail=$((fail + 1))

exit $fail
