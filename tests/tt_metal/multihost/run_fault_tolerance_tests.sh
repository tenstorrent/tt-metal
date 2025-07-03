#!/bin/bash
# Purpose: Run fault tolerance tests for tt_metal

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the wrapper script
MPIRUN="$SCRIPT_DIR/mpirun_wrapper.sh"

TT_METAL_HOME=$(cd $SCRIPT_DIR && git rev-parse --show-toplevel)

$MPIRUN --with-ft ulfm -np 8 $TT_METAL_HOME/build/test/tt_metal/fault_tolerance_tests --gtest_filter=FaultTolerance.ShrinkAfterRankFailure
$MPIRUN --with-ft ulfm -np 8 $TT_METAL_HOME/build/test/tt_metal/fault_tolerance_tests --gtest_filter=FaultTolerance.DisableBrokenBlock
