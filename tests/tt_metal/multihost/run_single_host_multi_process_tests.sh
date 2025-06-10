#!/bin/bash
# Purpose: Run single host multi-process tests for tt_metal

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the wrapper script
MPIRUN="$SCRIPT_DIR/mpirun_wrapper.sh"
TT_METAL_HOME=$(cd $SCRIPT_DIR && git rev-parse --show-toplevel)

$MPIRUN --with-ft ulfm -np 4 $TT_METAL_HOME/build/test/tt_metal/single_host_mp_tests
