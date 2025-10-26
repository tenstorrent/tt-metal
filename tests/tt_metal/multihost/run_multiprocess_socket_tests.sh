#!/bin/bash

# TODO: These tests should be merged with other multi-process single host tests
# once we have those on CI.

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TT_METAL_HOME=$(cd $SCRIPT_DIR && git rev-parse --show-toplevel)

# Use the wrapper script
MPIRUN="$SCRIPT_DIR/mpirun_wrapper.sh"

cd $TT_METAL_HOME

#############################################
# Multi-Process Socket Tests                #
#############################################

echo "Running multi-process socket tests..."

$MPIRUN -n 2 --allow-run-as-root ./build/test/tt_metal/multi_host_socket_tests
