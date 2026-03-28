#!/bin/bash
# Run doorbell latency experiment across delay values.
# Usage: ./tools/run_doorbell_latency_experiment.sh [num_tests]

set -euo pipefail

NUM_TESTS=${1:-500}
BIN=build/test/tt_metal/perf_microbenchmark/7_kernel_launch/test_kernel_launch
ARGS="--num-tests $NUM_TESTS --bypass-check"

if [ ! -x "$BIN" ]; then
    echo "Binary not found. Build with: cmake --build build --target test_kernel_launch"
    exit 1
fi

for DELAY in 0 100 400 800 1700 5000; do
    AVG=$(TT_DOORBELL_DELAY_NS=$DELAY $BIN $ARGS 2>&1 \
        | grep 'CSV_OUTPUT:ElapsedTime' \
        | sed 's/.*ElapsedTime(us)://; s/ .*//')
    printf "TT_DOORBELL_DELAY_NS=%5d  avg=%sus\n" "$DELAY" "$AVG"
done
