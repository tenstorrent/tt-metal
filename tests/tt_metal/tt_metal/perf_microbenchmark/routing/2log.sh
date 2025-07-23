#!/bin/bash

# Create log file name with current script PID
LOG_FILE="test_tt_fabric_$$.log"
echo "Log file created: ${LOG_FILE}"
# Run command and redirect all output to log file
./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
    --test_config /localdev/marvinmok/tt-metal/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_t3k_2x2.yaml |& tee ${LOG_FILE}
