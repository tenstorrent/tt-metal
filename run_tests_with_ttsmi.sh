#!/bin/bash

export TT_METAL_DEVICE_PROFILER="1"

FORCE_AICLK=0
GO_BUSY=0

TOTAL_TESTS=150
TIME_LIMIT=300
RESET_BOARD=1

# Clean up directory with profiler data
rm -rf generated

# for each of the tests, reset board with tt-smi and then run it in its own
for (( TEST_ID=0 ; TEST_ID<$TOTAL_TESTS ; TEST_ID++ ));
do
    echo "test #"$TEST_ID
    echo "RESET_BOARD="$RESET_BOARD

    if [[ "RESET_BOARD" -eq 1 ]]; then
        timeout $TIME_LIMIT tt-smi -r 0 || exit 1
        RESET_BOARD=0

        if [[ "FORCE_AICLK" -eq 1]]; then
            ./bh_force_aiclk 1350
        fi

        if [[ "GO_BUSY" -eq 1]]; then
            ./go_busy --message go_busy
        fi
    fi

    export SKIP_COUNT=$TEST_ID
    timeout $TIME_LIMIT pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf

    if [ $? -eq 124 ]; then
        echo SKIPPED $TEST_ID >> $TT_METAL_HOME/generated/matmul_2d_host_perf_report.csv
        RESET_BOARD=1
    fi
done
