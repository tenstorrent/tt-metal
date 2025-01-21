#!/bin/bash

export TT_METAL_DEVICE_PROFILER="1"

TOTAL_TESTS=150
TIME_LIMIT=300

# for each of the tests, reset board with tt-smi and then run it in its own
for (( TEST_ID=0 ; TEST_ID<$TOTAL_TESTS ; TEST_ID++ ));
do
    echo "test #"$TEST_ID

    timeout $TIME_LIMIT tt-smi -r 0 || exit 1

    export SKIP_COUNT=$TEST_ID
    # echo "SKIP_COUNT="$SKIP_COUNT
    timeout $TIME_LIMIT pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf || echo SKIPPED $TEST_ID >> /proj_sw/user_dev/rdjogo/work/blackhole/tt-metal/generated/matmul_2d_host_perf_report.csv
done
