#!/bin/bash
# Stress test loop for post-commit tests. Runs for ~23.5h to allow next scheduled run to kick off.
# Usage: ./tests/scripts/run_stress_post_commit_tests.sh [fast|slow]
#   fast: run_python_api_unit_tests.sh + run_cpp_unit_tests.sh
#   slow: run_cpp_fd2_tests.sh + run_cpp_unit_tests.sh

set -eo pipefail

dispatch_mode=${1:-fast}

max_duration=84600
iter=1
cur_duration=0
expected_duration=0

while [ $expected_duration -lt $max_duration ]; do
    echo "Info: [stress] Doing iteration $iter"
    start_time=$(date +%s%N)
    if [[ $dispatch_mode == "slow" ]]; then
        ./tests/scripts/run_cpp_fd2_tests.sh
        ./tests/scripts/run_cpp_unit_tests.sh
    else
        ./tests/scripts/run_python_api_unit_tests.sh
        ./tests/scripts/run_cpp_unit_tests.sh
    fi
    end_time=$(date +%s%N)
    elapsed=$(( (end_time - start_time) / 1000000000 ))
    cur_duration=$((cur_duration + elapsed))
    avg_duration=$((cur_duration / iter))
    expected_duration=$((cur_duration + avg_duration))
    iter=$((iter + 1))
    echo "Info: [stress] expected elapsed time $expected_duration, elapsed time $cur_duration, avg iteration time $avg_duration"
done
