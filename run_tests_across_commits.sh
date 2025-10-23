#!/bin/bash

# Script to run tests across multiple commits with timing
# Usage: ./run_tests_across_commits.sh

set -e

PYTHON_BIN="./python_env/bin/python"
TEST_FILE="models/demos/gemma3/tests/test_perf_vision_cross_attention_transformer.py"
RESULTS_FILE="test_results_$(date +%Y%m%d_%H%M%S).txt"
TIMEOUT_SECONDS=600  # 10 minutes timeout per test

# Environment variables from launch.json
export HF_MODEL="google/gemma-3-27b-it"

# Commits to test
COMMIT1="mstojko/better_gather"
COMMIT2="69a5ddb1a1"

echo "Test Results - $(date)" | tee "$RESULTS_FILE"
echo "=======================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Function to run test with timeout
run_test_with_timeout() {
    local commit=$1
    local run_number=$2
    local start_time
    local end_time
    local duration
    local status

    # Create safe filename by replacing / with _
    local safe_commit=$(echo "$commit" | sed 's/\//_/g')

    echo "----------------------------------------" | tee -a "$RESULTS_FILE"
    echo "Commit: $commit - Run #$run_number" | tee -a "$RESULTS_FILE"
    echo "Started at: $(date)" | tee -a "$RESULTS_FILE"

    start_time=$(date +%s)

    # Run test with timeout
    if timeout $TIMEOUT_SECONDS $PYTHON_BIN -m pytest "$TEST_FILE" -x > "test_output_${safe_commit}_run${run_number}.log" 2>&1; then
        status="PASSED"
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            status="TIMEOUT (killed after ${TIMEOUT_SECONDS}s)"
            echo "Test timed out, killing any remaining processes..." | tee -a "$RESULTS_FILE"
        else
            status="FAILED (exit code: $exit_code)"
        fi
    fi

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "Ended at: $(date)" | tee -a "$RESULTS_FILE"
    echo "Duration: ${duration} seconds" | tee -a "$RESULTS_FILE"
    echo "Status: $status" | tee -a "$RESULTS_FILE"
    echo "Log file: test_output_${safe_commit}_run${run_number}.log" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"

    # Reset device
    echo "Running tt-smi -r..." | tee -a "$RESULTS_FILE"
    tt-smi -r || true
    sleep 5
}

# Save current branch/commit
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Original branch: $ORIGINAL_BRANCH" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Test Commit 1
echo "========================================" | tee -a "$RESULTS_FILE"
echo "Testing Commit 1: $COMMIT1" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
git checkout "$COMMIT1"
git log -1 --oneline | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

run_test_with_timeout "$COMMIT1" 1
run_test_with_timeout "$COMMIT1" 2

# Test Commit 2
echo "========================================" | tee -a "$RESULTS_FILE"
echo "Testing Commit 2: $COMMIT2" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
git checkout "$COMMIT2"
git log -1 --oneline | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

run_test_with_timeout "$COMMIT2" 1
run_test_with_timeout "$COMMIT2" 2

# Return to original branch
echo "========================================" | tee -a "$RESULTS_FILE"
echo "Returning to original branch: $ORIGINAL_BRANCH" | tee -a "$RESULTS_FILE"
git checkout "$ORIGINAL_BRANCH"

echo "" | tee -a "$RESULTS_FILE"
echo "=======================================" | tee -a "$RESULTS_FILE"
echo "All tests completed!" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Summary
echo "SUMMARY:" | tee -a "$RESULTS_FILE"
echo "--------" | tee -a "$RESULTS_FILE"
grep -E "(Commit:|Duration:|Status:)" "$RESULTS_FILE" | tail -20
