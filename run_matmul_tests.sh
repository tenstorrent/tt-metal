#!/bin/bash

# Run minimal matmul correctness and performance tests
# Outputs JSON: {"accuracy_passed": bool, "performance_ns": float}

TEST_FILE="tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul_optimization.py"
TIMEOUT_SECONDS=20

# Function to reset device on timeout
reset_device() {
    echo "Timeout hit, resetting device..." >&2
    tt-smi -r
}

# Build before running tests (suppress output unless error)
build_output=$(./build_metal.sh 2>&1)
build_exit_code=$?

if [[ $build_exit_code -ne 0 ]]; then
    echo "Error: Build failed (exit code: $build_exit_code)" >&2
    echo "$build_output" >&2
    exit 1
fi

# Run correctness test and capture output with timeout
correctness_output=$(timeout ${TIMEOUT_SECONDS} pytest "${TEST_FILE}::test_correctness" 2>&1)
correctness_exit_code=$?

# Check for timeout (exit code 124)
if [[ $correctness_exit_code -eq 124 ]]; then
    reset_device
    echo "Error: Correctness test timed out after ${TIMEOUT_SECONDS} seconds" >&2
    exit 1
fi

# Check if there was a non-accuracy error (compile error, import error, etc.)
# pytest exit codes: 0=passed, 1=tests failed, 2=interrupted, 3=internal error, 4=usage error, 5=no tests collected
if [[ $correctness_exit_code -ge 2 ]]; then
    echo "Error: Test infrastructure failure (exit code: $correctness_exit_code)" >&2
    echo "$correctness_output" >&2
    exit 1
fi

# Check for collection errors or import errors in output
if echo "$correctness_output" | grep -qE "(ImportError|ModuleNotFoundError|SyntaxError|compilation failed|ERRORS)"; then
    echo "Error: Compile or import error detected" >&2
    echo "$correctness_output" >&2
    exit 1
fi

# Check if the test was even collected and run
if echo "$correctness_output" | grep -qE "no tests ran|collected 0 items"; then
    echo "Error: No tests were collected" >&2
    echo "$correctness_output" >&2
    exit 1
fi

# Determine if accuracy passed
# Exit code 0 means passed, exit code 1 with AssertionError means accuracy failed
accuracy_passed=false
if [[ $correctness_exit_code -eq 0 ]]; then
    accuracy_passed=true
elif [[ $correctness_exit_code -eq 1 ]]; then
    # Check if it was an assertion error (accuracy failure) vs other runtime error
    if echo "$correctness_output" | grep -qE "AssertionError|assert check_result"; then
        accuracy_passed=false
    else
        # Some other runtime error occurred
        echo "Error: Runtime error during correctness test" >&2
        echo "$correctness_output" >&2
        exit 1
    fi
fi

# Run performance test and capture output with timeout
performance_output=$(timeout ${TIMEOUT_SECONDS} pytest "${TEST_FILE}::test_performance" 2>&1)
performance_exit_code=$?

# Check for timeout (exit code 124)
if [[ $performance_exit_code -eq 124 ]]; then
    reset_device
    echo "Error: Performance test timed out after ${TIMEOUT_SECONDS} seconds" >&2
    exit 1
fi

# Check for performance test errors
if [[ $performance_exit_code -ne 0 ]]; then
    echo "Error: Performance test failed (exit code: $performance_exit_code)" >&2
    echo "$performance_output" >&2
    exit 1
fi

# Extract performance_ns from the output
# The test logs: "Mean latency across N: XXXX"
performance_ns=$(echo "$performance_output" | grep -oP "Mean latency across \d+: \K[\d.]+")

if [[ -z "$performance_ns" ]]; then
    echo "Error: Could not extract performance_ns from output" >&2
    echo "$performance_output" >&2
    exit 1
fi

# Output JSON result
if [[ "$accuracy_passed" == "true" ]]; then
    echo "{\"accuracy_passed\": true, \"performance_ns\": ${performance_ns}}"
else
    echo "{\"accuracy_passed\": false, \"performance_ns\": ${performance_ns}}"
fi
