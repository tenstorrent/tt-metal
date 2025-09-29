#!/bin/bash

# Export environment variables
export TT_METAL_ENV=dev
export TT_METAL_HOME=$PWD
export ARCH_NAME=wormhole_b0

# Set log file name (modify as needed)
LOG_FILE="test_output.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Starting command sequence (100 iterations)..."
echo "Environment: TT_METAL_ENV=$TT_METAL_ENV, TT_METAL_HOME=$TT_METAL_HOME, ARCH_NAME=$ARCH_NAME"
echo "========================================"

# Counters for tracking failures
TOTAL_ITERATIONS=100
CMD1_FAILURES=0
CMD2_FAILURES=0

# Loop 100 times
for i in $(seq 1 $TOTAL_ITERATIONS); do
    echo ""
    echo "=== ITERATION $i/$TOTAL_ITERATIONS ==="
    echo "----------------------------------------"

    # Command 1: Reset GLX (with one retry on failure)
    echo "Running: tt-smi -glx_reset"
    tt-smi -glx_reset > /dev/null 2>&1
    CMD1_STATUS=$?

    if [ $CMD1_STATUS -ne 0 ]; then
        echo -e "${YELLOW}WARNING: tt-smi -glx_reset failed with exit code $CMD1_STATUS${NC}"
        echo "Retrying tt-smi -glx_reset..."
        sleep 1
        tt-smi -glx_reset > /dev/null 2>&1
        CMD1_STATUS=$?

        if [ $CMD1_STATUS -ne 0 ]; then
            echo -e "${RED}WARNING: tt-smi -glx_reset failed again with exit code $CMD1_STATUS${NC}"
            CMD1_FAILURES=$((CMD1_FAILURES + 1))
        else
            echo -e "${GREEN}tt-smi -glx_reset succeeded on retry${NC}"
        fi
    else
        echo -e "${GREEN}tt-smi -glx_reset completed successfully${NC}"
    fi

    echo "----------------------------------------"

    # Command 2: Run unit tests
    echo "Running: ./build_Release/test/tt_metal/unit_tests_dispatch --gtest_filter=\"*ShardedBufferLargeDRAMReadWrites\""
    ./build_Release/test/tt_metal/unit_tests_dispatch --gtest_filter="*ShardedBufferLargeDRAMReadWrites" 2>&1 | tee -a "$LOG_FILE"
    CMD2_STATUS=${PIPESTATUS[0]}

    if [ $CMD2_STATUS -ne 0 ]; then
        echo -e "${RED}WARNING: unit_tests_dispatch failed with exit code $CMD2_STATUS${NC}"
        echo -e "${YELLOW}Check log file: $LOG_FILE${NC}"
        CMD2_FAILURES=$((CMD2_FAILURES + 1))
    else
        echo -e "${GREEN}unit_tests_dispatch completed successfully${NC}"
    fi

    echo "----------------------------------------"
    echo -e "Iteration $i complete. Failures so far: tt-smi=${CMD1_FAILURES}, tests=${CMD2_FAILURES}"
done

# Final Summary
echo ""
echo "========================================"
echo "ALL ITERATIONS COMPLETE"
echo "========================================"
echo "Total iterations: $TOTAL_ITERATIONS"
echo -e "tt-smi failures: ${CMD1_FAILURES}"
echo -e "unit_tests_dispatch failures: ${CMD2_FAILURES}"

if [ $CMD1_FAILURES -gt 0 ] || [ $CMD2_FAILURES -gt 0 ]; then
    echo -e "${RED}Some commands failed during execution!${NC}"
    echo -e "${YELLOW}Check log file: $LOG_FILE${NC}"
    exit 1
else
    echo -e "${GREEN}All commands completed successfully across all iterations!${NC}"
    exit 0
fi
