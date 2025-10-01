#!/bin/bash

# Script to run fabric test up to 5 times or until failure
# Usage: ./run_fabric_test.sh

set -e  # Exit on any error

# Configuration
TT_METAL_HOME="/localdev/jhai/tt-metal"
MAX_RUNS=15
TEST_CONFIG="${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_6U_galaxy_quick.yaml"
TEST_BINARY="${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting fabric test runner...${NC}"
echo "Test binary: ${TEST_BINARY}"
echo "Test config: ${TEST_CONFIG}"
echo "Max runs: ${MAX_RUNS}"
echo ""

# Check if test binary exists
if [[ ! -f "${TEST_BINARY}" ]]; then
    echo -e "${RED}Error: Test binary not found at ${TEST_BINARY}${NC}"
    echo "Please build the project first."
    exit 1
fi

# Check if test config exists
if [[ ! -f "${TEST_CONFIG}" ]]; then
    echo -e "${RED}Error: Test config not found at ${TEST_CONFIG}${NC}"
    exit 1
fi

# Run the test up to MAX_RUNS times
for run in $(seq 1 ${MAX_RUNS}); do
    echo -e "${YELLOW}=== Run ${run}/${MAX_RUNS} ===${NC}"

    # Set environment variables and run the command
    if TT_METAL_HOME="${TT_METAL_HOME}" \
       TT_METAL_CLEAR_L1=1 \
       "${TEST_BINARY}" --test_config "${TEST_CONFIG}"; then

        echo -e "${GREEN}✓ Run ${run} completed successfully${NC}"
        echo ""
    else
        exit_code=$?
        echo -e "${RED}✗ Run ${run} failed with exit code ${exit_code}${NC}"
        echo -e "${RED}Stopping execution due to failure.${NC}"
        exit ${exit_code}
    fi
done

echo -e "${GREEN}All ${MAX_RUNS} runs completed successfully!${NC}"
