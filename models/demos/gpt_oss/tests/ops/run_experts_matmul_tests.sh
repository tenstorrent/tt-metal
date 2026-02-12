#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Helper script to run experts matmul unit tests
# This script provides convenient shortcuts for running the stress test and parameter sweep

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test file path
TEST_FILE="models/demos/gpt_oss/tests/ops/test_experts_matmul.py"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Experts Matmul Unit Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Function to print usage
usage() {
    echo "Usage: $0 [stress|sweep|both|help]"
    echo ""
    echo "Commands:"
    echo "  stress  - Run stress test (100 iterations)"
    echo "  sweep   - Run parameter sweep test"
    echo "  both    - Run both stress and sweep tests"
    echo "  help    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 stress          # Run stress test only"
    echo "  $0 sweep           # Run parameter sweep only"
    echo "  $0 both            # Run all tests"
    echo ""
}

# Function to run stress test
run_stress() {
    echo -e "${YELLOW}Running Stress Test...${NC}"
    echo ""
    HF_MODEL=/proj_sw/user_dev/gpt-oss-weights/gpt-oss-120b/ pytest "${TEST_FILE}::test_experts_matmul_stress" -v -s
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Stress test passed${NC}"
    else
        echo -e "${RED}✗ Stress test failed${NC}"
        exit 1
    fi
}

# Function to run parameter sweep
run_sweep() {
    echo -e "${YELLOW}Running Parameter Sweep...${NC}"
    echo ""
    HF_MODEL=/proj_sw/user_dev/gpt-oss-weights/gpt-oss-120b/ pytest "${TEST_FILE}::test_experts_matmul_param_sweep" -v -s
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Parameter sweep completed${NC}"
        echo ""
        echo -e "${GREEN}Results saved to: param_sweep_results/${NC}"

        # List result files if they exist
        if [ -d "param_sweep_results" ]; then
            echo -e "${YELLOW}Result files:${NC}"
            ls -lh param_sweep_results/*.json | tail -5
        fi
    else
        echo -e "${RED}✗ Parameter sweep failed${NC}"
        exit 1
    fi
}

# Main script logic
case "${1:-help}" in
    stress)
        run_stress
        ;;
    sweep)
        run_sweep
        ;;
    both)
        run_stress
        echo ""
        run_sweep
        ;;
    help)
        usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        echo ""
        usage
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All requested tests completed!${NC}"
echo -e "${GREEN}========================================${NC}"
