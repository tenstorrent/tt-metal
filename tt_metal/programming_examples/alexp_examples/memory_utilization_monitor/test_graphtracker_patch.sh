#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test script to validate the GraphTracker allocation tracking patch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}  Testing GraphTracker Allocation Tracking Patch${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if patch is applied
echo -e "${CYAN}Checking if patch is applied...${NC}"
if grep -q "AllocationClient::report_allocation" /home/tt-metal-apv/tt_metal/graph/graph_tracking.cpp; then
    echo -e "${GREEN}✓ Patch is applied!${NC}"
else
    echo -e "${RED}✗ Patch is NOT applied!${NC}"
    echo -e "${YELLOW}Run this first:${NC}"
    echo -e "  See: ${SCRIPT_DIR}/PATCH_GRAPHTRACKER_TRACKING.md"
    exit 1
fi

# Check if code is built
echo -e "\n${CYAN}Checking if code is built...${NC}"
if [ ! -f "/home/tt-metal-apv/build/lib/libtt_metal.so" ]; then
    echo -e "${YELLOW}⚠ libtt_metal.so not found. Building...${NC}"
    cd /home/tt-metal-apv
    ./build_metal.sh
fi

# Step 1: Kill old server
echo -e "\n${YELLOW}Step 1: Cleaning up old allocation server...${NC}"
pkill -f allocation_server_poc || true
sleep 1

# Step 2: Start new server
echo -e "${YELLOW}Step 2: Starting allocation server...${NC}"
./allocation_server_poc > /tmp/alloc_server_patch_test.log 2>&1 &
SERVER_PID=$!
sleep 2

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Failed to start allocation server!${NC}"
    cat /tmp/alloc_server_patch_test.log
    exit 1
fi
echo -e "${GREEN}✓ Server running (PID: $SERVER_PID)${NC}"

# Function to run a test
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected_pattern="$3"

    echo -e "\n${BOLD}${CYAN}═══ Test: ${test_name} ═══${NC}"

    # Clear server log
    > /tmp/alloc_server_patch_test.log

    # Run test
    echo -e "${BLUE}Running: ${test_cmd}${NC}"
    export TT_ALLOC_TRACKING_ENABLED=1
    eval "$test_cmd" > /tmp/test_output.log 2>&1 &
    TEST_PID=$!

    # Wait for test
    sleep 5
    if kill -0 $TEST_PID 2>/dev/null; then
        echo -e "${YELLOW}Test still running, waiting...${NC}"
        wait $TEST_PID || true
    fi

    # Check results
    if grep -q "$expected_pattern" /tmp/alloc_server_patch_test.log; then
        echo -e "${GREEN}✓ PASS: Found '$expected_pattern' in server log${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL: Did not find '$expected_pattern' in server log${NC}"
        echo -e "${YELLOW}Last 10 lines of server log:${NC}"
        tail -10 /tmp/alloc_server_patch_test.log
        return 1
    fi
}

# Test 1: Simple allocation test
run_test \
    "Simple Buffer Allocation" \
    "cd /home/tt-metal-apv/build/programming_examples && ./test_tracking_cpp" \
    "ALLOC"

# Test 2: GraphTracker check
run_test \
    "GraphTracker Status Check" \
    "cd /home/tt-metal-apv/build/programming_examples && ./test_graphtracker_check" \
    "ALLOC"

# Test 3: Mesh allocation (C++)
run_test \
    "Mesh Allocation (C++)" \
    "cd /home/tt-metal-apv/build/programming_examples && ./test_mesh_allocation_cpp" \
    "Device 0.*ALLOC"

# Test 4: Matmul (the big one!)
echo -e "\n${BOLD}${CYAN}═══ Test: Matmul Multicore Reuse (THE BIG TEST!) ═══${NC}"
echo -e "${YELLOW}This test previously FAILED - allocations were not tracked${NC}"
echo -e "${YELLOW}If this passes, the patch is working!${NC}"

# Clear server log
> /tmp/alloc_server_patch_test.log

# Run matmul
echo -e "${BLUE}Running: matmul_multicore_reuse${NC}"
export TT_ALLOC_TRACKING_ENABLED=1
cd /home/tt-metal-apv/build/programming_examples
timeout 30 ./matmul_multicore_reuse > /tmp/matmul_output.log 2>&1 || true

# Check for allocations
sleep 2
ALLOC_COUNT=$(grep -c "ALLOC" /tmp/alloc_server_patch_test.log || echo "0")
DRAM_COUNT=$(grep -c "DRAM" /tmp/alloc_server_patch_test.log || echo "0")
L1_COUNT=$(grep -c "L1" /tmp/alloc_server_patch_test.log || echo "0")

echo -e "\n${BOLD}Results:${NC}"
echo -e "  Total allocations: ${ALLOC_COUNT}"
echo -e "  DRAM allocations: ${DRAM_COUNT}"
echo -e "  L1 allocations: ${L1_COUNT}"

if [ "$ALLOC_COUNT" -gt "0" ]; then
    echo -e "\n${GREEN}${BOLD}✓✓✓ SUCCESS! Matmul allocations are now tracked! ✓✓✓${NC}"
    echo -e "${GREEN}The patch is working correctly!${NC}"
else
    echo -e "\n${RED}${BOLD}✗✗✗ FAILURE! No allocations tracked for matmul ✗✗✗${NC}"
    echo -e "${RED}The patch may not be working correctly.${NC}"
    echo -e "\n${YELLOW}Server log (last 20 lines):${NC}"
    tail -20 /tmp/alloc_server_patch_test.log
fi

# Cleanup
echo -e "\n${YELLOW}Cleaning up...${NC}"
kill $SERVER_PID 2>/dev/null || true
sleep 1

echo -e "\n${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}  Test Complete${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📖 For detailed patch information, see:${NC}"
echo -e "   ${SCRIPT_DIR}/PATCH_GRAPHTRACKER_TRACKING.md"
echo ""

if [ "$ALLOC_COUNT" -gt "0" ]; then
    echo -e "${GREEN}${BOLD}🎉 All tests passed! The patch is working! 🎉${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}⚠ Some tests failed. Check the logs above. ⚠${NC}"
    exit 1
fi
