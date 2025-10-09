#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test script to diagnose why matmul_multicore_reuse allocations are not tracked

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Diagnosing Why matmul_multicore_reuse Allocations Are Not Tracked${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Kill old server
echo -e "${YELLOW}Step 1: Cleaning up old allocation server...${NC}"
pkill -f allocation_server_poc || true
sleep 1

# Step 2: Start new server
echo -e "${YELLOW}Step 2: Starting allocation server...${NC}"
./allocation_server_poc > /tmp/alloc_server.log 2>&1 &
SERVER_PID=$!
sleep 2

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Failed to start allocation server!${NC}"
    cat /tmp/alloc_server.log
    exit 1
fi
echo -e "${GREEN}✓ Server running (PID: $SERVER_PID)${NC}"

# Step 3: Test GraphTracker status
echo -e "\n${YELLOW}Step 3: Checking GraphTracker status...${NC}"
export TT_ALLOC_TRACKING_ENABLED=1
cd /home/tt-metal-apv/build/programming_examples

echo -e "${BLUE}Running test_graphtracker_check...${NC}"
./test_graphtracker_check > /tmp/graphtracker_test.log 2>&1 &
TEST_PID=$!

# Monitor server output for 6 seconds
echo -e "${BLUE}Monitoring server for 6 seconds...${NC}"
sleep 6

# Check if test is still running
if kill -0 $TEST_PID 2>/dev/null; then
    echo -e "${YELLOW}Test still running, waiting...${NC}"
    wait $TEST_PID
fi

# Display test output
echo -e "\n${BLUE}Test Output:${NC}"
cat /tmp/graphtracker_test.log

# Display server log
echo -e "\n${BLUE}Server Log (last 20 lines):${NC}"
tail -20 /tmp/alloc_server.log

# Check if allocation was tracked
if grep -q "ALLOC" /tmp/alloc_server.log; then
    echo -e "\n${GREEN}✓ SUCCESS: Allocations ARE being tracked!${NC}"
    echo -e "${GREEN}  The issue with matmul_multicore_reuse must be something else.${NC}"
else
    echo -e "\n${RED}✗ PROBLEM: No allocations tracked!${NC}"
    echo -e "${RED}  This confirms GraphTracker or another mechanism is bypassing the allocator.${NC}"
fi

# Step 4: Cleanup
echo -e "\n${YELLOW}Step 4: Cleaning up...${NC}"
kill $SERVER_PID 2>/dev/null || true
sleep 1

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Diagnosis Complete${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📖 For detailed explanation, see:${NC}"
echo -e "   ${SCRIPT_DIR}/WHY_MATMUL_NOT_TRACKED.md"
echo ""
echo -e "${YELLOW}🔧 To fix this issue, we need to:${NC}"
echo -e "   1. Add tracking to GraphTracker::track_allocate()"
echo -e "   2. Add tracking to Program::allocate_circular_buffers()"
echo -e "   3. Rebuild tt-metal"
echo ""
