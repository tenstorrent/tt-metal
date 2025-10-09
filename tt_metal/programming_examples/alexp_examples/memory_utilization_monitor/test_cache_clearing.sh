#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test script to demonstrate program cache clearing functionality

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Program Cache Clearing Test                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if allocation server is running
if ! pgrep -x allocation_server_poc > /dev/null; then
    echo -e "${RED}❌ Allocation server is not running!${NC}"
    echo -e "${YELLOW}Starting allocation server in background...${NC}"
    ./allocation_server_poc > /tmp/allocation_server.log 2>&1 &
    SERVER_PID=$!
    sleep 2

    if ! pgrep -x allocation_server_poc > /dev/null; then
        echo -e "${RED}Failed to start allocation server!${NC}"
        echo "Check /tmp/allocation_server.log for errors"
        exit 1
    fi

    echo -e "${GREEN}✓ Allocation server started (PID: $SERVER_PID)${NC}"
    KILL_SERVER=true
else
    echo -e "${GREEN}✓ Allocation server is already running${NC}"
    KILL_SERVER=false
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Running test_mesh_allocation.py with cache clearing...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Watch for:${NC}"
echo -e "  ${YELLOW}1.${NC} Tensor allocations (~4MB per device)"
echo -e "  ${YELLOW}2.${NC} Tensor deallocations"
echo -e "  ${YELLOW}3.${NC} ${GREEN}[NEW]${NC} Program cache clearing (~36KB per device)"
echo -e "  ${YELLOW}4.${NC} System buffer cleanup (~14MB per device)"
echo ""
echo -e "${YELLOW}Press Enter to start the test, or Ctrl+C to cancel${NC}"
read

# Export tracking environment variable
export TT_ALLOC_TRACKING_ENABLED=1

# Source environment setup
if [ -f "/home/tt-metal-apv/build_Release_tracy/env_vars_setup.sh" ]; then
    source /home/tt-metal-apv/build_Release_tracy/env_vars_setup.sh
elif [ -f "/home/tt-metal-apv/build/env_vars_setup.sh" ]; then
    source /home/tt-metal-apv/build/env_vars_setup.sh
else
    echo -e "${YELLOW}⚠  Warning: Could not find env_vars_setup.sh${NC}"
fi

# Run the test
echo -e "\n${GREEN}Starting test...${NC}\n"
python3 test_mesh_allocation.py

EXIT_CODE=$?

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Test Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Test passed successfully!${NC}"
    echo ""
    echo -e "${GREEN}You should have seen:${NC}"
    echo -e "  ${GREEN}•${NC} Tensor allocations (~4MB per device × 8 devices)"
    echo -e "  ${GREEN}•${NC} Distributed computations (add, matmul)"
    echo -e "  ${GREEN}•${NC} Tensor deallocations"
    echo -e "  ${GREEN}•${NC} ${YELLOW}Program cache cleared${NC} - freed ~36KB cached kernels per device"
    echo -e "  ${GREEN}•${NC} Device close - freed ~14MB system buffers per device"
    echo -e "  ${GREEN}•${NC} ${YELLOW}All memory returned to 0${NC}"
else
    echo -e "${RED}❌ Test failed with exit code $EXIT_CODE${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}What's the 36KB?${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "The 36KB per device is ${YELLOW}cached compiled programs${NC} (kernel binaries):"
echo -e "  • 12KB: Compiled kernels for ${YELLOW}ttnn.add${NC} operation"
echo -e "  • 24KB: Compiled kernels for ${YELLOW}ttnn.matmul${NC} operation"
echo ""
echo -e "This is ${GREEN}NOT a memory leak${NC} - it's a ${YELLOW}performance optimization!${NC}"
echo ""
echo -e "Why cache programs?"
echo -e "  • ${GREEN}10-100x faster${NC} repeated operations"
echo -e "  • Avoids expensive recompilation"
echo -e "  • Standard practice in JIT systems"
echo ""
echo -e "How to free cached programs:"
echo -e "  Python: ${YELLOW}mesh_device.disable_and_clear_program_cache()${NC}"
echo -e "  C++:    ${YELLOW}mesh_device->disable_and_clear_program_cache()${NC}"
echo ""
echo -e "${BLUE}See PROGRAM_CACHE_EXPLANATION.md for full details!${NC}"
echo ""

# Cleanup
if [ "$KILL_SERVER" = true ]; then
    echo -e "${YELLOW}Stopping allocation server...${NC}"
    kill $SERVER_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Server stopped${NC}"
fi

exit $EXIT_CODE
