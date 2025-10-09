#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Helper script to run the allocation tracking test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  Allocation Tracking Test Runner${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""

# Check if allocation server is built
if [ ! -f "allocation_server_poc" ]; then
    echo -e "${YELLOW}⚠ Allocation server not built. Building...${NC}"
    ./build_allocation_server.sh
    echo ""
fi

# Check if server is already running
if pgrep -x "allocation_serv" > /dev/null; then
    echo -e "${GREEN}✓ Allocation server is already running${NC}"
else
    echo -e "${BLUE}Starting allocation server in background...${NC}"
    ./allocation_server_poc > server_test.log 2>&1 &
    SERVER_PID=$!
    sleep 1

    if kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
    else
        echo -e "${RED}✗ Server failed to start${NC}"
        cat server_test.log
        exit 1
    fi
fi

# Check if socket exists
if [ -S "/tmp/tt_allocation_server.sock" ]; then
    echo -e "${GREEN}✓ Server socket exists${NC}"
else
    echo -e "${RED}✗ Server socket not found${NC}"
    exit 1
fi

echo ""
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  Running TTNN Test${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}💡 TIP: Open another terminal and run:${NC}"
echo -e "${BLUE}   ./allocation_monitor_client -r 500${NC}"
echo -e "${BLUE}   to see real-time memory updates!${NC}"
echo ""
echo -e "${YELLOW}Press Enter to start the test (or Ctrl+C to cancel)...${NC}"
read

# Enable tracking
export TT_ALLOC_TRACKING_ENABLED=1

echo -e "${GREEN}✓ Tracking enabled: TT_ALLOC_TRACKING_ENABLED=1${NC}"
echo ""

# Run the test
python test_ttnn_allocations.py

echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  Test Complete!${NC}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Server log saved to: server_test.log${NC}"
echo -e "${BLUE}Check for allocation messages: grep 'Allocated' server_test.log${NC}"
echo ""
echo -e "${YELLOW}To stop the server:${NC}"
echo -e "${YELLOW}  killall allocation_server_poc${NC}"
echo ""
