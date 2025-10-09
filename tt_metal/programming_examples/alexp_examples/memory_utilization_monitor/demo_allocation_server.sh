#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Automated demo of allocation server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Allocation Server POC - Automated Demo${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Check if binaries exist
if [ ! -f "allocation_server_poc" ] || [ ! -f "allocation_client_demo" ] || [ ! -f "allocation_monitor_client" ]; then
    echo -e "${YELLOW}⚠ Binaries not found. Building...${NC}"
    ./build_allocation_server.sh
    echo ""
fi

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    rm -f /tmp/tt_allocation_server.sock
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start server in background
echo -e "${BLUE}Starting allocation server...${NC}"
./allocation_server_poc > server.log 2>&1 &
SERVER_PID=$!
sleep 1

# Check if server started successfully
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Server failed to start${NC}"
    cat server.log
    exit 1
fi
echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"

# Start monitor in background
echo -e "${BLUE}Starting monitor...${NC}"
./allocation_monitor_client -r 500 > monitor.log 2>&1 &
MONITOR_PID=$!
sleep 1

# Check if monitor started successfully
if ! kill -0 $MONITOR_PID 2>/dev/null; then
    echo -e "${RED}✗ Monitor failed to start${NC}"
    cat monitor.log
    cleanup
fi
echo -e "${GREEN}✓ Monitor started (PID: $MONITOR_PID)${NC}"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Demo Running - Watch the logs!${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Server log:  tail -f $SCRIPT_DIR/server.log"
echo -e "Monitor log: tail -f $SCRIPT_DIR/monitor.log"
echo ""
echo -e "${BLUE}Running client demo...${NC}"
echo ""

# Run client demo
./allocation_client_demo

echo ""
echo -e "${GREEN}✓ Client demo complete${NC}"
echo ""
echo -e "${YELLOW}Waiting 5 seconds for you to review the logs...${NC}"
sleep 5

# Show final server stats
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Final Server Statistics:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
tail -n 20 server.log

echo ""
echo -e "${GREEN}✅ Demo complete!${NC}"
echo ""
echo -e "To run manually in separate terminals:"
echo -e "  Terminal 1: ./allocation_server_poc"
echo -e "  Terminal 2: ./allocation_monitor_client -r 500"
echo -e "  Terminal 3: ./allocation_client_demo"
echo ""

cleanup
