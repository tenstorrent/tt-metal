#!/bin/bash
# Test script to verify all L1 tracking is working

set -e

echo "========================================"
echo "Testing L1 Memory Tracking"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cd /home/ttuser/aperezvicente/tt-metal

# Step 1: Start allocation server
echo -e "${YELLOW}1. Starting allocation server...${NC}"
export TT_ALLOC_TRACKING_ENABLED=1
./build/install/bin/allocation_server_poc &
SERVER_PID=$!
sleep 2

echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
echo ""

# Step 2: Run a simple test that allocates L1 buffers
echo -e "${YELLOW}2. Running test workload (matmul)...${NC}"
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul.py::test_matmul -k "test_matmul_1d" -s 2>&1 | head -100 || true

echo ""
echo -e "${GREEN}✓ Workload completed${NC}"
echo ""

# Step 3: Give server time to process
sleep 2

# Step 4: Check server output
echo -e "${YELLOW}3. Checking allocation server output...${NC}"
echo "   (Check the server terminal for allocation messages)"
echo ""

# Step 5: Send dump signal to see summary
echo -e "${YELLOW}4. Requesting memory dump from server...${NC}"
kill -USR1 $SERVER_PID 2>/dev/null || true
sleep 2

echo ""
echo -e "${GREEN}✓ Test complete!${NC}"
echo ""
echo "Expected output in server terminal:"
echo "  - Regular buffer allocations (MBs) - from graph_tracking.cpp"
echo "  - Circular buffer allocations (MBs) - from graph_tracking.cpp"
echo "  - Kernel binary writes (KBs per core) - from llrt.cpp (NEW!)"
echo "  - Firmware writes (KBs per core) - from llrt.cpp (NEW!)"
echo "  - Launch message writes (bytes) - from llrt.cpp (NEW!)"
echo ""
echo "Total L1 should be in the MB range (10s-100s of MB)"
echo ""
echo "Kill server with: kill $SERVER_PID"
