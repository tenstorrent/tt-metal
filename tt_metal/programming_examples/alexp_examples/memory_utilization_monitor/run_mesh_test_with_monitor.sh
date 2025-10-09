#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Script to run the mesh allocation test with monitoring

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  Mesh Allocation Test with Monitoring"
echo "========================================="
echo ""

# Check if server is running
if ! pgrep -f allocation_server_poc > /dev/null; then
    echo "⚠️  Allocation server is not running!"
    echo "Starting allocation server..."
    ./allocation_server_poc &
    SERVER_PID=$!
    sleep 2
    echo "✓ Server started (PID: $SERVER_PID)"
    echo ""
else
    echo "✓ Allocation server is already running"
    SERVER_PID=""
    echo ""
fi

echo "========================================="
echo "  Instructions:"
echo "========================================="
echo "1. In another terminal, run the monitor:"
echo "   cd $SCRIPT_DIR"
echo "   ./allocation_monitor_client -a -r 500"
echo ""
echo "2. Press ENTER when ready to start the test..."
read

echo ""
echo "Starting C++ mesh allocation test..."
echo "Watch the monitor to see allocations across all 8 devices!"
echo ""

cd /home/tt-metal-apv
source env_vars_setup.sh
export TT_ALLOC_TRACKING_ENABLED=1
./build_Release_tracy/programming_examples/test_mesh_allocation_cpp

echo ""
echo "========================================="
echo "  Test Complete!"
echo "========================================="

if [ -n "$SERVER_PID" ]; then
    echo "Stopping allocation server..."
    kill $SERVER_PID 2>/dev/null
fi

echo "Done!"
