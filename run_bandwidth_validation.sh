#!/bin/bash
# Simple bandwidth validation runner
# Coordinates telemetry and fabric test to validate bandwidth metrics

set -e

TELEMETRY_SERVER="./build/tt_telemetry/tt_telemetry_server"
FSD_FILE="${FSD_FILE:-/data/btrzynadlowski/tt-metal/fsd.textproto}"
FABRIC_TEST="./build/test/tt_metal/tt_fabric/fabric_unit_tests"
FABRIC_FILTER="Fabric2DFixture.TestUnicastConnAPIDRAM"

echo "==============================================="
echo "Fabric Bandwidth Validation Test"
echo "==============================================="
echo ""
echo "This script validates bandwidth telemetry by:"
echo "1. Starting telemetry server (local chips only)"
echo "2. Waiting for initialization"
echo "3. Running fabric test"
echo "4. Checking if bandwidth metrics appeared"
echo ""

# Export required env vars
export TT_METAL_FABRIC_TELEMETRY=1

# Start telemetry server in background
echo "Starting telemetry server (local chips only)..."
$TELEMETRY_SERVER --fsd=$FSD_FILE --watchdog-timeout 120 > telemetry_server.log 2>&1 &
TELEMETRY_PID=$!

# Give it time to initialize
echo "Waiting for telemetry to initialize..."
sleep 15

# Check if telemetry is responding
if ! curl -s http://localhost:8080/api/status > /dev/null 2>&1; then
    echo "ERROR: Telemetry server not responding"
    kill $TELEMETRY_PID 2>/dev/null || true
    exit 1
fi

echo "✓ Telemetry server running (PID $TELEMETRY_PID)"

# Read baseline metrics
echo ""
echo "Reading baseline metrics..."
BASELINE_COUNT=$(curl -s http://localhost:8080/api/metrics | grep -c "BandwidthMBps" || echo 0)
echo "Baseline bandwidth metric count: $BASELINE_COUNT"

# Stop telemetry before running fabric test
echo ""
echo "Stopping telemetry to run fabric test..."
kill $TELEMETRY_PID
wait $TELEMETRY_PID 2>/dev/null || true
sleep 2

# Run fabric test
echo "Running fabric test: $FABRIC_FILTER"
echo "This will generate fabric traffic and initialize firmware..."
$FABRIC_TEST --gtest_filter="$FABRIC_FILTER" > fabric_test.log 2>&1
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo "WARNING: Fabric test exited with code $TEST_RESULT"
    echo "Check fabric_test.log for details"
    echo "Continuing to check if telemetry works..."
fi

# Wait a moment for firmware to settle
sleep 2

# Restart telemetry
echo ""
echo "Restarting telemetry server..."
$TELEMETRY_SERVER --fsd=$FSD_FILE --watchdog-timeout 120 > telemetry_server_after.log 2>&1 &
TELEMETRY_PID=$!

# Give it time to read metrics
echo "Waiting for telemetry to read metrics..."
sleep 15

# Check bandwidth metrics
echo ""
echo "Checking for bandwidth metrics..."
BANDWIDTH_METRICS=$(curl -s http://localhost:8080/api/metrics | grep "BandwidthMBps{")

if [ -z "$BANDWIDTH_METRICS" ]; then
    echo "❌ FAIL: No bandwidth metrics found"
    kill $TELEMETRY_PID 2>/dev/null || true
    exit 1
fi

# Count non-zero bandwidth values
NON_ZERO=$(echo "$BANDWIDTH_METRICS" | grep -v " 0.000000 " | wc -l)

echo "✓ Found bandwidth metrics!"
echo ""
echo "Sample bandwidth values:"
echo "$BANDWIDTH_METRICS" | head -10
echo ""
echo "Non-zero bandwidth values: $NON_ZERO"

# Check supportedStats
SUPPORTED_STATS=$(curl -s http://localhost:8080/api/metrics | grep "supportedStats{" | head -1)
echo ""
echo "Firmware initialization:"
echo "$SUPPORTED_STATS"

# Cleanup
kill $TELEMETRY_PID 2>/dev/null || true

echo ""
echo "==============================================="
if [ $NON_ZERO -gt 0 ]; then
    echo "✓ SUCCESS: Bandwidth telemetry working!"
    echo "  Found $NON_ZERO endpoints with non-zero bandwidth"
else
    echo "⚠️  Bandwidth metrics exist but all zero"
    echo "  Firmware may have reset between test and telemetry read"
fi
echo "==============================================="
