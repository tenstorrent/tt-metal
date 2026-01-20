#!/bin/bash
# Properly reset all devices and wait for them to be ready

set -e

echo "=================================="
echo "RESETTING ALL DEVICES"
echo "=================================="

# Number of devices to reset
NUM_DEVICES=${1:-8}

echo "Resetting $NUM_DEVICES devices..."

# Reset each device
for i in $(seq 0 $((NUM_DEVICES - 1))); do
    echo "  Resetting device $i..."
    tt-smi -r $i || echo "Warning: Failed to reset device $i"
    sleep 5
done

echo ""
echo "Waiting 60 seconds for devices to stabilize..."
sleep 60

echo ""
echo "Checking device status..."
tt-smi

echo ""
echo "=================================="
echo "RESET COMPLETE"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run diagnostics: python3 telemetry-perf/diagnose_devices.py"
echo "2. If diagnostics pass, run benchmarks"
