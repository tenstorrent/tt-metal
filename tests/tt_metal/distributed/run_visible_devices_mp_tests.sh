#!/bin/bash

# Script to run distributed multi-process visible devices tests

# Array of device configurations to test
DEVICE_CONFIGS=("0" "1" "2" "3" "0,1" "0,3" "1,2" "2,3")

echo "Testing TT_METAL_VISIBLE_DEVICES functionality with distributed_mp_unit_tests"
echo "============================================================================"

# Build if needed
if [ ! -f "build/test/tt_metal/distributed/distributed_mp_unit_tests" ]; then
    echo "Building tests..."
    ./build_metal.sh --debug --build-tests
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
fi

# Track overall success
ALL_PASSED=true

for config in "${DEVICE_CONFIGS[@]}"; do
    echo
    echo "Testing with TT_METAL_VISIBLE_DEVICES=\"$config\""
    echo "------------------------------------------------"

    # Run with mpirun, setting the environment variable
    TT_METAL_VISIBLE_DEVICES="$config" mpirun -np 1 ./build/test/tt_metal/distributed/distributed_mp_unit_tests --gtest_filter="VisibleDevicesMPTest.*"

    if [ $? -eq 0 ]; then
        echo "✓ Test passed for configuration: $config"
    else
        echo "✗ Test failed for configuration: $config"
        ALL_PASSED=false
    fi
done

echo
if [ "$ALL_PASSED" = true ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi
