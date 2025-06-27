#!/bin/bash

# Script to run distributed multi-process visible devices tests

# Array of device configurations to test
DEVICE_CONFIGS=("0" "1" "2" "3" "0,1" "0,3" "1,2" "2,3")

echo "[distributed tests] Testing TT_METAL_VISIBLE_DEVICES functionality with distributed_mp_unit_tests"
echo "============================================================================"

# Track overall success
ALL_PASSED=true

for config in "${DEVICE_CONFIGS[@]}"; do
    echo
    echo "[distributed tests] Testing with TT_METAL_VISIBLE_DEVICES=\"$config\""
    echo "------------------------------------------------"

    # Run with mpirun, setting the environment variable
    TT_METAL_VISIBLE_DEVICES="$config" mpirun --allow-run-as-root -np 1 ./build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests

    if [ $? -eq 0 ]; then
        echo "✓ [distributed tests] Test passed for configuration: $config"
    else
        echo "✗ [distributed tests] Test failed for configuration: $config"
        ALL_PASSED=false
    fi
done

echo
if [ "$ALL_PASSED" = true ]; then
    echo "[distributed tests] All tests passed!"
    exit 0
else
    echo "[distributed tests] Some tests failed!"
    exit 1
fi
