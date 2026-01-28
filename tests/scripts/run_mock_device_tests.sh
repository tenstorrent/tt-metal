#!/usr/bin/env bash

# Mock Device Test Suite Runner
# This script runs existing tests with mock device environment variables

set -e

# Check if cluster descriptor is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <cluster_descriptor.yaml>"
    echo "Example: $0 tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P100.yaml"
    exit 1
fi

CLUSTER_DESC="$1"

# Set required environment variable
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_DESC"

echo "=========================================="
echo "Running Mock Device Test Suite"
echo "Cluster Descriptor: $CLUSTER_DESC"
echo "=========================================="

# Run device pool tests (Basic device operations)
echo ""
echo "=== Testing Device Pool and Mesh Device ==="
./build_Release/test/tt_metal/unit_tests_device --gtest_filter="DevicePool.*:DeviceFixture.CreateMeshDeviceHandle"

# Run dispatch tests
echo ""
echo "=== Testing Dispatch (Fast Dispatch) ==="
./build_Release/test/tt_metal/unit_tests_dispatch --gtest_filter="MeshDispatchFixture.*"

echo ""
echo "=========================================="
echo "Mock Device Tests Completed!"
echo "Summary:"
echo "  - Device Pool: Basic device operations"
echo "  - Dispatch: All MeshDispatchFixture tests"
echo "  - Config: $CLUSTER_DESC"
echo "=========================================="
