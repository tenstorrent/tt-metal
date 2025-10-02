#!/bin/bash
# Debug CB0 (A tiles) Corruption - Focus on Device 1

echo "🔍 CB0 (A Tiles) Corruption Debug"
echo "================================="

cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/e4/elementwise_add

# Set debug environment
export TT_METAL_HOME="/home/tt-metal-apv"
export TT_METAL_DPRINT_CORES="all"
export TT_METAL_DPRINT_ENABLE=1
export TT_METAL_DPRINT_FILE="./cb0_debug.log"
export TT_METAL_DPRINT_PRINT_ALL=1
export TT_METAL_SLOW_DISPATCH_MODE=1

# Disable profiler
unset TT_METAL_PROFILER
unset TT_METAL_DEVICE_PROFILER

echo "🔧 Testing CB0 corruption patterns..."

# Test with different A buffer configurations
echo "Test 1: Single device (should work)"
/home/tt-metal-apv/build-cmake/tt_metal/programming_examples/alexp_examples/e4/elementwise_add/programming_examples/alexp_examples/e4/elementwise_add/alexp_distributed_elementwise_add 8 > cb0_test_8tiles.log 2>&1
result_8=$(grep "verification:" cb0_test_8tiles.log | tail -1)
echo "  8 tiles: $result_8"

echo "Test 2: Multi-device (Device 1 fails)"
/home/tt-metal-apv/build-cmake/tt_metal/programming_examples/alexp_examples/e4/elementwise_add/programming_examples/alexp_examples/e4/elementwise_add/alexp_distributed_elementwise_add 16 > cb0_test_16tiles.log 2>&1
result_16=$(grep "verification:" cb0_test_16tiles.log | tail -1)
echo "  16 tiles: $result_16"

echo ""
echo "🔍 Analyzing CB0 corruption..."

# Extract CB0 corruption patterns
echo "CB0 Values from 8 tiles (good):"
grep "COMPUTE in0 tile\[0\](0):" cb0_test_8tiles.log | head -3

echo ""
echo "CB0 Values from 16 tiles (bad):"
grep "COMPUTE in0 tile\[0\](0):" cb0_test_16tiles.log | head -5

echo ""
echo "📊 Pattern Analysis:"
echo "- Good CB0: Should show '1' consistently"
echo "- Bad CB0: Shows huge numbers like -1.93257e+26"
echo "- This suggests memory corruption in A tile transfer to CB0"

echo ""
echo "🔧 Potential Issues:"
echo "1. CB0 buffer size too small for distributed A tiles"
echo "2. Memory alignment issues in distributed buffer"
echo "3. Device 1 DRAM addressing problems"
echo "4. CB0 synchronization with distributed sharding"
