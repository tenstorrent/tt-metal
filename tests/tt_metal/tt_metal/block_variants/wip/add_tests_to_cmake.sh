#!/bin/bash
# Add block variant tests to CMakeLists.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAKE_FILE="${SCRIPT_DIR}/../../CMakeLists.txt"

echo "📝 Adding block variant tests to CMakeLists.txt..."

if [ ! -f "$CMAKE_FILE" ]; then
    echo "❌ CMakeLists.txt not found: $CMAKE_FILE"
    exit 1
fi

# Check if tests are already added
if grep -q "block_variants/test_eltwise_binary_block.cpp" "$CMAKE_FILE"; then
    echo "⚠️  Block variant tests already in CMakeLists.txt"
    exit 0
fi

# Backup
cp "$CMAKE_FILE" "${CMAKE_FILE}.backup"
echo "✅ Backup created: ${CMAKE_FILE}.backup"

# Find the line with test_untilize_eltwise_binary.cpp and add after it
# This is in the UNIT_TESTS_LEGACY_SRC list
sed -i '/test_untilize_eltwise_binary.cpp/a\    block_variants/test_eltwise_binary_block.cpp\n    block_variants/test_reduce_block.cpp\n    block_variants/test_broadcast_block.cpp\n    block_variants/test_transpose_block.cpp\n    block_variants/test_pack_block.cpp' "$CMAKE_FILE"

echo "✅ Added 5 block variant tests to UNIT_TESTS_LEGACY_SRC"
echo ""
echo "📋 Next steps:"
echo "   1. cd ../../../../.. (to tt-metal root)"
echo "   2. ./build_metal.sh --build-tests"
echo "   3. ./build/test/tt_metal/unit_tests_legacy --gtest_filter='*Block*'"
echo ""
echo "Note: Tests are part of unit_tests_legacy executable"
