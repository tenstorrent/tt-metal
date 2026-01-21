#!/bin/bash
# Add block variant tests to CMakeLists.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAKE_FILE="${SCRIPT_DIR}/tt-metal/tests/tt_metal/tt_metal/CMakeLists.txt"

echo "📝 Adding block variant tests to CMakeLists.txt..."

if [ ! -f "$CMAKE_FILE" ]; then
    echo "❌ CMakeLists.txt not found: $CMAKE_FILE"
    exit 1
fi

# Backup
cp "$CMAKE_FILE" "${CMAKE_FILE}.backup"
echo "✅ Backup created: ${CMAKE_FILE}.backup"

# Add tests
cat >> "$CMAKE_FILE" << 'EOF'

# Block variant tests (Issue #35739)
tt_metal_add_gtest(test_eltwise_binary_block
    block_variants/test_eltwise_binary_block.cpp
)

tt_metal_add_gtest(test_reduce_block
    block_variants/test_reduce_block.cpp
)

tt_metal_add_gtest(test_broadcast_block
    block_variants/test_broadcast_block.cpp
)

tt_metal_add_gtest(test_transpose_block
    block_variants/test_transpose_block.cpp
)

tt_metal_add_gtest(test_pack_block
    block_variants/test_pack_block.cpp
)
EOF

echo "✅ Added 5 block variant tests to CMakeLists.txt"
echo ""
echo "📋 Next steps:"
echo "   1. cd tt-metal"
echo "   2. ./build_metal.sh --build-tests"
echo "   3. ./build/test/tt_metal/test_eltwise_binary_block"
echo ""
echo "Or use: ./run_block_tests.sh"
