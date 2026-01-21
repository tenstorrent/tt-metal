#!/bin/bash
# Build and run block variant tests

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/tt-metal"

echo "🔨 Building tests..."
cd "$REPO_DIR"
./build_metal.sh --build-tests

echo ""
echo "🧪 Running block variant tests..."
echo ""

for test in test_eltwise_binary_block test_reduce_block test_broadcast_block test_transpose_block test_pack_block; do
    if [ -f "./build/test/tt_metal/$test" ]; then
        echo "▶ Running $test..."
        ./build/test/tt_metal/$test || echo "❌ $test failed"
        echo ""
    else
        echo "⚠️  $test not found (not added to CMakeLists.txt?)"
    fi
done

echo "✅ Test run complete!"
