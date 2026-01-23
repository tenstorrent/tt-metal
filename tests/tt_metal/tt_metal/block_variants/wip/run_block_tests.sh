#!/bin/bash
# Build and run block variant tests

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/../../../../.."

echo "🔨 Building tests..."
cd "$REPO_DIR"
./build_metal.sh --build-tests

echo ""
echo "🧪 Running block variant tests..."
echo ""

if [ -f "./build/test/tt_metal/unit_tests_legacy" ]; then
    echo "▶ Running block variant tests from unit_tests_legacy..."
    ./build/test/tt_metal/unit_tests_legacy --gtest_filter='*Block*' || echo "❌ Some tests failed"
    echo ""
else
    echo "⚠️  unit_tests_legacy not found (build failed?)"
    exit 1
fi

echo "✅ Test run complete!"
