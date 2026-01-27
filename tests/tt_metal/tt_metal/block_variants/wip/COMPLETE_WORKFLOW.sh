#!/bin/bash
# Complete workflow: Generate tests, complete TODOs with AI, add to build, run tests

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Complete Block Variant Test Workflow                 ║"
echo "║  tt-metal Compute API - Issue #35739                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Generate test skeletons
echo "📝 Step 1/5: Generating test skeletons..."
./run_test_generation.sh --all
echo ""

# Step 2: AI agents complete TODOs
echo "🤖 Step 2/5: AI agents completing TODOs (parallel)..."
./run_test_completion.sh --parallel
echo ""

# Step 3: Add to CMakeLists.txt
echo "⚙️  Step 3/5: Adding tests to CMakeLists.txt..."
./add_tests_to_cmake.sh
echo ""

# Step 4: Build tests
echo "🔨 Step 4/5: Building tests..."
cd ../../../../..
./build_metal.sh --build-tests
cd - > /dev/null
echo ""

# Step 5: Run tests
echo "🧪 Step 5/5: Running all block variant tests..."
./run_block_tests.sh
echo ""

echo "✅ Complete workflow finished!"
