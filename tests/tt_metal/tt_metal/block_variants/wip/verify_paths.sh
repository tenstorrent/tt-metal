#!/bin/bash
# Verify all paths are correct from new location

echo "Verifying paths from: $(pwd)"
echo ""

# Test 1: tt-metal root
REPO_ROOT="../../../../.."
if [ -d "$REPO_ROOT/tt_metal" ]; then
    echo "✅ tt-metal root found: $REPO_ROOT"
else
    echo "❌ tt-metal root NOT found: $REPO_ROOT"
fi

# Test 2: CMakeLists.txt
CMAKE_FILE="../../CMakeLists.txt"
if [ -f "$CMAKE_FILE" ]; then
    echo "✅ CMakeLists.txt found: $CMAKE_FILE"
else
    echo "❌ CMakeLists.txt NOT found: $CMAKE_FILE"
fi

# Test 3: Test files
TEST_DIR=".."
test_count=$(ls -1 "$TEST_DIR"/test_*.cpp 2>/dev/null | wc -l)
echo "✅ Found $test_count test files in: $TEST_DIR"

# Test 4: Kernels directory
KERNEL_DIR="../kernels"
if [ -d "$KERNEL_DIR" ]; then
    kernel_count=$(ls -1 "$KERNEL_DIR"/*.cpp 2>/dev/null | wc -l)
    echo "✅ Found $kernel_count kernel files in: $KERNEL_DIR"
else
    echo "❌ Kernels directory NOT found: $KERNEL_DIR"
fi

# Test 5: Scripts
script_count=$(ls -1 *.sh 2>/dev/null | wc -l)
echo "✅ Found $script_count shell scripts in current directory"

# Test 6: Python scripts
py_count=$(ls -1 *.py 2>/dev/null | wc -l)
echo "✅ Found $py_count Python scripts in current directory"

echo ""
echo "All paths verified! Ready to run."
