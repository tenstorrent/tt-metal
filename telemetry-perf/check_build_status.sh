#!/bin/bash
# Check if tt-metal is properly built and configured

echo "================================================================================"
echo "TT-METAL BUILD STATUS CHECK"
echo "================================================================================"
echo ""

cd /localdev/kkfernandez/tt-metal

# 1. Check for build directory
echo "=== 1. Checking build directory ==="
if [ -d "build" ]; then
    echo "✓ build/ directory exists"
    ls -lah build/ | head -10
else
    echo "✗ build/ directory not found"
fi
echo ""

# 2. Check for lib directory and shared objects
echo "=== 2. Checking for built libraries ==="
if [ -d "build/lib" ]; then
    echo "✓ build/lib/ exists"
    echo "Contents:"
    ls -lah build/lib/*.so 2>/dev/null | head -5 || echo "  No .so files found"
else
    echo "✗ build/lib/ not found"
fi
echo ""

# 3. Check for Python environment
echo "=== 3. Checking Python environment ==="
if [ -d "build/python_env" ]; then
    echo "✓ build/python_env/ exists"
    ls -lah build/python_env/
else
    echo "✗ build/python_env/ not found"
fi
echo ""

# 4. Check for ttnn module files
echo "=== 4. Checking ttnn module files ==="
echo "Looking for .so files in ttnn/:"
find ttnn -name "*.so" 2>/dev/null | head -10
if [ $? -eq 0 ]; then
    echo "✓ Found some .so files"
else
    echo "✗ No .so files found in ttnn/"
fi
echo ""

# 5. Check Python path
echo "=== 5. Checking PYTHONPATH ==="
echo "Current PYTHONPATH:"
python3 -c "import sys; print('\n'.join([p for p in sys.path if 'tt-metal' in p]))"
echo ""

# 6. Check environment variables
echo "=== 6. Checking environment variables ==="
echo "TT_METAL_HOME: ${TT_METAL_HOME:-NOT SET}"
echo "ARCH_NAME: ${ARCH_NAME:-NOT SET}"
echo "LD_LIBRARY_PATH (tt-metal related):"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep tt-metal
echo ""

# 7. Try importing ttnn and checking what's available
echo "=== 7. Checking ttnn module import ==="
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/localdev/kkfernandez/tt-metal')

try:
    import ttnn
    print("✓ ttnn imports")

    # Check if it's a proper module or empty
    attrs = [a for a in dir(ttnn) if not a.startswith('_')]
    print(f"  Non-private attributes: {len(attrs)}")
    if len(attrs) > 0:
        print(f"  Attributes: {attrs[:10]}")
    else:
        print("  ✗ Module is empty (no attributes)")

    # Check for expected functions
    expected = ['CreateDevice', 'close_device', 'add', 'multiply', 'from_torch']
    found = [e for e in expected if hasattr(ttnn, e)]
    missing = [e for e in expected if not hasattr(ttnn, e)]

    if found:
        print(f"  ✓ Found: {found}")
    if missing:
        print(f"  ✗ Missing: {missing}")

except ImportError as e:
    print(f"✗ Failed to import ttnn: {e}")
PYEOF
echo ""

# 8. Check for build script
echo "=== 8. Checking for build script ==="
if [ -f "build_metal.sh" ]; then
    echo "✓ build_metal.sh exists"
elif [ -f "scripts/build_scripts/build_with_profiler_opt.sh" ]; then
    echo "✓ Found build script in scripts/"
else
    echo "✗ No build script found"
fi
echo ""

# 9. Check if any tests can import ttnn successfully
echo "=== 9. Testing if pytest tests can import ttnn ==="
timeout 10 python3 -c "
import sys
sys.path.insert(0, '/localdev/kkfernandez/tt-metal')
import pytest
# Try to import from a test file
exec(open('tests/ttnn/unit_tests/operations/test_add.py').read())
print('✓ Test file imports successfully')
" 2>&1 | head -20 || echo "✗ Test import failed"
echo ""

# 10. Check for cmake build files
echo "=== 10. Checking CMake configuration ==="
if [ -f "build/CMakeCache.txt" ]; then
    echo "✓ CMakeCache.txt exists"
    echo "Build type:"
    grep CMAKE_BUILD_TYPE build/CMakeCache.txt | head -1
else
    echo "✗ No CMakeCache.txt found - project not configured"
fi
echo ""

# Summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""

# Determine if build is complete
BUILD_COMPLETE=true

if [ ! -d "build/lib" ]; then
    BUILD_COMPLETE=false
    echo "❌ Build incomplete: build/lib/ directory missing"
fi

if ! python3 -c "import sys; sys.path.insert(0, '/localdev/kkfernandez/tt-metal'); import ttnn; assert len([a for a in dir(ttnn) if not a.startswith('_')]) > 5" 2>/dev/null; then
    BUILD_COMPLETE=false
    echo "❌ Build incomplete: ttnn module is empty or not properly built"
fi

if [ "$BUILD_COMPLETE" = true ]; then
    echo "✅ tt-metal appears to be properly built"
    echo ""
    echo "You should be able to run tests with:"
    echo "  pytest tests/ttnn/unit_tests/operations/test_add.py::test_add -v"
else
    echo ""
    echo "❌ tt-metal is NOT properly built"
    echo ""
    echo "To build tt-metal, try:"
    echo "  cd /localdev/kkfernandez/tt-metal"
    echo "  ./build_metal.sh"
    echo ""
    echo "Or check the build documentation for your specific setup."
fi

echo ""
echo "================================================================================"
