#!/bin/bash
# Fix ttnn Python module imports by symlinking .so files to correct locations

cd /localdev/kkfernandez/tt-metal

echo "=== Fixing ttnn Python module imports ==="
echo ""

# Find where the .so files are
echo "1. Locating .so files..."
TTNN_SO=$(find . -name "_ttnn.so" 2>/dev/null | head -1)
TTNNCPP_SO=$(find . -name "_ttnncpp.so" 2>/dev/null | head -1)

if [ -z "$TTNN_SO" ]; then
    echo "✗ _ttnn.so not found!"
    exit 1
fi

echo "   Found _ttnn.so at: $TTNN_SO"
echo "   Found _ttnncpp.so at: $TTNNCPP_SO"
echo ""

# Check if they're in the right place
echo "2. Checking ttnn module structure..."
if [ -f "ttnn/ttnn/_ttnn.so" ]; then
    echo "✓ _ttnn.so already in ttnn/ttnn/"
else
    echo "✗ _ttnn.so NOT in ttnn/ttnn/"

    # Create symlink
    echo "   Creating symlink..."
    ln -sf "$PWD/$TTNN_SO" ttnn/ttnn/_ttnn.so
    echo "✓ Symlinked $TTNN_SO -> ttnn/ttnn/_ttnn.so"
fi

if [ -f "ttnn/ttnn/_ttnncpp.so" ] && [ -n "$TTNNCPP_SO" ]; then
    echo "✓ _ttnncpp.so already in ttnn/ttnn/"
elif [ -n "$TTNNCPP_SO" ]; then
    echo "✗ _ttnncpp.so NOT in ttnn/ttnn/"
    ln -sf "$PWD/$TTNNCPP_SO" ttnn/ttnn/_ttnncpp.so
    echo "✓ Symlinked $TTNNCPP_SO -> ttnn/ttnn/_ttnncpp.so"
fi

echo ""
echo "3. Verifying imports..."

export LD_LIBRARY_PATH=/localdev/kkfernandez/tt-metal/build/lib:$LD_LIBRARY_PATH

python3 << 'EOF'
import sys
sys.path.insert(0, '/localdev/kkfernandez/tt-metal')

try:
    import ttnn._ttnn
    print("✓ ttnn._ttnn imports successfully!")
    print(f"  Attributes: {len(dir(ttnn._ttnn))}")
except ImportError as e:
    print(f"✗ Still cannot import ttnn._ttnn: {e}")
    sys.exit(1)

try:
    import ttnn
    non_private = [a for a in dir(ttnn) if not a.startswith('_')]
    print(f"✓ ttnn imports successfully!")
    print(f"  Non-private attributes: {len(non_private)}")

    if len(non_private) > 10:
        print(f"  Sample attributes: {non_private[:10]}")

        # Check for key functions
        if hasattr(ttnn, 'CreateDevice'):
            print("  ✓ ttnn.CreateDevice is available!")
        if hasattr(ttnn, 'close_device'):
            print("  ✓ ttnn.close_device is available!")
    else:
        print(f"  ✗ Module still mostly empty")

except Exception as e:
    print(f"✗ Error with ttnn: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "SUCCESS! ttnn module is now working"
    echo "================================================================================"
    echo ""
    echo "Add this to your environment:"
    echo "  export LD_LIBRARY_PATH=/localdev/kkfernandez/tt-metal/build/lib:\$LD_LIBRARY_PATH"
    echo ""
    echo "Now you can run your tests:"
    echo "  cd telemetry-perf"
    echo "  python3 comprehensive_single_device_benchmark.py reduced"
else
    echo ""
    echo "================================================================================"
    echo "FAILED - Additional debugging needed"
    echo "================================================================================"
fi
