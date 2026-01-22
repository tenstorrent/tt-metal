#!/bin/bash
# Set up environment to make ttnn imports work

cd /localdev/kkfernandez/tt-metal

echo "=== Setting up ttnn environment ==="
echo ""

# 1. Create symlinks for .so files in the location Python expects
echo "1. Creating symlinks for .so files..."
ln -sf build/lib/_ttnn.so ttnn/_ttnn.so 2>/dev/null && echo "   ✓ ttnn/_ttnn.so" || echo "   (already exists)"
ln -sf build/lib/_ttnncpp.so ttnn/_ttnncpp.so 2>/dev/null && echo "   ✓ ttnn/_ttnncpp.so" || echo "   (already exists)"

# 2. Add tools to Python path (for tracy module)
echo ""
echo "2. Adding tools/ to PYTHONPATH for tracy module..."
export PYTHONPATH=/localdev/kkfernandez/tt-metal/tools:/localdev/kkfernandez/tt-metal:$PYTHONPATH
echo "   ✓ PYTHONPATH updated"

# 3. Set library path
echo ""
echo "3. Setting LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=/localdev/kkfernandez/tt-metal/build/lib:$LD_LIBRARY_PATH
echo "   ✓ LD_LIBRARY_PATH updated"

# 4. Set TT_METAL_HOME
echo ""
echo "4. Setting TT_METAL_HOME..."
export TT_METAL_HOME=/localdev/kkfernandez/tt-metal
echo "   ✓ TT_METAL_HOME=$TT_METAL_HOME"

echo ""
echo "5. Testing imports..."

python3 << 'EOF'
import sys
sys.path.insert(0, '/localdev/kkfernandez/tt-metal/tools')
sys.path.insert(0, '/localdev/kkfernandez/tt-metal')

try:
    import tracy
    print("   ✓ tracy imported")
except ImportError as e:
    print(f"   ✗ tracy import failed: {e}")

try:
    import ttnn._ttnn
    print("   ✓ ttnn._ttnn imported")
    print(f"     Attributes: {len(dir(ttnn._ttnn))}")
except ImportError as e:
    print(f"   ✗ ttnn._ttnn import failed: {e}")
    sys.exit(1)

try:
    import ttnn
    attrs = [a for a in dir(ttnn) if not a.startswith('_')]
    print(f"   ✓ ttnn imported with {len(attrs)} attributes")

    if hasattr(ttnn, 'CreateDevice'):
        print("   ✓ ttnn.CreateDevice is available!")
    else:
        print("   ✗ ttnn.CreateDevice NOT available")
        print(f"     Available: {attrs[:20]}")
except Exception as e:
    print(f"   ✗ ttnn import failed: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "SUCCESS! Environment is configured"
    echo "================================================================================"
    echo ""
    echo "To make these changes permanent, add to your ~/.bashrc or venv activate:"
    echo ""
    echo "export PYTHONPATH=/localdev/kkfernandez/tt-metal/tools:/localdev/kkfernandez/tt-metal:\$PYTHONPATH"
    echo "export LD_LIBRARY_PATH=/localdev/kkfernandez/tt-metal/build/lib:\$LD_LIBRARY_PATH"
    echo "export TT_METAL_HOME=/localdev/kkfernandez/tt-metal"
    echo ""
    echo "Now run your tests:"
    echo "  cd telemetry-perf"
    echo "  python3 comprehensive_single_device_benchmark.py reduced"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "FAILED - Still have import issues"
    echo "================================================================================"
fi
