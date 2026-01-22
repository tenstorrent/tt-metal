#!/bin/bash
# Fix ttnn .so file location

cd /localdev/kkfernandez/tt-metal

echo "=== Fixing ttnn .so file locations ==="
echo ""

# The .so files should be in ttnn/ not ttnn/ttnn/
echo "1. Checking current .so locations..."
find ttnn -name "_ttnn*.so" 2>/dev/null

echo ""
echo "2. Creating symlinks in the correct location (ttnn/)..."

# _ttnn.so should be at ttnn/_ttnn.so for "import ttnn._ttnn" to work
if [ -f "build/lib/_ttnn.so" ]; then
    ln -sf ../build/lib/_ttnn.so ttnn/_ttnn.so
    echo "✓ Linked build/lib/_ttnn.so -> ttnn/_ttnn.so"
fi

if [ -f "ttnn/ttnn/_ttnn.so" ]; then
    ln -sf ttnn/_ttnn.so ttnn/_ttnn.so 2>/dev/null || true
    echo "✓ Linked ttnn/ttnn/_ttnn.so -> ttnn/_ttnn.so"
fi

if [ -f "build/lib/_ttnncpp.so" ]; then
    ln -sf ../build/lib/_ttnncpp.so ttnn/_ttnncpp.so
    echo "✓ Linked build/lib/_ttnncpp.so -> ttnn/_ttnncpp.so"
fi

if [ -f "build_Release/ttnn/_ttnncpp.so" ]; then
    ln -sf ../build_Release/ttnn/_ttnncpp.so ttnn/_ttnncpp.so
    echo "✓ Linked build_Release/ttnn/_ttnncpp.so -> ttnn/_ttnncpp.so"
fi

echo ""
echo "3. Checking if ttnn/__init__.py exists..."
if [ -f "ttnn/__init__.py" ]; then
    echo "✓ ttnn/__init__.py exists"
else
    echo "✗ ttnn/__init__.py missing - creating minimal one..."
    cat > ttnn/__init__.py << 'PYEOF'
# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Import the ttnn submodule which contains all the functionality
from ttnn.ttnn import *
PYEOF
    echo "✓ Created ttnn/__init__.py"
fi

echo ""
echo "4. Current structure:"
ls -la ttnn/*.so 2>/dev/null || echo "  No .so files directly in ttnn/"
ls -la ttnn/ttnn/*.so 2>/dev/null | head -3

echo ""
echo "5. Testing import..."
export LD_LIBRARY_PATH=/localdev/kkfernandez/tt-metal/build/lib:$LD_LIBRARY_PATH

python3 << 'EOF'
import sys
sys.path.insert(0, '/localdev/kkfernandez/tt-metal')

try:
    import ttnn._ttnn
    print("✓ SUCCESS: ttnn._ttnn imported!")
    print(f"  Has {len(dir(ttnn._ttnn))} attributes")
except ImportError as e:
    print(f"✗ ttnn._ttnn import failed: {e}")

try:
    from ttnn.ttnn import *
    print("✓ SUCCESS: from ttnn.ttnn import * worked!")
except ImportError as e:
    print(f"✗ from ttnn.ttnn import * failed: {e}")

try:
    import ttnn
    attrs = [a for a in dir(ttnn) if not a.startswith('_')]
    print(f"✓ ttnn has {len(attrs)} non-private attributes")
    if hasattr(ttnn, 'CreateDevice'):
        print("  ✓ ttnn.CreateDevice available!")
    else:
        print("  ✗ ttnn.CreateDevice NOT available")
except Exception as e:
    print(f"✗ Error: {e}")
EOF

echo ""
echo "=== Done ==="
