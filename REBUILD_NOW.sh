#!/bin/bash
# Rebuild TT-Metal with the double-free fix

set -e  # Exit on error

echo "========================================="
echo "Rebuilding TT-Metal with Double-Free Fix"
echo "========================================="
echo ""

cd /home/tt-metal-apv

echo "📊 Current status:"
echo "   buffer.cpp modified: $(stat -c %y tt_metal/impl/buffers/buffer.cpp)"
echo "   Current library: $(stat -c %y build_Release/tt_metal/libtt_metal.so 2>/dev/null || echo 'Not found')"
echo ""

echo "🔨 Starting rebuild..."
echo ""

# Try to rebuild with cmake
if [ -d "build" ]; then
    echo "Building with cmake (build/)..."
    cmake --build build --target tt_metal -j$(nproc)
    echo "✅ Build completed!"
elif [ -d "build_Release" ]; then
    echo "Building with cmake (build_Release/)..."
    cmake --build build_Release --target tt_metal -j$(nproc)
    echo "✅ Build completed!"
else
    echo "❌ No build directory found!"
    echo "Please run cmake configuration first"
    exit 1
fi

echo ""
echo "📊 New library timestamp:"
ls -lh --time-style='+%Y-%m-%d %H:%M:%S' build*/tt_metal/libtt_metal.so 2>/dev/null || echo "Library not found"

echo ""
echo "========================================="
echo "✅ Rebuild Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Kill any running allocation server"
echo "2. Restart the allocation server"
echo "3. Run your test again"
echo "4. Check that unknown warnings dropped from 1,050 to ~0-50"
echo ""
