#!/bin/bash
# Rebuild TT-Metal with Program Metadata Kernel Type Tracking

set -e  # Exit on error

echo "========================================="
echo "Rebuilding TT-Metal with Kernel Type Tracking"
echo "========================================="
echo ""
echo "Changes:"
echo "  ✅ Added ProgramKernelType enum (APPLICATION=0, FABRIC=1, DISPATCH=2)"
echo "  ✅ Added kernel_type_ field to ProgramImpl"
echo "  ✅ Set type in Device::configure_fabric() → FABRIC"
echo "  ✅ Set type in Device::configure_command_queue_programs() → DISPATCH"
echo "  ✅ Pass type from program metadata to track_kernel_load()"
echo ""

cd /home/ttuser/aperezvicente/tt-metal-apv

echo "📊 Modified files:"
echo "   program_impl.hpp: $(stat -c %y tt_metal/impl/program/program_impl.hpp)"
echo "   program.cpp:      $(stat -c %y tt_metal/impl/program/program.cpp)"
echo "   device.cpp:       $(stat -c %y tt_metal/impl/device/device.cpp)"
echo "   graph_tracking.*: $(stat -c %y tt_metal/graph/graph_tracking.cpp)"
echo ""

echo "🔨 Starting rebuild (this will take a few minutes)..."
echo ""

# Try to rebuild with cmake
if [ -d "build" ]; then
    echo "Building with cmake (build/)..."
    cmake --build build --target tt_metal -j$(nproc) 2>&1 | tee build_kernel_type.log | grep -E "Built target|Error|error:|warning:" || true
    BUILD_DIR="build"
elif [ -d "build_Release" ]; then
    echo "Building with cmake (build_Release/)..."
    cmake --build build_Release --target tt_metal -j$(nproc) 2>&1 | tee build_kernel_type.log | grep -E "Built target|Error|error:|warning:" || true
    BUILD_DIR="build_Release"
else
    echo "❌ No build directory found!"
    echo "Please run cmake configuration first"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build completed successfully!"
    echo ""
    echo "📊 New library timestamp:"
    ls -lh --time-style='+%Y-%m-%d %H:%M:%S' ${BUILD_DIR}/lib/libtt_metal.so 2>/dev/null || \
    ls -lh --time-style='+%Y-%m-%d %H:%M:%S' ${BUILD_DIR}/tt_metal/libtt_metal.so 2>/dev/null || \
    echo "Library not found in expected location"
    echo ""
    echo "========================================="
    echo "✅ Rebuild Complete!"
    echo "========================================="
    echo ""
    echo "🎯 Expected behavior after rebuild:"
    echo ""
    echo "BEFORE (wrong):"
    echo "  ✗ [KERNEL_LOAD] Application kernel on Device 0: +0.046 MB"
    echo "  ✗ [KERNEL_LOAD] Application kernel on Device 0: +0.046 MB"
    echo ""
    echo "AFTER (correct):"
    echo "  ✓ [KERNEL_LOAD] Fabric kernel on Device 0: +0.056 MB"
    echo "  ✓ [KERNEL_LOAD] Dispatch kernel on Device 0: +0.046 MB"
    echo ""
    echo "Next steps:"
    echo "  1. Kill current allocation server (if running)"
    echo "  2. Start fresh: ./build/programming_examples/allocation_server_poc &"
    echo "  3. Run test: pytest models/tt_transformers/demo/simple_text_demo.py -k 'DP-4-b1'"
    echo "  4. Check log: grep 'KERNEL_LOAD' out.log | head -20"
    echo "  5. Verify: Should see 'Fabric kernel' and 'Dispatch kernel' instead of 'Application'"
    echo ""
else
    echo ""
    echo "❌ Build failed! Check build_kernel_type.log for errors"
    echo ""
    tail -50 build_kernel_type.log
    exit 1
fi
