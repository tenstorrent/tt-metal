#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Debug Distributed Elementwise Add - Enable RISC-V Core Debug Output

echo "🔍 Distributed Elementwise Add Kernel Debug Mode"
echo "================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Setting up debug environment..."

# Set TT_METAL_HOME (required)
export TT_METAL_HOME="/home/tt-metal-apv"

# CRITICAL: Enable DPRINT for kernel debugging
export TT_METAL_DPRINT_CORES="all"           # Debug all cores
export TT_METAL_DPRINT_DISABLE_ASSERT=1      # Disable assertions for cleaner output
export TT_METAL_DPRINT_ENABLE=1              # Enable DPRINT system

# Disable profiler (conflicts with DPRINT)
unset TT_METAL_PROFILER
unset TT_METAL_DEVICE_PROFILER

# Enable debug logging
export TT_METAL_LOG_LEVEL=Debug
export TT_METAL_LOGGER_LEVEL=Debug

# Set up debug output
export TT_METAL_DPRINT_FILE="./elementwise_kernel_debug.log"
export TT_METAL_DPRINT_PRINT_ALL=1

# Enable slow dispatch for better debugging
export TT_METAL_SLOW_DISPATCH_MODE=1

# Optional: Split debug output per RISC-V core
export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1

echo "✅ Debug environment configured:"
echo "   TT_METAL_DPRINT_CORES = $TT_METAL_DPRINT_CORES"
echo "   TT_METAL_DPRINT_FILE = $TT_METAL_DPRINT_FILE"
echo "   TT_METAL_LOG_LEVEL = $TT_METAL_LOG_LEVEL"
echo "   TT_METAL_DPRINT_ONE_FILE_PER_RISC = $TT_METAL_DPRINT_ONE_FILE_PER_RISC"
echo ""

# Clean previous debug output
rm -f elementwise_kernel_debug.log
rm -rf /home/tt-metal-apv/generated/dprint/

# Build with debug flags
echo "🔨 Building with debug flags..."
cd /home/tt-metal-apv
cmake --build build-cmake --target alexp_distributed_elementwise_add -j$(nproc)

# Check build status
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check compilation errors."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Run the program with debug output
echo "🚀 Running Distributed Elementwise Add with kernel debugging..."
echo "   Debug output will be in: $TT_METAL_DPRINT_FILE"
echo "   Per-RISC debug files will be in: /home/tt-metal-apv/generated/dprint/"
echo ""

cd "$SCRIPT_DIR"

# Test with 16 tiles to reproduce the issue
echo "Testing with 16 tiles (known failure case):"
/home/tt-metal-apv/build-cmake/tt_metal/programming_examples/alexp_examples/e4/elementwise_add/programming_examples/alexp_examples/e4/elementwise_add/alexp_distributed_elementwise_add 16 2>&1 | tee elementwise_run.log

echo ""
echo "📊 Debug Results:"
echo "=================="

# Show debug files if they exist
if [ -d "/home/tt-metal-apv/generated/dprint/" ]; then
    echo "🔍 Per-RISC debug files created:"
    ls -la /home/tt-metal-apv/generated/dprint/
    echo ""

    echo "📋 NCRISC (Reader Kernel) Debug Output:"
    echo "----------------------------------------"
    find /home/tt-metal-apv/generated/dprint/ -name "*NCRISC*" -exec head -50 {} \; 2>/dev/null || echo "No NCRISC debug output found"
    echo ""

    echo "📋 TRISC (Compute Kernel) Debug Output:"
    echo "---------------------------------------"
    find /home/tt-metal-apv/generated/dprint/ -name "*TRISC*" -exec head -50 {} \; 2>/dev/null || echo "No TRISC debug output found"
    echo ""

    echo "📋 BRISC (Writer Kernel) Debug Output:"
    echo "--------------------------------------"
    find /home/tt-metal-apv/generated/dprint/ -name "*BRISC*" -exec head -50 {} \; 2>/dev/null || echo "No BRISC debug output found"

elif [ -f "$TT_METAL_DPRINT_FILE" ]; then
    echo "🔍 Kernel debug output (first 100 lines):"
    head -100 "$TT_METAL_DPRINT_FILE"
    echo ""
    echo "📊 Debug file stats:"
    wc -l "$TT_METAL_DPRINT_FILE"
else
    echo "⚠️  No debug output found. Check if hardware is available or try CPU simulation mode."
fi

echo ""
echo "📁 Debug files saved:"
echo "   - Program output: elementwise_run.log"
echo "   - Kernel debug: $TT_METAL_DPRINT_FILE"
echo "   - Per-RISC debug: /home/tt-metal-apv/generated/dprint/"
echo ""
echo "🔧 Next steps:"
echo "   1. Examine NCRISC (reader) debug output for CB1 overflow"
echo "   2. Check TRISC (compute) debug output for synchronization issues"
echo "   3. Look for patterns in critical tiles (14-15)"
echo "   4. Compare successful vs failed tile processing"
