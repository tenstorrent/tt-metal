#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Debug Mandelbrot Kernels - Enable RISC-V Core Debug Output

echo "🔍 Mandelbrot Kernel Debug Mode"
echo "==============================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Setting up debug environment..."

# Set TT_METAL_HOME (required)
export TT_METAL_HOME="/home/tt-metal-apv"

# CRITICAL: Enable DPRINT for kernel debugging
export TT_METAL_DPRINT_CORES="all"
export TT_METAL_DPRINT_DISABLE_ASSERT=1
export TT_METAL_DPRINT_ENABLE=1

# Disable profiler (conflicts with DPRINT)
unset TT_METAL_PROFILER
unset TT_METAL_DEVICE_PROFILER

# Enable debug logging
export TT_METAL_LOG_LEVEL=Debug
export TT_METAL_LOGGER_LEVEL=Debug

# Set up debug output
export TT_METAL_DPRINT_FILE="./kernel_debug.log"
export TT_METAL_DPRINT_PRINT_ALL=1

# Enable slow dispatch for better debugging
export TT_METAL_SLOW_DISPATCH_MODE=1

echo "✅ Debug environment configured:"
echo "   TT_METAL_DPRINT_CORES = $TT_METAL_DPRINT_CORES"
echo "   TT_METAL_DPRINT_FILE = $TT_METAL_DPRINT_FILE"
echo "   TT_METAL_LOG_LEVEL = $TT_METAL_LOG_LEVEL"
echo ""

# Clean previous debug output
rm -f kernel_debug.log

# Build with debug flags
echo "🔨 Building with debug flags..."
cd /home/tt-metal-apv
make -C build-cmake programming_examples/mandelbrot_mesh

# Check build status
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check compilation errors."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Run the program with debug output
echo "🚀 Running Mandelbrot with kernel debugging..."
echo "   Debug output will be in: $TT_METAL_DPRINT_FILE"
echo "   Live debug output:"
echo ""

cd "$SCRIPT_DIR"

# Run and capture both stdout and debug output
/home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_mesh 2>&1 | tee mandelbrot_run.log

echo ""
echo "📊 Debug Results:"
echo "=================="

# Show debug file if it exists
if [ -f "$TT_METAL_DPRINT_FILE" ]; then
    echo "🔍 Kernel debug output (first 50 lines):"
    head -50 "$TT_METAL_DPRINT_FILE"
    echo ""
    echo "📁 Full kernel debug log: $TT_METAL_DPRINT_FILE"
    echo "📊 Total debug lines: $(wc -l < "$TT_METAL_DPRINT_FILE")"
else
    echo "⚠️  No kernel debug output found at $TT_METAL_DPRINT_FILE"
    echo "   This could mean:"
    echo "   • Kernels didn't execute"
    echo "   • DPRINT is not working"
    echo "   • No TT hardware available"
fi

# Show run log
if [ -f "mandelbrot_run.log" ]; then
    echo ""
    echo "📝 Application output summary:"
    tail -20 mandelbrot_run.log
fi

echo ""
echo "🎯 Debug Analysis:"
echo "=================="
echo "Look for these patterns in the debug output:"
echo "• 'MANDELBROT COMPUTE KERNEL STARTED' - Compute kernels launching"
echo "• 'MANDELBROT WRITER KERNEL STARTED' - Dataflow kernels launching"
echo "• 'Device ID: X' - Which mesh device is processing"
echo "• 'Pixel(x,y) c=(cx,cy) iter=N' - Actual Mandelbrot calculations"
echo "• 'Writing tile X/Y' - Data being written to DRAM"
echo ""
echo "🔧 To see more detailed output:"
echo "   cat $TT_METAL_DPRINT_FILE | grep 'MANDELBROT'"
echo "   cat $TT_METAL_DPRINT_FILE | grep 'Pixel'"
echo "   cat $TT_METAL_DPRINT_FILE | grep 'Device ID'"
