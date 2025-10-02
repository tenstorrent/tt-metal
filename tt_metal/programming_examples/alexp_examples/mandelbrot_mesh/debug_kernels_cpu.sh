#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Debug Mandelbrot Kernels - CPU Simulation Mode for Testing Debug Prints

echo "🔍 Mandelbrot Kernel Debug Mode (CPU Simulation)"
echo "================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Setting up debug environment for CPU simulation..."

# Set TT_METAL_HOME (required)
export TT_METAL_HOME="/home/tt-metal-apv"

# CRITICAL: Enable DPRINT for kernel debugging
export TT_METAL_DPRINT_CORES="all"
export TT_METAL_DPRINT_DISABLE_ASSERT=1
export TT_METAL_DPRINT_ENABLE=1

# Force CPU simulation mode for testing
export TT_METAL_SIMULATOR_EN=1
export TT_METAL_SIMULATOR_MODE=1

# Disable profiler (conflicts with DPRINT)
unset TT_METAL_PROFILER
unset TT_METAL_DEVICE_PROFILER

# Enable debug logging
export TT_METAL_LOG_LEVEL=Debug
export TT_METAL_LOGGER_LEVEL=Debug

# Set up debug output
export TT_METAL_DPRINT_FILE="./kernel_debug_cpu.log"
export TT_METAL_DPRINT_PRINT_ALL=1

# Enable slow dispatch for better debugging
export TT_METAL_SLOW_DISPATCH_MODE=1

echo "✅ Debug environment configured:"
echo "   TT_METAL_HOME = $TT_METAL_HOME"
echo "   TT_METAL_DPRINT_CORES = $TT_METAL_DPRINT_CORES"
echo "   TT_METAL_DPRINT_FILE = $TT_METAL_DPRINT_FILE"
echo "   TT_METAL_SIMULATOR_EN = $TT_METAL_SIMULATOR_EN"
echo ""

# Clean previous debug output
rm -f kernel_debug_cpu.log mandelbrot_run_cpu.log

echo "🚀 Running Mandelbrot with kernel debugging (CPU simulation)..."
echo "   Debug output will be in: $TT_METAL_DPRINT_FILE"
echo ""

# Run and capture both stdout and debug output
/home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_mesh 2>&1 | tee mandelbrot_run_cpu.log

echo ""
echo "📊 Debug Results:"
echo "=================="

# Show debug file if it exists
if [ -f "$TT_METAL_DPRINT_FILE" ]; then
    echo "🔍 Kernel debug output:"
    cat "$TT_METAL_DPRINT_FILE"
    echo ""
    echo "📁 Full kernel debug log: $TT_METAL_DPRINT_FILE"
    echo "📊 Total debug lines: $(wc -l < "$TT_METAL_DPRINT_FILE")"
else
    echo "⚠️  No kernel debug output found at $TT_METAL_DPRINT_FILE"
fi

# Show run log
if [ -f "mandelbrot_run_cpu.log" ]; then
    echo ""
    echo "📝 Application output:"
    cat mandelbrot_run_cpu.log
fi

echo ""
echo "🎯 Debug Analysis Complete!"
