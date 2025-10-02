#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Debug All RISC-V Cores - Enhanced Runtime Debug Output

echo "🔍 Enhanced RISC-V Core Debug Mode"
echo "===================================="
echo "Showing: BRISC, NCRISC, TRISC0, TRISC1, TRISC2, ERISC"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Setting up enhanced RISC-V debugging environment..."

# Set TT_METAL_HOME (required)
export TT_METAL_HOME="/home/tt-metal-apv"

# CRITICAL: Enable DPRINT for ALL RISC-V cores
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
export TT_METAL_DPRINT_FILE="./risc_cores_debug.log"
export TT_METAL_DPRINT_PRINT_ALL=1

# Enable slow dispatch for better debugging
export TT_METAL_SLOW_DISPATCH_MODE=1

echo "✅ Enhanced RISC-V debug environment configured:"
echo "   TT_METAL_HOME = $TT_METAL_HOME"
echo "   TT_METAL_DPRINT_CORES = $TT_METAL_DPRINT_CORES"
echo "   TT_METAL_DPRINT_FILE = $TT_METAL_DPRINT_FILE"
echo ""

# Clean previous debug output
rm -f risc_cores_debug.log

# Build with debug flags
echo "🔨 Building with enhanced debug flags..."
cd /home/tt-metal-apv
make -C build-cmake programming_examples/mandelbrot_mesh

# Check build status
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check compilation errors."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Run the program with enhanced debug output
echo "🚀 Running Mandelbrot with ALL RISC-V core debugging..."
echo "   Debug output will be in: $TT_METAL_DPRINT_FILE"
echo ""

cd "$SCRIPT_DIR"

# Run and capture both stdout and debug output
/home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_mesh 2>&1 | tee mandelbrot_risc_run.log

echo ""
echo "📊 Enhanced RISC-V Core Debug Results:"
echo "======================================"

# Show debug file if it exists
if [ -f "$TT_METAL_DPRINT_FILE" ]; then
    echo "🔍 RISC-V Core debug output:"
    echo ""

    # Show core startup messages
    echo "=== CORE STARTUP MESSAGES ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "KERNEL STARTED" | head -20
    echo ""

    # Show core type identification
    echo "=== CORE TYPE IDENTIFICATION ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "CORE TYPE:" | head -20
    echo ""

    # Show TRISC core details
    echo "=== TRISC CORE DETAILS ==="
    cat "$TT_METAL_DPRINT_FILE" | grep -E "(TRISC0|TRISC1|TRISC2)" | head -20
    echo ""

    # Show BRISC/NCRISC details
    echo "=== DATAFLOW CORE DETAILS ==="
    cat "$TT_METAL_DPRINT_FILE" | grep -E "(BRISC|NCRISC|ERISC)" | head -20
    echo ""

    # Show runtime core activity
    echo "=== RUNTIME CORE ACTIVITY ==="
    cat "$TT_METAL_DPRINT_FILE" | grep -E "\[(BRISC|NCRISC|TRISC|ERISC)" | head -30
    echo ""

    echo "📁 Full RISC-V core debug log: $TT_METAL_DPRINT_FILE"
    echo "📊 Total debug lines: $(wc -l < "$TT_METAL_DPRINT_FILE")"

    # Core activity summary
    echo ""
    echo "📈 RISC-V Core Activity Summary:"
    echo "BRISC cores: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "BRISC")"
    echo "NCRISC cores: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "NCRISC")"
    echo "TRISC0 cores: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "TRISC0")"
    echo "TRISC1 cores: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "TRISC1")"
    echo "TRISC2 cores: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "TRISC2")"
    echo "ERISC cores: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "ERISC")"

else
    echo "⚠️  No RISC-V core debug output found at $TT_METAL_DPRINT_FILE"
    echo "   This could mean:"
    echo "   • Kernels didn't execute"
    echo "   • DPRINT is not working"
    echo "   • No TT hardware available"
fi

echo ""
echo "🎯 RISC-V Core Analysis Commands:"
echo "=================================="
echo "# Show all core types:"
echo "cat $TT_METAL_DPRINT_FILE | grep 'CORE TYPE'"
echo ""
echo "# Show TRISC activity:"
echo "cat $TT_METAL_DPRINT_FILE | grep 'TRISC'"
echo ""
echo "# Show BRISC/NCRISC activity:"
echo "cat $TT_METAL_DPRINT_FILE | grep -E '(BRISC|NCRISC)'"
echo ""
echo "# Show runtime core activity:"
echo "cat $TT_METAL_DPRINT_FILE | grep -E '\[(BRISC|NCRISC|TRISC)'"
echo ""
echo "🎉 Enhanced RISC-V Core Debug Complete!"
