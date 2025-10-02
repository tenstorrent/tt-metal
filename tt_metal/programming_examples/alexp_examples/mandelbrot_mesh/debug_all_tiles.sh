#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Debug ALL Tiles - Print Every Single Tile Processing

echo "🔍 ALL TILES DEBUG MODE"
echo "======================="
echo "WARNING: This will generate MASSIVE debug output!"
echo "Each device processes 4096 tiles × 3 TRISC cores = 12,288 debug lines per device"
echo "8 devices × 12,288 = 98,304 tile processing messages"
echo "Plus 1000 dataflow messages per device = 8,000 more messages"
echo "Total: ~106,304 debug messages!"
echo ""

read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled - that's probably wise!"
    exit 0
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Setting up ALL TILES debugging environment..."

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
export TT_METAL_DPRINT_FILE="./all_tiles_debug.log"
export TT_METAL_DPRINT_PRINT_ALL=1

# Enable slow dispatch for better debugging
export TT_METAL_SLOW_DISPATCH_MODE=1

echo "✅ ALL TILES debug environment configured:"
echo "   TT_METAL_DPRINT_FILE = $TT_METAL_DPRINT_FILE"
echo ""

# Clean previous debug output
rm -f all_tiles_debug.log

# Build with debug flags
echo "🔨 Building with ALL TILES debug flags..."
cd /home/tt-metal-apv
make -C build-cmake programming_examples/mandelbrot_mesh

# Check build status
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check compilation errors."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Run the program with ALL TILES debug output
echo "🚀 Running Mandelbrot with ALL TILES debugging..."
echo "   This will take longer due to massive debug output..."
echo "   Debug output will be in: $TT_METAL_DPRINT_FILE"
echo ""

cd "$SCRIPT_DIR"

# Run and capture both stdout and debug output
timeout 120s /home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_mesh 2>&1 | tee mandelbrot_all_tiles_run.log

echo ""
echo "📊 ALL TILES Debug Results:"
echo "=========================="

# Show debug file if it exists
if [ -f "$TT_METAL_DPRINT_FILE" ]; then
    echo "🔍 ALL TILES debug output generated:"
    echo ""

    # Show file size
    echo "📁 Debug log size: $(du -h "$TT_METAL_DPRINT_FILE" | cut -f1)"
    echo "📊 Total debug lines: $(wc -l < "$TT_METAL_DPRINT_FILE")"
    echo ""

    # Show first few tiles from each core type
    echo "=== FIRST 10 TRISC0 TILES ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "TRISC0.*Processing tile" | head -10
    echo ""

    echo "=== FIRST 10 TRISC1 TILES ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "TRISC1.*Processing tile" | head -10
    echo ""

    echo "=== FIRST 10 TRISC2 TILES ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "TRISC2.*Processing tile" | head -10
    echo ""

    echo "=== FIRST 10 BRISC TILES ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "BRISC.*Writing tile" | head -10
    echo ""

    # Show some middle tiles
    echo "=== MIDDLE RANGE TILES (around tile 2000) ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "Processing tile 20[0-9][0-9]" | head -10
    echo ""

    # Show final tiles
    echo "=== FINAL TILES (4090+) ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "Processing tile 409[0-9]" | head -10
    echo ""

    # Tile processing summary
    echo "📈 Tile Processing Summary:"
    echo "TRISC0 tiles: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "TRISC0.*Processing tile")"
    echo "TRISC1 tiles: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "TRISC1.*Processing tile")"
    echo "TRISC2 tiles: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "TRISC2.*Processing tile")"
    echo "BRISC tiles: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "BRISC.*Writing tile")"
    echo ""

    # Completion summary
    echo "📈 Tile Completion Summary:"
    echo "Compute completions: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "Completed tile.*/")"
    echo "Dataflow completions: $(cat "$TT_METAL_DPRINT_FILE" | grep -c "BRISC.*COMPLETED")"

else
    echo "⚠️  No ALL TILES debug output found at $TT_METAL_DPRINT_FILE"
    echo "   This could mean:"
    echo "   • Kernels didn't execute"
    echo "   • DPRINT is not working"
    echo "   • Program timed out due to massive output"
fi

echo ""
echo "🎯 ALL TILES Analysis Commands:"
echo "==============================="
echo "# Count all tile processing messages:"
echo "cat $TT_METAL_DPRINT_FILE | grep -c 'Processing tile'"
echo ""
echo "# Show tiles from specific device (e.g., device 6):"
echo "cat $TT_METAL_DPRINT_FILE | grep '^6:' | grep 'Processing tile' | head -20"
echo ""
echo "# Show all tiles from TRISC1 cores:"
echo "cat $TT_METAL_DPRINT_FILE | grep 'TRISC1.*Processing tile'"
echo ""
echo "# Show tile range (e.g., tiles 100-110):"
echo "cat $TT_METAL_DPRINT_FILE | grep 'Processing tile 1[0-1][0-9]'"
echo ""
echo "🎉 ALL TILES Debug Complete!"
echo "Full log: $TT_METAL_DPRINT_FILE"
