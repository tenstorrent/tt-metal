#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Debug Reduced Tiles - Manageable Debug Output

echo "🔍 REDUCED TILES DEBUG MODE"
echo "==========================="
echo "Shows: First 5 tiles + Every 1000th tile + Last tile per device"
echo "Compute: ~15 messages per core (vs 4096 for all tiles)"
echo "Dataflow: ~8 messages per core (vs 1000 for all tiles)"
echo "Total: ~184 tile messages (vs 200K+ for all tiles)"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Setting up reduced tiles debugging environment..."

# Set TT_METAL_HOME (required)
export TT_METAL_HOME="/home/tt-metal-apv"

# Enable DPRINT for all devices but with reduced output
export TT_METAL_DPRINT_CORES="0,0,0;1,0,0;2,0,0;3,0,0;4,0,0;5,0,0;6,0,0;7,0,0"
export TT_METAL_DPRINT_DISABLE_ASSERT=1
export TT_METAL_DPRINT_ENABLE=1

# Disable profiler (conflicts with DPRINT)
unset TT_METAL_PROFILER
unset TT_METAL_DEVICE_PROFILER

# Enable debug logging
export TT_METAL_LOG_LEVEL=Debug
export TT_METAL_LOGGER_LEVEL=Debug

# Set up debug output
export TT_METAL_DPRINT_FILE="./reduced_tiles_debug.log"
export TT_METAL_DPRINT_PRINT_ALL=1

# Enable slow dispatch for better debugging
export TT_METAL_SLOW_DISPATCH_MODE=1

echo "✅ Reduced tiles debug environment configured:"
echo "   TT_METAL_DPRINT_FILE = $TT_METAL_DPRINT_FILE"
echo ""

# Clean previous debug output
rm -f reduced_tiles_debug.log

# Build with debug flags
echo "🔨 Building with reduced tiles debug flags..."
cd /home/tt-metal-apv
make -C build-cmake programming_examples/mandelbrot_mesh

# Check build status
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check compilation errors."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Run the program with reduced debug output
echo "🚀 Running Mandelbrot with reduced tiles debugging..."
echo "   This should complete much faster with manageable output..."
echo "   Debug output will be in: $TT_METAL_DPRINT_FILE"
echo ""

cd "$SCRIPT_DIR"

# Run and capture both stdout and debug output
timeout 45s /home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_mesh 2>&1 | tee mandelbrot_reduced_run.log

echo ""
echo "📊 Reduced Tiles Debug Results:"
echo "==============================="

# Show debug file if it exists
if [ -f "$TT_METAL_DPRINT_FILE" ]; then
    echo "🔍 Reduced tiles debug output generated:"
    echo ""

    # Show file size
    echo "📁 Debug log size: $(du -h "$TT_METAL_DPRINT_FILE" | cut -f1)"
    echo "📊 Total debug lines: $(wc -l < "$TT_METAL_DPRINT_FILE")"
    echo ""

    # Show all device startups
    echo "=== ALL DEVICE STARTUPS ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "KERNEL STARTED" | head -20
    echo ""

    # Show tile processing samples
    echo "=== TILE PROCESSING SAMPLES ==="
    echo "First tiles from each core type:"
    cat "$TT_METAL_DPRINT_FILE" | grep "Processing tile 0/" | head -12
    echo ""
    echo "Progress markers (1000th tiles):"
    cat "$TT_METAL_DPRINT_FILE" | grep "Processing tile.*000/" | head -8
    echo ""
    echo "Final tiles:"
    cat "$TT_METAL_DPRINT_FILE" | grep "Processing tile.*4095/" | head -8
    echo ""

    # Show dataflow samples
    echo "=== DATAFLOW SAMPLES ==="
    echo "First dataflow tiles:"
    cat "$TT_METAL_DPRINT_FILE" | grep "Writing tile 0/" | head -4
    echo ""
    echo "Progress markers (200th tiles):"
    cat "$TT_METAL_DPRINT_FILE" | grep "Writing tile.*00/" | head -4
    echo ""

    # Device activity summary
    echo "📈 Device Activity Summary:"
    for device in {0..7}; do
        tile_count=$(cat "$TT_METAL_DPRINT_FILE" | grep "^${device}:" | grep -c "Processing tile")
        dataflow_count=$(cat "$TT_METAL_DPRINT_FILE" | grep "^${device}:" | grep -c "Writing tile")
        if [ $tile_count -gt 0 ] || [ $dataflow_count -gt 0 ]; then
            echo "Physical Device $device: $tile_count compute tiles, $dataflow_count dataflow tiles"
        else
            echo "Physical Device $device: ❌ NO ACTIVITY"
        fi
    done
    echo ""

    # Completion status
    echo "📈 Completion Status:"
    completed_devices=$(cat "$TT_METAL_DPRINT_FILE" | grep -c "COMPLETED.*wrote.*tiles")
    echo "Devices completed: $completed_devices/8"

    # Show completion messages
    if [ $completed_devices -gt 0 ]; then
        echo ""
        echo "=== COMPLETION MESSAGES ==="
        cat "$TT_METAL_DPRINT_FILE" | grep "COMPLETED.*wrote.*tiles"
    fi

else
    echo "⚠️  No reduced tiles debug output found at $TT_METAL_DPRINT_FILE"
fi

echo ""
echo "🎯 Reduced Tiles Analysis Commands:"
echo "===================================="
echo "# Count tile processing messages:"
echo "cat $TT_METAL_DPRINT_FILE | grep -c 'Processing tile'"
echo ""
echo "# Show specific tile numbers:"
echo "cat $TT_METAL_DPRINT_FILE | grep 'Processing tile' | cut -d' ' -f4 | sort -n | uniq"
echo ""
echo "# Show device completion status:"
echo "cat $TT_METAL_DPRINT_FILE | grep 'COMPLETED'"
echo ""
echo "🎉 Reduced Tiles Debug Complete!"
echo "Much more manageable output with key progress markers!"
