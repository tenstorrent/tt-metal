#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Debug Specific Devices - More Manageable Output

echo "🔍 SPECIFIC DEVICES DEBUG MODE"
echo "=============================="
echo "This will debug specific cores to see all 8 devices working"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Setting up targeted device debugging..."

# Set TT_METAL_HOME (required)
export TT_METAL_HOME="/home/tt-metal-apv"

# CRITICAL: Enable DPRINT for specific cores to see all devices
# Format: "chip_id,core_x,core_y" for each device
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
export TT_METAL_DPRINT_FILE="./specific_devices_debug.log"
export TT_METAL_DPRINT_PRINT_ALL=1

# Enable slow dispatch for better debugging
export TT_METAL_SLOW_DISPATCH_MODE=1

echo "✅ Targeted device debug environment configured:"
echo "   TT_METAL_DPRINT_CORES = $TT_METAL_DPRINT_CORES"
echo "   TT_METAL_DPRINT_FILE = $TT_METAL_DPRINT_FILE"
echo ""

# Clean previous debug output
rm -f specific_devices_debug.log

# Build with debug flags
echo "🔨 Building with targeted debug flags..."
cd /home/tt-metal-apv
make -C build-cmake programming_examples/mandelbrot_mesh

# Check build status
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check compilation errors."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Run the program with targeted debug output
echo "🚀 Running Mandelbrot with targeted device debugging..."
echo "   This should show all 8 devices working..."
echo "   Debug output will be in: $TT_METAL_DPRINT_FILE"
echo ""

cd "$SCRIPT_DIR"

# Run and capture both stdout and debug output
timeout 60s /home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_mesh 2>&1 | tee mandelbrot_targeted_run.log

echo ""
echo "📊 Targeted Device Debug Results:"
echo "================================="

# Show debug file if it exists
if [ -f "$TT_METAL_DPRINT_FILE" ]; then
    echo "🔍 Targeted device debug output generated:"
    echo ""

    # Show file size
    echo "📁 Debug log size: $(du -h "$TT_METAL_DPRINT_FILE" | cut -f1)"
    echo "📊 Total debug lines: $(wc -l < "$TT_METAL_DPRINT_FILE")"
    echo ""

    # Show all device startups
    echo "=== ALL DEVICE STARTUPS ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "KERNEL STARTED" | head -20
    echo ""

    # Show all device IDs
    echo "=== ALL DEVICE IDs ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "Device ID:" | sort | uniq
    echo ""

    # Show physical device mapping
    echo "=== PHYSICAL DEVICE MAPPING ==="
    echo "Physical Device -> Logical Device ID:"
    for device in {0..7}; do
        device_id=$(cat "$TT_METAL_DPRINT_FILE" | grep "^${device}:" | grep "Device ID:" | head -1 | sed 's/.*Device ID: //')
        if [ ! -z "$device_id" ]; then
            echo "  Physical Device $device -> Logical Device ID $device_id"
        else
            echo "  Physical Device $device -> ❌ NOT FOUND"
        fi
    done
    echo ""

    # Show coordinate bounds for each device
    echo "=== COORDINATE BOUNDS PER DEVICE ==="
    cat "$TT_METAL_DPRINT_FILE" | grep "Coordinate bounds" | head -8
    echo ""

    # Show first few tiles from each device
    echo "=== FIRST TILES FROM EACH DEVICE ==="
    for device in {0..7}; do
        first_tile=$(cat "$TT_METAL_DPRINT_FILE" | grep "^${device}:" | grep "Processing tile 0/" | head -1)
        if [ ! -z "$first_tile" ]; then
            echo "Device $device: $first_tile"
        else
            echo "Device $device: ❌ NO TILES FOUND"
        fi
    done
    echo ""

    # Device activity summary
    echo "📈 Device Activity Summary:"
    for device in {0..7}; do
        tile_count=$(cat "$TT_METAL_DPRINT_FILE" | grep "^${device}:" | grep -c "Processing tile")
        echo "Physical Device $device: $tile_count tile processing messages"
    done

else
    echo "⚠️  No targeted device debug output found at $TT_METAL_DPRINT_FILE"
fi

echo ""
echo "🎯 Device Analysis Commands:"
echo "============================"
echo "# Show all physical devices:"
echo "cat $TT_METAL_DPRINT_FILE | grep 'KERNEL STARTED' | cut -d':' -f1 | sort | uniq"
echo ""
echo "# Show device ID mapping:"
echo "cat $TT_METAL_DPRINT_FILE | grep 'Device ID:' | sort"
echo ""
echo "# Show tiles from specific device (e.g., device 0):"
echo "cat $TT_METAL_DPRINT_FILE | grep '^0:' | grep 'Processing tile' | head -10"
echo ""
echo "🎉 Targeted Device Debug Complete!"
