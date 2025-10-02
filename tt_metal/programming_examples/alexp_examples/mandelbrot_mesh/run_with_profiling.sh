#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Run Mandelbrot Mesh with Full RISC-V Profiling

echo "🎯 Mandelbrot Mesh - RISC-V Profiling Demo"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Enable profiling
echo "🔧 Enabling comprehensive RISC-V profiling..."
source ./enable_full_profiling.sh

echo ""
echo "🚀 Running Mandelbrot computation with profiling..."

# Option 1: Run C++ version
if [ "$1" = "cpp" ] || [ "$1" = "" ]; then
    echo "📊 Running C++ version with profiling..."
    ./build_and_run.sh
fi

# Option 2: Run Python version
if [ "$1" = "python" ] || [ "$1" = "py" ]; then
    echo "📊 Running Python version with profiling..."
    python3 python_mandelbrot_mesh.py
fi

# Option 3: Run both
if [ "$1" = "both" ]; then
    echo "📊 Running both C++ and Python versions with profiling..."
    echo ""
    echo "--- C++ Version ---"
    ./build_and_run.sh
    echo ""
    echo "--- Python Version ---"
    python3 python_mandelbrot_mesh.py
fi

echo ""
echo "✅ Profiling completed!"
echo ""
echo "📈 Profiling results available in:"
echo "   Directory: $TT_METAL_PROFILER_OUTPUT_DIR"
echo ""

# Check for profiling output files
if [ -d "$TT_METAL_PROFILER_OUTPUT_DIR" ]; then
    echo "📁 Generated profiling files:"
    ls -la "$TT_METAL_PROFILER_OUTPUT_DIR" | grep -E "\.(csv|log|json)$" | while read -r line; do
        echo "   $line"
    done

    echo ""
    echo "🔍 Quick analysis:"

    # Show kernel profiling summary if available
    for file in "$TT_METAL_PROFILER_OUTPUT_DIR"/*.csv; do
        if [ -f "$file" ]; then
            echo "   📊 $(basename "$file"): $(wc -l < "$file") entries"
        fi
    done

    echo ""
    echo "📖 View profiling guide: RISCV_PROFILING_GUIDE.md"
    echo "🛠️  Analyze results with:"
    echo "   cat $TT_METAL_PROFILER_OUTPUT_DIR/*.csv | head -10"
    echo "   python -m tt_metal.tools.profiler.process_device_log $TT_METAL_PROFILER_OUTPUT_DIR/"

else
    echo "⚠️  No profiling output found. Check that:"
    echo "   • TT hardware is available"
    echo "   • Profiling environment variables are set"
    echo "   • Application ran successfully"
fi

echo ""
echo "🎉 RISC-V Profiling Demo Complete!"
