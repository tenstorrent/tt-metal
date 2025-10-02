#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# TT-Metal Comprehensive RISC-V Profiling Setup Script

echo "🔧 Enabling comprehensive TT-Metal RISC-V profiling..."

# Core profiler settings
export TT_METAL_PROFILER=1
export TT_METAL_DEVICE_PROFILER=1
export PROFILE_KERNEL=1

# Disable conflicting features
unset TT_METAL_DPRINT_CORES

# Enable all profiler types
export TT_METAL_KERNEL_PROFILER=1
export TT_METAL_NOC_EVENT_PROFILER=1
export TT_METAL_DISPATCH_PROFILER=1
export TT_METAL_FABRIC_EVENT_PROFILER=1
export TT_METAL_CUSTOM_CYCLE_PROFILER=1

# Profiler configuration
export TT_METAL_PROFILER_CORES="all"
export TT_METAL_PROFILER_TRACE=1
export TT_METAL_PROFILER_BUFFER_SIZE=128
export TT_METAL_PROFILER_LOG_LEVEL=DEBUG

# Output settings
export TT_METAL_PROFILER_OUTPUT_DIR="./profiler_output"
export TT_METAL_PROFILER_CSV=1
export TT_METAL_PROFILER_FILE_PREFIX="mandelbrot_riscv_profile"

# Tracy integration (optional)
export TT_METAL_TRACY_ENABLE=1

# Create output directory
mkdir -p $TT_METAL_PROFILER_OUTPUT_DIR

echo "✅ Profiling enabled! Output will be in: $TT_METAL_PROFILER_OUTPUT_DIR"
echo ""
echo "📊 Enabled profiling types:"
echo "   • Kernel Profiler (RISC-V execution)"
echo "   • NOC Event Profiler (Network-on-Chip)"
echo "   • Dispatch Core Profiler (Control flow)"
echo "   • Fabric Event Profiler (Mesh communication)"
echo "   • Custom Cycle Profiler (Detailed timing)"
echo ""
echo "🎯 Environment variables set:"
echo "   TT_METAL_PROFILER = $TT_METAL_PROFILER"
echo "   PROFILE_KERNEL = $PROFILE_KERNEL"
echo "   TT_METAL_PROFILER_CORES = $TT_METAL_PROFILER_CORES"
echo "   TT_METAL_PROFILER_OUTPUT_DIR = $TT_METAL_PROFILER_OUTPUT_DIR"
echo ""
echo "🚀 Ready to run TT-Metal applications with full profiling!"
echo ""
echo "Usage examples:"
echo "   ./build_and_run.sh                    # Run C++ version"
echo "   python3 python_mandelbrot_mesh.py    # Run Python version"
echo ""
echo "📈 After execution, check profiler_output/ for detailed RISC-V analysis"
