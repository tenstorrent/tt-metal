#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Performance Benchmarking Script for Mandelbrot Implementations
# This script runs comprehensive performance benchmarks and generates reports

set -e

echo "🏁 Mandelbrot Performance Benchmark Suite"
echo "=========================================="

# Default configuration
ENABLE_PROFILING=false
IMAGE_SIZE="1024"
MAX_ITERATIONS="100"
OUTPUT_DIR="./benchmark_results"
BUILD_DIR="/home/tt-metal-apv/build-cmake"
WARMUP_RUNS=2
BENCHMARK_RUNS=5
RUN_CPU=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            ENABLE_PROFILING=true
            shift
            ;;
        --size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --warmup-runs)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --benchmark-runs)
            BENCHMARK_RUNS="$2"
            shift 2
            ;;
        --no-cpu)
            RUN_CPU=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --profile              Enable device profiling (requires PROFILER build)"
            echo "  --size <n>             Image size (width=height, default: 1024)"
            echo "  --iterations <n>       Max iterations (default: 100)"
            echo "  --output-dir <dir>     Output directory (default: ./benchmark_results)"
            echo "  --build-dir <dir>      Build directory (default: /home/tt-metal-apv/build-cmake)"
            echo "  --warmup-runs <n>      Number of warmup runs (default: 2)"
            echo "  --benchmark-runs <n>   Number of benchmark runs (default: 5)"
            echo "  --no-cpu               Skip CPU reference benchmark"
            echo "  --help                 Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📋 Benchmark Configuration:"
echo "  • Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  • Max iterations: $MAX_ITERATIONS"
echo "  • Profiling: $ENABLE_PROFILING"
echo "  • CPU reference: $RUN_CPU"
echo "  • Warmup runs: $WARMUP_RUNS"
echo "  • Benchmark runs: $BENCHMARK_RUNS"
echo "  • Output directory: $OUTPUT_DIR"
echo "  • Build directory: $BUILD_DIR"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Build directory not found: $BUILD_DIR"
    echo "Please build the project first or specify correct build directory with --build-dir"
    exit 1
fi

# Build the benchmark executables
echo "🔨 Building benchmark executables..."
cd /home/tt-metal-apv
cmake --build "$BUILD_DIR" --target mandelbrot_multi_core_benchmark -j$(nproc)
cmake --build "$BUILD_DIR" --target benchmark_all_implementations -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build completed successfully"
echo ""

# Set up environment for profiling if enabled
if [ "$ENABLE_PROFILING" = true ]; then
    echo "🔍 Setting up profiling environment..."
    export TT_METAL_PROFILER=1
    export TT_METAL_PROFILER_DISPATCH=1
    echo "✅ Profiling environment configured"
    echo ""
fi

# Function to run a single benchmark
run_benchmark() {
    local executable=$1
    local benchmark_name=$2
    local extra_args=$3

    echo "🚀 Running $benchmark_name..."
    echo "Command: $executable --width $IMAGE_SIZE --height $IMAGE_SIZE --iterations $MAX_ITERATIONS --warmup-runs $WARMUP_RUNS --benchmark-runs $BENCHMARK_RUNS --output-dir $OUTPUT_DIR $extra_args"
    echo ""

    cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh

    if ! $executable --width "$IMAGE_SIZE" --height "$IMAGE_SIZE" --iterations "$MAX_ITERATIONS" \
                     --warmup-runs "$WARMUP_RUNS" --benchmark-runs "$BENCHMARK_RUNS" \
                     --output-dir "$OUTPUT_DIR" $extra_args; then
        echo "❌ $benchmark_name failed!"
        return 1
    fi

    echo "✅ $benchmark_name completed successfully"
    echo ""
}

# Run individual benchmark (single implementation with detailed profiling)
PROFILING_ARGS=""
if [ "$ENABLE_PROFILING" = true ]; then
    PROFILING_ARGS="--profile"
fi

CPU_ARGS=""
if [ "$RUN_CPU" = false ]; then
    CPU_ARGS="--no-cpu"
fi

echo "🏁 Starting benchmark suite..."
echo ""

# Run the comprehensive benchmark
run_benchmark "${BUILD_DIR}/programming_examples/benchmark_all_implementations" \
              "Comprehensive Benchmark Suite" \
              "$PROFILING_ARGS $CPU_ARGS"

# Run detailed single implementation benchmark if profiling is enabled
if [ "$ENABLE_PROFILING" = true ]; then
    run_benchmark "${BUILD_DIR}/programming_examples/mandelbrot_multi_core_benchmark" \
                  "Detailed Multi-Core Benchmark with Profiling" \
                  "$PROFILING_ARGS --cores 64"
fi

echo "🎉 All benchmarks completed successfully!"
echo ""

# Generate summary report
echo "📊 Generating summary report..."

SUMMARY_FILE="$OUTPUT_DIR/benchmark_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Mandelbrot Performance Benchmark Summary
========================================

Configuration:
- Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE} pixels
- Max Iterations: $MAX_ITERATIONS
- Warmup Runs: $WARMUP_RUNS
- Benchmark Runs: $BENCHMARK_RUNS
- Profiling Enabled: $ENABLE_PROFILING
- CPU Reference: $RUN_CPU
- Timestamp: $(date)

Results:
--------
EOF

# Add CSV results to summary if they exist
if [ -f "$OUTPUT_DIR/comprehensive_comparison.csv" ]; then
    echo "" >> "$SUMMARY_FILE"
    echo "Performance Comparison (from comprehensive_comparison.csv):" >> "$SUMMARY_FILE"
    echo "Implementation,Cores,Total_Time_ms,Pixels_Per_Second,Iterations_Per_Second,Speedup_vs_CPU" >> "$SUMMARY_FILE"
    tail -n +2 "$OUTPUT_DIR/comprehensive_comparison.csv" >> "$SUMMARY_FILE"
fi

echo "📄 Summary report saved to: $SUMMARY_FILE"

# List all generated files
echo ""
echo "📁 Generated files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"

if [ "$ENABLE_PROFILING" = true ]; then
    echo ""
    echo "📈 Device profiling results:"
    echo "Look for CSV files with detailed kernel-level performance data:"
    find "$OUTPUT_DIR" -name "*.csv" -type f | grep -v "benchmark\|comparison" || echo "No profiling CSV files found"
fi

echo ""
echo "🏆 Benchmark suite completed! Check the results in: $OUTPUT_DIR"
echo "Key files to examine:"
echo "  • benchmark_summary.txt - Overall summary"
echo "  • comprehensive_comparison.csv - Performance comparison"
echo "  • *_benchmark.csv - Individual benchmark results"
if [ "$ENABLE_PROFILING" = true ]; then
    echo "  • Device profiling CSV files - Kernel-level performance data"
fi
