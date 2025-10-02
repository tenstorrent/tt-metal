# Mandelbrot Performance Measurement Guide

This guide explains how to measure performance and cycles for the Mandelbrot implementations using the comprehensive benchmarking tools provided.

## Overview

The performance measurement system provides multiple levels of analysis:

1. **Host-side timing** - Measures execution time from the host perspective
2. **Device-side profiling** - Uses TT-Metal profiler APIs for detailed kernel analysis
3. **Cycle counting** - Kernel-level cycle measurements for fine-grained analysis
4. **Throughput analysis** - Pixels/second and iterations/second metrics
5. **Comparative benchmarking** - Compare different implementations and configurations

## Tools Available

### 1. Performance Measurement Library (`performance_measurement.hpp`)

A comprehensive C++ library providing:
- High-resolution timing measurements
- Device profiler integration
- Benchmark result collection and analysis
- CSV export capabilities
- Automatic performance metric calculations

### 2. Enhanced Implementations

#### `mandelbrot_multi_core_benchmark.cpp`
- Multi-core implementation with integrated performance measurement
- Supports different core counts (1, 4, 16, 64 cores per device)
- Device profiling integration
- Detailed timing breakdown

#### `benchmark_all_implementations.cpp`
- Comprehensive benchmark suite
- Compares CPU reference vs TT-Metal implementations
- Tests multiple core configurations
- Generates comparative analysis reports

### 3. Benchmark Script (`run_performance_benchmarks.sh`)

Automated benchmark execution with:
- Configurable parameters (image size, iterations, core counts)
- Profiling enable/disable
- Multiple warmup and benchmark runs
- Automatic report generation

## Usage Instructions

### Quick Start

```bash
# Run basic benchmark suite
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
./run_performance_benchmarks.sh

# Run with profiling enabled (requires PROFILER build)
./run_performance_benchmarks.sh --profile

# Run with custom parameters
./run_performance_benchmarks.sh --size 2048 --iterations 200 --benchmark-runs 10
```

### Manual Execution

```bash
# Build the benchmark executables
cd /home/tt-metal-apv
cmake --build build-cmake --target mandelbrot_multi_core_benchmark -j$(nproc)
cmake --build build-cmake --target benchmark_all_implementations -j$(nproc)

# Run comprehensive benchmark
./build-cmake/programming_examples/benchmark_all_implementations \
    --width 1024 --height 1024 --iterations 100 \
    --warmup-runs 2 --benchmark-runs 5 \
    --output-dir ./results

# Run detailed single implementation benchmark with profiling
./build-cmake/programming_examples/mandelbrot_multi_core_benchmark \
    --profile --cores 64 --width 1024 --height 1024 \
    --output-dir ./results
```

### Command Line Options

#### Common Options (both benchmark tools)
- `--width <n>` / `--height <n>`: Image dimensions (default: 1024x1024)
- `--iterations <n>`: Maximum Mandelbrot iterations (default: 100)
- `--output-dir <dir>`: Output directory for results (default: ./benchmark_results)
- `--warmup-runs <n>`: Number of warmup runs (default: 2)
- `--benchmark-runs <n>`: Number of benchmark runs for averaging (default: 5)

#### Profiling Options
- `--profile`: Enable device-side profiling (requires PROFILER build)
- `--no-cpu`: Skip CPU reference benchmark (TT-Metal only)

#### Implementation-Specific Options
- `--cores <n>`: Cores per device for multi-core benchmark (1, 4, 16, 64)

## Performance Metrics Explained

### Host-Side Timing Metrics

1. **mesh_device_init**: Time to initialize the mesh device
2. **buffer_setup**: Time to create and configure buffers
3. **workload_creation**: Time to create programs and mesh workload
4. **kernel_execution**: Time for actual kernel execution on device
5. **result_readback**: Time to read results back to host
6. **image_save**: Time to save results as PPM image

### Device-Side Profiling Metrics

When profiling is enabled, detailed CSV files are generated with:

1. **Kernel Execution Times**: Per-kernel timing data
2. **Memory Bandwidth**: DRAM and L1 memory utilization
3. **Core Utilization**: Usage statistics for each Tensix core
4. **Dispatch Overhead**: Time spent in command dispatch

### Throughput Metrics

1. **Pixels per Second**: Total pixels processed per second
2. **Iterations per Second**: Total Mandelbrot iterations computed per second
3. **Speedup vs CPU**: Performance improvement over CPU reference

### Cycle-Level Metrics

The enhanced kernel (`mandelbrot_with_cycles.cpp`) provides:

1. **Total Cycles**: Complete kernel execution cycles
2. **Compute Cycles**: Cycles spent in actual computation
3. **Memory Cycles**: Cycles spent in memory operations
4. **Cycles per Pixel**: Average cycles needed per pixel
5. **Cycles per Iteration**: Average cycles per Mandelbrot iteration

## Understanding the Results

### Output Files

After running benchmarks, you'll find these files in the output directory:

```
benchmark_results/
├── benchmark_summary.txt              # Overall summary
├── comprehensive_comparison.csv        # Performance comparison table
├── CPU-Reference_benchmark.csv        # CPU benchmark details
├── Multi-Core-Mesh-64cores_benchmark.csv  # TT-Metal benchmark details
├── Single-Core-Mesh-1cores_benchmark.csv  # Single-core benchmark details
└── [profiler_*.csv]                   # Device profiling data (if enabled)
```

### Interpreting Performance Data

#### Good Performance Indicators:
- **High Pixels/Second**: Indicates efficient computation
- **Low Memory Cycles %**: Good memory access patterns
- **High Core Utilization**: Effective parallelization
- **Consistent Timing**: Low variance across benchmark runs

#### Performance Optimization Opportunities:
- **High Memory Cycles**: May indicate memory bandwidth bottlenecks
- **Low Speedup vs CPU**: Parallelization or kernel efficiency issues
- **High Dispatch Overhead**: Too many small kernel launches

### Example Interpretation

```
Implementation: Multi-Core-Mesh-64cores
Total Time: 45.2ms
Pixels/Second: 2.3e+07
Speedup vs CPU: 15.6x
```

This shows:
- 64-core implementation processes ~23 million pixels/second
- Achieves 15.6x speedup compared to CPU
- Total execution time of 45.2ms for the computation

## Advanced Usage

### Custom Performance Measurement

You can integrate the performance measurement library in your own code:

```cpp
#include "performance_measurement.hpp"

// Initialize performance measurement
PerformanceMeasurement perf(true, "./my_results");  // Enable profiling
auto& benchmark = perf.create_benchmark("My-Implementation");

// Measure specific operations
double setup_time = perf.measure_execution_time("setup", [&]() {
    // Your setup code here
});

// Add timing to benchmark
perf.add_timing(benchmark, "setup", setup_time);

// Print and save results
perf.print_benchmark_results(benchmark);
perf.save_benchmark_to_csv(benchmark);
```

### Kernel-Level Cycle Counting

To add cycle counting to your kernels:

```cpp
// In your compute kernel
uint64_t start_cycles = get_cycle_count();

// Your computation here

uint64_t end_cycles = get_cycle_count();
uint64_t elapsed_cycles = end_cycles - start_cycles;
```

### Profiler Build Requirements

To use device-side profiling, you need a PROFILER build:

```bash
# Build with profiler support
export TT_METAL_PROFILER=1
export TT_METAL_PROFILER_DISPATCH=1
cmake --build build-cmake --target your_target
```

## Performance Optimization Tips

### Host-Side Optimizations

1. **Minimize Host-Device Synchronization**: Use asynchronous operations where possible
2. **Batch Operations**: Combine multiple small operations into larger ones
3. **Memory Management**: Pre-allocate buffers and reuse them

### Device-Side Optimizations

1. **Core Utilization**: Use all available cores effectively
2. **Memory Access Patterns**: Optimize for sequential access
3. **Fixed-Point Arithmetic**: Use fixed-point instead of floating-point for better performance
4. **Loop Unrolling**: Unroll inner loops for better instruction throughput

### Kernel Optimizations

1. **Minimize Divergent Branches**: Reduce conditional statements in tight loops
2. **Optimize Memory Layout**: Use appropriate data formats and alignment
3. **Circular Buffer Management**: Efficiently manage input/output buffers

## Troubleshooting

### Common Issues

1. **Profiler Not Working**: Ensure you have a PROFILER build and set environment variables
2. **Build Failures**: Check that all dependencies are properly linked
3. **Performance Regression**: Compare with baseline measurements and check for configuration changes

### Debug Options

1. **Verbose Output**: Most tools support verbose logging
2. **Single Run Mode**: Use `--benchmark-runs 1` for quick testing
3. **CPU Only Mode**: Use `--no-cpu false` to test CPU implementation only

## Best Practices

1. **Consistent Environment**: Run benchmarks in consistent system conditions
2. **Multiple Runs**: Always use multiple runs for statistical significance
3. **Warmup Runs**: Include warmup runs to account for initialization overhead
4. **Document Configuration**: Record system configuration and build settings
5. **Version Control**: Track benchmark results over time to detect regressions

## Further Reading

- [TT-Metal Profiler Documentation](https://docs.tenstorrent.com/ttnn/latest/profiler.html)
- [Performance Optimization Guide](https://docs.tenstorrent.com/ttnn/latest/performance.html)
- [Kernel Development Best Practices](https://docs.tenstorrent.com/ttnn/latest/kernel_development.html)
