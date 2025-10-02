# How to Measure Performance on Tenstorrent Hardware

## Quick Summary

Here are the main ways to measure performance, cycles, and timing on Tenstorrent hardware:

### 1. **Host-Side Timing (Microseconds/Milliseconds)**
```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
// Your TT-Metal operation here
EnqueueProgram(cq, program, false);
Finish(cq);
auto end = std::chrono::high_resolution_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
double ms = duration.count() / 1000.0;
std::cout << "Execution time: " << ms << " ms" << std::endl;
```

### 2. **Device-Side Profiling (Detailed Analysis)**
```bash
# Set environment variables
export TT_METAL_PROFILER=1
export TT_METAL_PROFILER_DISPATCH=1
export TT_METAL_HOME=/home/tt-metal-apv

# Build with profiler support
cmake --build build-cmake --target your_app

# Run your application - profiler results will be saved as CSV files
```

```cpp
// In your C++ code
#include "tt-metalium/tt_metal_profiler.hpp"

// Initialize profiler
tt::tt_metal::detail::InitDeviceProfiler(device);
tt::tt_metal::detail::SetDeviceProfilerDir("./profiler_results");

// Your computation here
EnqueueProgram(cq, program, false);
Finish(cq);

// Read profiler results
std::map<std::pair<chip_id_t, uint32_t>, std::string> runtime_map;
ProfilerOptionalMetadata metadata(std::move(runtime_map));
tt::tt_metal::detail::ReadDeviceProfilerResults(device, ProfilerReadState::NORMAL, metadata);
```

### 3. **Kernel Cycle Counting (Hardware Cycles)**
```cpp
// In your compute kernel (.cpp file)
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    uint64_t start_cycles = get_cycle_count();

    // Your computation here
    for (uint32_t i = 0; i < num_iterations; ++i) {
        // Mandelbrot computation or other work
    }

    uint64_t end_cycles = get_cycle_count();
    uint64_t elapsed_cycles = end_cycles - start_cycles;

    // Store cycles in output buffer for host readback
}
}
```

### 4. **Python Performance Measurement Script**
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh

# Set required environment variable
export TT_METAL_HOME=/home/tt-metal-apv

# Run performance measurement
python3 measure_mandelbrot_performance.py --size 1024 --runs 5 --include-cpu
```

## Environment Setup

**Required Environment Variables:**
```bash
export TT_METAL_HOME=/home/tt-metal-apv
export ARCH_NAME=wormhole_b0  # or your specific architecture

# For profiling (optional)
export TT_METAL_PROFILER=1
export TT_METAL_PROFILER_DISPATCH=1
```

## Performance Metrics You Can Measure

### 1. **Execution Time**
- **Host-side timing**: Overall end-to-end time
- **Kernel execution time**: Time spent in actual computation
- **Memory transfer time**: Time for host-device data movement

### 2. **Throughput Metrics**
- **Pixels per second**: For image processing applications
- **Operations per second**: Total computational operations
- **GFLOPS**: Billion floating-point operations per second
- **Memory bandwidth**: GB/s data movement rate

### 3. **Hardware Utilization**
- **Core utilization**: Percentage of cores actively computing
- **Memory bandwidth utilization**: Percentage of theoretical bandwidth used
- **Dispatch overhead**: Time spent in command dispatch vs computation

### 4. **Cycle-Level Metrics**
- **Total cycles**: Complete kernel execution cycles
- **Cycles per operation**: Average cycles needed per computation
- **Memory access cycles**: Cycles spent in memory operations
- **Compute cycles**: Cycles spent in actual computation

## Typical Performance Ranges

**For Mandelbrot Set (1024x1024, 100 iterations):**

| Implementation | Execution Time | Speedup | Notes |
|----------------|----------------|---------|--------|
| CPU Reference | ~500-1000ms | 1x | Single-threaded Python/C++ |
| Single TT Core | ~200-300ms | 2-3x | One Tensix core |
| 64 TT Cores (1 device) | ~20-50ms | 10-25x | Full device utilization |
| 64 Cores × 8 devices | ~5-15ms | 30-100x | Full mesh utilization |

## Hardware Specifications

**Per Tenstorrent Device:**
- **Tensix Cores**: Up to 64 compute cores (8×8 grid)
- **Clock Frequency**: ~1 GHz
- **DRAM Bandwidth**: ~100+ GB/s
- **L1 Memory per Core**: ~1 MB with ~1000+ GB/s bandwidth

## Tools and Files Available

### 1. **Performance Measurement Library**
- `performance_measurement.hpp` - C++ performance measurement utilities
- Provides timing, profiling, and metric calculation functions

### 2. **Benchmark Applications**
- `mandelbrot_multi_core_benchmark.cpp` - Enhanced multi-core implementation with built-in performance measurement
- `benchmark_all_implementations.cpp` - Comprehensive benchmark suite comparing different implementations

### 3. **Python Performance Script**
- `measure_mandelbrot_performance.py` - Easy-to-use Python script for performance measurement
- Supports multiple runs, statistical analysis, and CPU comparison

### 4. **Kernel with Cycle Counting**
- `mandelbrot_with_cycles.cpp` - Compute kernel with detailed cycle measurement
- Provides cycle-level performance analysis

### 5. **Shell Scripts**
- `run_performance_benchmarks.sh` - Automated benchmark execution script
- Handles environment setup and result collection

### 6. **Documentation**
- `TENSTORRENT_PERFORMANCE_GUIDE.md` - Comprehensive performance measurement guide
- `PERFORMANCE_MEASUREMENT_GUIDE.md` - Detailed usage instructions

## Quick Start Commands

### 1. **Basic Performance Measurement**
```bash
# Set environment
export TT_METAL_HOME=/home/tt-metal-apv

# Build executables
cd /home/tt-metal-apv
cmake --build build-cmake --target mandelbrot_multi_core_mesh -j$(nproc)

# Run with timing
cd tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
time /home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_multi_core_mesh
```

### 2. **Python Performance Analysis**
```bash
export TT_METAL_HOME=/home/tt-metal-apv
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh

# Quick test
python3 measure_mandelbrot_performance.py --size 512 --runs 3

# Comprehensive benchmark
python3 measure_mandelbrot_performance.py --size 1024 --runs 5 --include-cpu
```

### 3. **Profiling with Detailed Analysis**
```bash
# Enable profiling
export TT_METAL_PROFILER=1
export TT_METAL_PROFILER_DISPATCH=1
export TT_METAL_HOME=/home/tt-metal-apv

# Run with profiling
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
./run_performance_benchmarks.sh --profile --size 1024
```

## Understanding Results

### Good Performance Indicators:
- **High throughput**: >10M pixels/second for Mandelbrot
- **Good speedup**: 10-50x vs CPU reference
- **High core utilization**: >90% of available cores active
- **Low variance**: Consistent timing across multiple runs

### Performance Issues to Watch For:
- **Memory bottlenecks**: High memory access time in profiler
- **Load imbalance**: Some cores idle while others work
- **Dispatch overhead**: >10% of total execution time
- **Poor scaling**: Speedup doesn't increase with more cores

## Key Performance Optimization Tips

1. **Use All Available Cores**: Configure for 64 cores per device (8×8 grid)
2. **Optimize Memory Access**: Sequential access patterns are faster
3. **Use Fixed-Point Arithmetic**: Often faster than floating-point on TT hardware
4. **Minimize Host-Device Sync**: Batch operations to reduce overhead
5. **Profile Regularly**: Use profiler to identify bottlenecks

## Files You Can Run Right Now

1. **Python Performance Script** (Works immediately):
   ```bash
   export TT_METAL_HOME=/home/tt-metal-apv
   cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
   python3 measure_mandelbrot_performance.py --help
   ```

2. **Existing Mandelbrot Implementations** (If environment is set):
   ```bash
   export TT_METAL_HOME=/home/tt-metal-apv
   time /home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_multi_core_mesh
   ```

3. **Performance Guides** (Read immediately):
   - `TENSTORRENT_PERFORMANCE_GUIDE.md` - Comprehensive guide
   - `PERFORMANCE_MEASUREMENT_GUIDE.md` - Detailed instructions

The key to measuring performance on Tenstorrent hardware is using the right tool for your needs: host-side timing for overall performance, device profiling for detailed analysis, and cycle counting for fine-grained optimization.
