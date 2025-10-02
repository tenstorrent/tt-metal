# Complete Guide: Measuring Performance on Tenstorrent Hardware

## 🚀 Quick Answer: How to Measure Performance

Here are the **working methods** to measure performance, cycles, and timing on Tenstorrent hardware:

### 1. **Host-Side Timing (Works Immediately)**

```cpp
#include <chrono>
#include <iostream>

// Simple timing measurement
auto start = std::chrono::high_resolution_clock::now();

// Your TT-Metal operation here
EnqueueProgram(cq, program, false);
Finish(cq);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

double ms = duration.count() / 1000.0;
double us = duration.count();

std::cout << "Execution time: " << ms << " ms (" << us << " μs)" << std::endl;
```

### 2. **Python Performance Script (Ready to Use)**

```bash
# Set required environment
export TT_METAL_HOME=/home/tt-metal-apv

# Run performance measurement
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
python3 measure_mandelbrot_performance.py --size 1024 --runs 5 --include-cpu
```

This script provides:
- Multiple implementation comparison
- Statistical analysis (multiple runs)
- CPU reference benchmark
- Throughput calculations (pixels/sec, GFLOPS)
- JSON output for further analysis

### 3. **Device Profiling (Detailed Analysis)**

```bash
# Enable profiling environment
export TT_METAL_PROFILER=1
export TT_METAL_PROFILER_DISPATCH=1
export TT_METAL_HOME=/home/tt-metal-apv

# Run your application - profiler automatically generates CSV files
./your_tt_metal_app
```

**Profiler APIs in C++:**
```cpp
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

### 4. **Kernel Cycle Counting (Hardware Level)**

```cpp
// In your compute kernel (.cpp file)
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    // Start cycle counting
    uint64_t start_cycles = get_cycle_count();

    // Your computation here
    for (uint32_t i = 0; i < num_iterations; ++i) {
        // Mandelbrot computation or other work
    }

    // End cycle counting
    uint64_t end_cycles = get_cycle_count();
    uint64_t elapsed_cycles = end_cycles - start_cycles;

    // Store cycles in output buffer for host readback
    // (requires additional buffer setup in host code)
}
}
```

## 📊 Performance Metrics You Can Measure

### **Timing Metrics:**
- **Execution Time**: Milliseconds/microseconds
- **Kernel Time**: Device-side execution time
- **Memory Transfer Time**: Host ↔ Device data movement
- **Dispatch Overhead**: Command queue overhead

### **Throughput Metrics:**
- **Pixels per Second**: For image processing
- **Operations per Second**: Total computational throughput
- **GFLOPS**: Billion floating-point operations per second
- **Memory Bandwidth**: GB/s data transfer rate

### **Hardware Utilization:**
- **Core Utilization**: % of cores actively computing
- **Memory Bandwidth Utilization**: % of theoretical bandwidth
- **Cycle Efficiency**: Compute cycles vs total cycles

## 🔧 Working Tools Available

### **1. Python Performance Script** ✅
- **File**: `measure_mandelbrot_performance.py`
- **Status**: Ready to use
- **Features**: Multi-implementation comparison, statistics, CPU reference

### **2. Existing Mandelbrot Implementations** ✅
- **Files**: `mandelbrot_multi_core_mesh`, `mandelbrot_mesh_simple`, `mandelbrot_mesh`
- **Status**: Built and working (with proper environment)
- **Usage**: Time with `time` command or integrate timing code

### **3. Performance Documentation** ✅
- **Files**: Multiple comprehensive guides
- **Status**: Complete and ready to read

### **4. Kernel with Cycle Counting** ✅
- **File**: `mandelbrot_with_cycles.cpp`
- **Status**: Ready to integrate into programs

## ⚡ Quick Performance Test

**1. Test with Python Script (Easiest):**
```bash
export TT_METAL_HOME=/home/tt-metal-apv
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
python3 measure_mandelbrot_performance.py --size 512 --runs 3 --include-cpu
```

**2. Test with existing executable:**
```bash
export TT_METAL_HOME=/home/tt-metal-apv
time /home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_multi_core_mesh
```

**3. Add timing to your own code:**
```cpp
auto start = std::chrono::high_resolution_clock::now();
// Your TT-Metal code here
auto end = std::chrono::high_resolution_clock::now();
auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
std::cout << "Time: " << ms << " ms" << std::endl;
```

## 🏆 Expected Performance Ranges

**Mandelbrot Set (1024×1024, 100 iterations):**

| Implementation | Time | Speedup | Notes |
|----------------|------|---------|--------|
| CPU Reference | 500-1000ms | 1x | Single-threaded |
| Single TT Core | 200-300ms | 2-3x | One Tensix core |
| 64 TT Cores | 20-50ms | 10-25x | Full device |
| 8 Devices × 64 Cores | 5-15ms | 30-100x | Full mesh |

## 🔍 Understanding Results

### **Good Performance Indicators:**
- High throughput (>10M pixels/sec for Mandelbrot)
- Good speedup (10-50x vs CPU)
- High core utilization (>90%)
- Low timing variance (<5%)

### **Performance Issues:**
- Memory bottlenecks (high memory access time)
- Load imbalance (some cores idle)
- High dispatch overhead (>10% total time)
- Poor scaling with more cores

## 💡 Optimization Tips

### **Hardware Utilization:**
- Use all 64 cores per device (8×8 grid)
- Configure for multiple devices in mesh
- Balance work across cores evenly

### **Memory Optimization:**
- Sequential access patterns are faster
- Use appropriate tile sizes (32×32 common)
- Minimize host-device synchronization

### **Compute Optimization:**
- Use fixed-point arithmetic when possible
- Minimize divergent branches in kernels
- Optimize inner loops for instruction throughput

## 🛠️ Environment Setup

**Required Environment Variables:**
```bash
export TT_METAL_HOME=/home/tt-metal-apv
export ARCH_NAME=wormhole_b0  # or your architecture

# For profiling (optional)
export TT_METAL_PROFILER=1
export TT_METAL_PROFILER_DISPATCH=1
```

**Build Commands:**
```bash
cd /home/tt-metal-apv
cmake --build build-cmake --target mandelbrot_multi_core_mesh -j$(nproc)
```

## 📈 Hardware Specifications

**Per Tenstorrent Device:**
- **Tensix Cores**: Up to 64 (8×8 grid)
- **Clock Frequency**: ~1 GHz
- **DRAM Bandwidth**: ~100+ GB/s
- **L1 per Core**: ~1 MB, ~1000+ GB/s

**Typical Performance:**
- **Peak FLOPS**: ~100+ TFLOPS per device
- **Memory Bandwidth**: ~100+ GB/s per device
- **Latency**: <1ms for small kernels

## 🎯 Summary: What Works Right Now

### **✅ Ready to Use:**
1. **Python performance script** - Works immediately
2. **Host-side timing with std::chrono** - Add to any C++ code
3. **Existing Mandelbrot executables** - Built and ready (with environment)
4. **Profiler APIs** - Available with proper build flags
5. **Comprehensive documentation** - Multiple detailed guides

### **🔧 Requires Integration:**
1. **Kernel cycle counting** - Add to your compute kernels
2. **Advanced profiling** - Requires PROFILER build
3. **Custom performance libraries** - Integration into your projects

### **📊 Performance Analysis:**
- **Basic**: Use Python script or time command
- **Intermediate**: Add std::chrono timing to your code
- **Advanced**: Use profiler APIs and kernel cycle counting

The key is to start simple with host-side timing and the Python script, then gradually add more detailed profiling as needed. The tools are comprehensive and ready to use for measuring performance on Tenstorrent hardware!
