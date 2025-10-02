# Tenstorrent Hardware Performance Measurement Guide

This comprehensive guide explains how to measure performance, cycles, and timing on Tenstorrent hardware using TT-Metal APIs.

## Overview of Performance Measurement Methods

Tenstorrent hardware supports multiple levels of performance measurement:

### 1. **Host-Side Timing (Microseconds/Milliseconds)**
- High-resolution timing from host perspective
- Measures end-to-end execution time
- Good for overall throughput analysis

### 2. **Device-Side Profiling (Detailed Analysis)**
- Kernel-level execution timing
- Memory bandwidth utilization
- Core utilization statistics
- Dispatch overhead analysis

### 3. **Kernel Cycle Counting (Hardware Cycles)**
- Fine-grained cycle-level measurements
- Direct hardware cycle counters
- Best for optimization and bottleneck identification

### 4. **Throughput Analysis**
- Operations per second
- Memory bandwidth utilization
- GFLOPS calculations

## Method 1: Host-Side Timing Measurements

### Basic Timing with std::chrono

```cpp
#include <chrono>

class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;

public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double get_elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }

    double get_elapsed_us() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        return static_cast<double>(duration.count());  // Microseconds
    }
};

// Usage example
PerformanceTimer timer;
timer.start();

// Your TT-Metal operation here
EnqueueProgram(cq, program, false);
Finish(cq);

double execution_time_ms = timer.get_elapsed_ms();
std::cout << "Execution time: " << execution_time_ms << " ms" << std::endl;
```

### RAII Timer for Automatic Measurement

```cpp
class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name_(name) {
        start_time_ = std::chrono::high_resolution_clock::now();
        std::cout << "Starting: " << name_ << std::endl;
    }

    ~ScopedTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        double ms = duration.count() / 1000.0;
        std::cout << name_ << ": " << ms << " ms" << std::endl;
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Usage with automatic timing
{
    ScopedTimer timer("Kernel Execution");
    EnqueueProgram(cq, program, false);
    Finish(cq);
} // Timer automatically prints result when going out of scope
```

## Method 2: Device-Side Profiling with TT-Metal APIs

### Setting Up the Profiler

First, you need to build with profiler support and set environment variables:

```bash
# Build with profiler support
export TT_METAL_PROFILER=1
export TT_METAL_PROFILER_DISPATCH=1

# Build your application
cmake --build build-cmake --target your_app
```

### Using Profiler APIs

```cpp
#include "tt-metalium/tt_metal_profiler.hpp"

// Initialize profiler for a device
void setup_profiling(IDevice* device) {
    tt::tt_metal::detail::InitDeviceProfiler(device);
    tt::tt_metal::detail::ClearProfilerControlBuffer(device);

    // Set output directory for profiler results
    tt::tt_metal::detail::SetDeviceProfilerDir("./profiler_results");
}

// Initialize profiler for mesh device
void setup_mesh_profiling(MeshDevice* mesh_device) {
    for (uint32_t device_id = 0; device_id < mesh_device->num_devices(); ++device_id) {
        auto device = mesh_device->get_device(device_id);
        setup_profiling(device);
    }
    tt::tt_metal::detail::ProfilerSync(ProfilerSyncState::INIT);
}

// Read profiler results
void read_profiling_results(IDevice* device) {
    // Create runtime map for metadata
    std::map<std::pair<chip_id_t, uint32_t>, std::string> runtime_map;
    runtime_map[{0, 0}] = "my_operation";
    ProfilerOptionalMetadata metadata(std::move(runtime_map));

    tt::tt_metal::detail::ReadDeviceProfilerResults(
        device, ProfilerReadState::NORMAL, metadata);
}

// For mesh devices
void read_mesh_profiling_results(MeshDevice& mesh_device) {
    std::map<std::pair<chip_id_t, uint32_t>, std::string> runtime_map;
    runtime_map[{0, 0}] = "mesh_operation";
    ProfilerOptionalMetadata metadata(std::move(runtime_map));

    ReadMeshDeviceProfilerResults(mesh_device, ProfilerReadState::NORMAL, metadata);
}

// Finalize profiling
void finalize_profiling() {
    tt::tt_metal::detail::ProfilerSync(ProfilerSyncState::CLOSE_DEVICE);
    tt::tt_metal::detail::FreshProfilerDeviceLog();
}
```

### Complete Profiling Example

```cpp
int main() {
    // Setup
    auto device = CreateDevice(0);
    setup_profiling(device);

    // Your computation
    auto program = CreateProgram();
    // ... add kernels to program ...

    auto cq = device->command_queue();

    // Execute with profiling
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Read results
    read_profiling_results(device);
    finalize_profiling();

    CloseDevice(device);
    return 0;
}
```

### Profiler Output Files

The profiler generates several CSV files:

- **`ops_perf_results_device_X.csv`**: Overall operation performance
- **`kernel_durations_device_X.csv`**: Individual kernel execution times
- **`memory_profiler_device_X.csv`**: Memory bandwidth utilization
- **`dispatch_core_runtime_device_X.csv`**: Dispatch core performance

## Method 3: Kernel Cycle Counting

### Using get_cycle_count() in Kernels

In your compute kernel (.cpp file):

```cpp
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    // Start cycle counting
    uint64_t start_cycles = get_cycle_count();

    // Your computation here
    for (uint32_t i = 0; i < num_iterations; ++i) {
        // Mandelbrot computation or other work
        // ...
    }

    // End cycle counting
    uint64_t end_cycles = get_cycle_count();
    uint64_t elapsed_cycles = end_cycles - start_cycles;

    // Store result in a buffer for readback (optional)
    // You would need to set up a dedicated buffer for performance data
}
}
```

### Detailed Cycle Measurement Example

```cpp
namespace NAMESPACE {
void MAIN {
    // Get runtime arguments
    uint32_t num_pixels = get_arg_val<uint32_t>(0);
    uint32_t max_iterations = get_arg_val<uint32_t>(1);

    // Performance counters
    uint64_t total_cycles = 0;
    uint64_t compute_cycles = 0;
    uint64_t memory_cycles = 0;
    uint32_t pixels_processed = 0;

    uint64_t start_total = get_cycle_count();

    for (uint32_t pixel_id = 0; pixel_id < num_pixels; ++pixel_id) {
        // Memory access timing
        uint64_t mem_start = get_cycle_count();
        // ... memory operations ...
        memory_cycles += get_cycle_count() - mem_start;

        // Compute timing
        uint64_t compute_start = get_cycle_count();

        // Mandelbrot computation
        uint32_t iterations = 0;
        // ... computation loop ...

        compute_cycles += get_cycle_count() - compute_start;
        pixels_processed++;
    }

    total_cycles = get_cycle_count() - start_total;

    // Calculate metrics
    uint32_t avg_cycles_per_pixel = pixels_processed > 0 ?
                                   (uint32_t)(total_cycles / pixels_processed) : 0;
    uint32_t compute_percentage = total_cycles > 0 ?
                                 (uint32_t)((compute_cycles * 100) / total_cycles) : 0;

    // These metrics could be stored in a performance buffer for host readback
}
}
```

## Method 4: Throughput and Performance Analysis

### Calculating Key Performance Metrics

```cpp
class PerformanceAnalyzer {
public:
    struct Metrics {
        double execution_time_ms;
        uint64_t total_operations;
        uint64_t total_pixels;
        double operations_per_second;
        double pixels_per_second;
        double memory_bandwidth_gbps;
        double gflops;
        uint32_t num_cores_used;
        uint32_t num_devices_used;
    };

    static Metrics calculate_metrics(
        double execution_time_ms,
        uint32_t width, uint32_t height, uint32_t max_iterations,
        size_t buffer_size_bytes,
        uint32_t num_cores, uint32_t num_devices) {

        Metrics metrics;
        metrics.execution_time_ms = execution_time_ms;
        metrics.total_pixels = static_cast<uint64_t>(width) * height;
        metrics.total_operations = metrics.total_pixels * max_iterations;
        metrics.num_cores_used = num_cores;
        metrics.num_devices_used = num_devices;

        // Calculate throughput (per second)
        double execution_time_s = execution_time_ms / 1000.0;
        metrics.operations_per_second = metrics.total_operations / execution_time_s;
        metrics.pixels_per_second = metrics.total_pixels / execution_time_s;

        // Estimate GFLOPS (assuming ~10 FLOPs per Mandelbrot iteration)
        double total_flops = metrics.total_operations * 10.0;
        metrics.gflops = (total_flops / 1e9) / execution_time_s;

        // Calculate memory bandwidth
        double buffer_size_gb = buffer_size_bytes / (1024.0 * 1024.0 * 1024.0);
        metrics.memory_bandwidth_gbps = buffer_size_gb / execution_time_s;

        return metrics;
    }

    static void print_metrics(const Metrics& m) {
        std::cout << "\n📊 Performance Metrics:" << std::endl;
        std::cout << "======================" << std::endl;
        std::cout << "Execution time: " << std::fixed << std::setprecision(3)
                  << m.execution_time_ms << " ms" << std::endl;
        std::cout << "Total pixels: " << m.total_pixels << std::endl;
        std::cout << "Total operations: " << m.total_operations << std::endl;
        std::cout << "Cores used: " << m.num_cores_used << std::endl;
        std::cout << "Devices used: " << m.num_devices_used << std::endl;
        std::cout << "Pixels/second: " << std::scientific << std::setprecision(2)
                  << m.pixels_per_second << std::endl;
        std::cout << "Operations/second: " << std::scientific << std::setprecision(2)
                  << m.operations_per_second << std::endl;
        std::cout << "Estimated GFLOPS: " << std::fixed << std::setprecision(2)
                  << m.gflops << std::endl;
        std::cout << "Memory bandwidth: " << std::fixed << std::setprecision(2)
                  << m.memory_bandwidth_gbps << " GB/s" << std::endl;
    }
};
```

## Method 5: Complete Performance Measurement Example

Here's a complete example that combines all measurement methods:

```cpp
#include <chrono>
#include <iostream>
#include <iomanip>
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/tt_metal_profiler.hpp"

class ComprehensivePerformanceMeasurement {
private:
    bool profiling_enabled_;
    std::chrono::high_resolution_clock::time_point start_time_;

public:
    ComprehensivePerformanceMeasurement(bool enable_profiling = false)
        : profiling_enabled_(enable_profiling) {}

    void start_measurement(IDevice* device) {
        start_time_ = std::chrono::high_resolution_clock::now();

        if (profiling_enabled_) {
            tt::tt_metal::detail::InitDeviceProfiler(device);
            tt::tt_metal::detail::ClearProfilerControlBuffer(device);
            tt::tt_metal::detail::SetDeviceProfilerDir("./performance_results");
        }
    }

    double end_measurement(IDevice* device) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        double execution_time_ms = duration.count() / 1000.0;

        if (profiling_enabled_) {
            std::map<std::pair<chip_id_t, uint32_t>, std::string> runtime_map;
            runtime_map[{0, 0}] = "performance_test";
            ProfilerOptionalMetadata metadata(std::move(runtime_map));

            tt::tt_metal::detail::ReadDeviceProfilerResults(
                device, ProfilerReadState::NORMAL, metadata);
            tt::tt_metal::detail::ProfilerSync(ProfilerSyncState::CLOSE_DEVICE);
        }

        return execution_time_ms;
    }
};

// Usage example
int main() {
    // Initialize
    auto device = CreateDevice(0);
    ComprehensivePerformanceMeasurement perf_measure(true); // Enable profiling

    // Create your program
    auto program = CreateProgram();
    // ... set up kernels ...

    auto cq = device->command_queue();

    // Measure performance
    perf_measure.start_measurement(device);

    EnqueueProgram(cq, program, false);
    Finish(cq);

    double execution_time = perf_measure.end_measurement(device);

    // Calculate and display metrics
    auto metrics = PerformanceAnalyzer::calculate_metrics(
        execution_time, 1024, 1024, 100,  // width, height, iterations
        1024*1024*4,  // buffer size
        64, 1         // cores, devices
    );

    PerformanceAnalyzer::print_metrics(metrics);

    CloseDevice(device);
    return 0;
}
```

## Hardware-Specific Performance Characteristics

### Tenstorrent Device Specifications

**Per Device:**
- **Tensix Cores**: Up to 64 compute cores (8x8 grid)
- **DRAM Bandwidth**: ~100+ GB/s per device
- **L1 Memory**: ~1 MB per core, ~1000+ GB/s bandwidth
- **Clock Frequency**: ~1 GHz (varies by model)

**Per Core:**
- **SFPU**: Special function processing unit
- **Math Units**: Multiple MAC units
- **Local Memory**: L1 cache with high bandwidth

### Performance Optimization Tips

1. **Maximize Core Utilization**
   ```cpp
   // Use all 64 cores (8x8 grid)
   auto core_range = CoreRange(CoreCoord{0,0}, CoreCoord{7,7});
   ```

2. **Optimize Memory Access Patterns**
   ```cpp
   // Sequential access is faster than random access
   // Use appropriate tile sizes (32x32 is common)
   ```

3. **Use Fixed-Point Arithmetic**
   ```cpp
   // Fixed-point is faster than floating-point on TT hardware
   int32_t fixed_point_value = (int32_t)(float_value * 65536);
   ```

4. **Minimize Host-Device Synchronization**
   ```cpp
   // Batch operations to reduce overhead
   EnqueueProgram(cq, program, false);  // Non-blocking
   // ... queue more work ...
   Finish(cq);  // Single synchronization point
   ```

## Interpreting Performance Results

### Good Performance Indicators
- **High Core Utilization**: >90% of cores actively computing
- **High Memory Bandwidth**: Close to theoretical limits
- **Low Dispatch Overhead**: <5% of total execution time
- **Consistent Timing**: Low variance across runs

### Common Performance Issues
- **Memory Bottlenecks**: High memory access time
- **Load Imbalance**: Some cores idle while others work
- **Dispatch Overhead**: Too many small kernel launches
- **Suboptimal Memory Patterns**: Random access patterns

### Typical Performance Ranges

**Mandelbrot Set (1024x1024, 100 iterations):**
- **Single Core**: ~500ms
- **64 Cores (1 device)**: ~50ms
- **64 Cores (8 devices)**: ~10ms
- **Expected Speedup**: 8-15x per device

## Environment Variables for Profiling

```bash
# Enable basic profiling
export TT_METAL_PROFILER=1

# Enable dispatch profiling
export TT_METAL_PROFILER_DISPATCH=1

# Set profiler output directory
export TT_METAL_PROFILER_OUTPUT_DIR=./profiler_results

# Enable Tracy profiler integration (if available)
export TT_METAL_TRACY_ENABLE=1
```

## Conclusion

Tenstorrent hardware provides multiple levels of performance measurement:

1. **Start with host-side timing** for overall performance understanding
2. **Use device profiling** for detailed kernel analysis
3. **Add cycle counting** for fine-grained optimization
4. **Calculate throughput metrics** for comparative analysis

The key to good performance on TT hardware is:
- Utilizing all available cores
- Optimizing memory access patterns
- Using appropriate data types (fixed-point when possible)
- Minimizing host-device synchronization

Regular profiling and measurement are essential for achieving optimal performance on Tenstorrent hardware.
