// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Simple Performance Measurement Example for Tenstorrent Hardware
 *
 * This example demonstrates various ways to measure performance on TT hardware:
 * 1. Host-side timing using std::chrono
 * 2. Device-side profiling using TT-Metal profiler
 * 3. Basic throughput calculations
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/tt_metal_profiler.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

class SimplePerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::string operation_name_;

public:
    SimplePerformanceTimer(const std::string& name) : operation_name_(name) {
        start_time_ = std::chrono::high_resolution_clock::now();
        std::cout << "⏱️  Starting: " << operation_name_ << std::endl;
    }

    ~SimplePerformanceTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);

        double ms = duration.count() / 1000.0;
        double us = duration.count();

        std::cout << "✅ " << operation_name_ << ": "
                  << std::fixed << std::setprecision(3) << ms << " ms"
                  << " (" << std::fixed << std::setprecision(1) << us << " μs)" << std::endl;
    }
};

#define TIME_OPERATION(name) SimplePerformanceTimer _timer(name)

int main() {
    std::cout << "🚀 Tenstorrent Hardware Performance Measurement Example" << std::endl;
    std::cout << "======================================================" << std::endl;

    try {
        // 1. HOST-SIDE TIMING MEASUREMENTS
        std::cout << "\n📊 Host-Side Timing Measurements:" << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        {
            TIME_OPERATION("Mesh Device Initialization");
            MeshShape mesh_shape({1, 8});
            auto mesh_device_config = MeshDeviceConfig(mesh_shape);
            auto mesh_device = MeshDevice::create(mesh_device_config);

            std::cout << "   • Initialized " << mesh_device->num_devices() << " devices" << std::endl;
        }

        // 2. DEVICE-SIDE PROFILING (requires PROFILER build)
        std::cout << "\n📈 Device-Side Profiling:" << std::endl;
        std::cout << "-------------------------" << std::endl;

        // Check if profiler is available
        const char* profiler_env = std::getenv("TT_METAL_PROFILER");
        bool profiler_available = (profiler_env != nullptr && std::string(profiler_env) == "1");

        if (profiler_available) {
            std::cout << "✅ TT-Metal Profiler is available" << std::endl;
            std::cout << "   • Set TT_METAL_PROFILER=1 to enable device profiling" << std::endl;
            std::cout << "   • Set TT_METAL_PROFILER_DISPATCH=1 for dispatch profiling" << std::endl;

            // Set profiler output directory
            tt::tt_metal::detail::SetDeviceProfilerDir("./profiler_results");
            std::cout << "   • Profiler results will be saved to: ./profiler_results" << std::endl;

        } else {
            std::cout << "⚠️  TT-Metal Profiler not available" << std::endl;
            std::cout << "   • To enable profiling:" << std::endl;
            std::cout << "     export TT_METAL_PROFILER=1" << std::endl;
            std::cout << "     export TT_METAL_PROFILER_DISPATCH=1" << std::endl;
            std::cout << "   • Requires PROFILER build configuration" << std::endl;
        }

        // 3. THROUGHPUT CALCULATIONS
        std::cout << "\n⚡ Throughput Calculations:" << std::endl;
        std::cout << "---------------------------" << std::endl;

        // Example calculation for Mandelbrot set
        int width = 1024, height = 1024, max_iterations = 100;
        uint64_t total_pixels = static_cast<uint64_t>(width) * height;
        uint64_t total_operations = total_pixels * max_iterations;

        // Simulate execution time (in real code, this would be actual measurement)
        double execution_time_ms = 50.0;  // Example: 50ms

        double pixels_per_second = (total_pixels * 1000.0) / execution_time_ms;
        double operations_per_second = (total_operations * 1000.0) / execution_time_ms;
        double gflops = operations_per_second / 1e9;  // Assuming 1 operation ≈ 1 FLOP

        std::cout << "   • Image size: " << width << "x" << height << " = " << total_pixels << " pixels" << std::endl;
        std::cout << "   • Total operations: " << total_operations << std::endl;
        std::cout << "   • Execution time: " << execution_time_ms << " ms" << std::endl;
        std::cout << "   • Throughput: " << std::scientific << std::setprecision(2)
                  << pixels_per_second << " pixels/second" << std::endl;
        std::cout << "   • Operations/sec: " << std::scientific << std::setprecision(2)
                  << operations_per_second << " ops/second" << std::endl;
        std::cout << "   • Estimated GFLOPS: " << std::fixed << std::setprecision(2)
                  << gflops << " GFLOPS" << std::endl;

        // 4. KERNEL-LEVEL CYCLE COUNTING
        std::cout << "\n🔄 Kernel-Level Cycle Counting:" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        std::cout << "   • Use get_cycle_count() in compute kernels" << std::endl;
        std::cout << "   • Example kernel code:" << std::endl;
        std::cout << "     uint64_t start_cycles = get_cycle_count();" << std::endl;
        std::cout << "     // ... your computation ..." << std::endl;
        std::cout << "     uint64_t end_cycles = get_cycle_count();" << std::endl;
        std::cout << "     uint64_t elapsed = end_cycles - start_cycles;" << std::endl;

        // 5. MEMORY BANDWIDTH MEASUREMENTS
        std::cout << "\n💾 Memory Bandwidth Considerations:" << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        size_t buffer_size_bytes = total_pixels * sizeof(uint32_t);
        double buffer_size_mb = buffer_size_bytes / (1024.0 * 1024.0);
        double bandwidth_gbps = (buffer_size_mb / 1024.0) / (execution_time_ms / 1000.0);

        std::cout << "   • Buffer size: " << buffer_size_mb << " MB" << std::endl;
        std::cout << "   • Estimated bandwidth: " << std::fixed << std::setprecision(2)
                  << bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "   • DRAM bandwidth (theoretical): ~100+ GB/s per device" << std::endl;
        std::cout << "   • L1 bandwidth (theoretical): ~1000+ GB/s per core" << std::endl;

        // 6. PROFILER API USAGE EXAMPLE
        std::cout << "\n🔍 Profiler API Usage:" << std::endl;
        std::cout << "----------------------" << std::endl;

        if (profiler_available) {
            std::cout << "   • Profiler APIs available for detailed analysis:" << std::endl;
            std::cout << "     - tt::tt_metal::detail::InitDeviceProfiler(device)" << std::endl;
            std::cout << "     - tt::tt_metal::detail::ReadDeviceProfilerResults(device)" << std::endl;
            std::cout << "     - tt::tt_metal::detail::ProfilerSync(state)" << std::endl;
            std::cout << "   • Results saved as CSV files with detailed metrics" << std::endl;
        } else {
            std::cout << "   • Build with PROFILER=1 to access profiler APIs" << std::endl;
        }

        // 7. PERFORMANCE OPTIMIZATION TIPS
        std::cout << "\n🚀 Performance Optimization Tips:" << std::endl;
        std::cout << "----------------------------------" << std::endl;
        std::cout << "   • Use all available cores (up to 64 per device)" << std::endl;
        std::cout << "   • Optimize memory access patterns (sequential > random)" << std::endl;
        std::cout << "   • Use fixed-point arithmetic when possible" << std::endl;
        std::cout << "   • Minimize host-device synchronization" << std::endl;
        std::cout << "   • Use circular buffers efficiently" << std::endl;
        std::cout << "   • Profile regularly to identify bottlenecks" << std::endl;

        std::cout << "\n🎉 Performance measurement example completed!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
