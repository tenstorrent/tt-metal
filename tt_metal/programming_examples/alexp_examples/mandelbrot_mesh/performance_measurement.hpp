// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tt_metal_profiler.hpp"
#include "tt-metalium/distributed.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

/**
 * Performance measurement utilities for Mandelbrot implementations
 */
class PerformanceMeasurement {
public:
    struct TimingResult {
        std::string name;
        double duration_ms;
        double duration_us;
        size_t iterations;
        double avg_duration_ms;
        double avg_duration_us;

        TimingResult(const std::string& n, double ms, size_t iters = 1)
            : name(n), duration_ms(ms), duration_us(ms * 1000.0), iterations(iters),
              avg_duration_ms(ms / iters), avg_duration_us(ms * 1000.0 / iters) {}
    };

    struct BenchmarkResult {
        std::string implementation_name;
        std::vector<TimingResult> timings;
        size_t total_pixels;
        size_t total_iterations;
        size_t num_devices;
        size_t cores_per_device;
        double total_compute_time_ms;
        double pixels_per_second;
        double iterations_per_second;

        BenchmarkResult(const std::string& name) : implementation_name(name) {}

        void calculate_metrics() {
            total_compute_time_ms = 0.0;
            for (const auto& timing : timings) {
                if (timing.name.find("compute") != std::string::npos ||
                    timing.name.find("kernel") != std::string::npos) {
                    total_compute_time_ms += timing.duration_ms;
                }
            }

            if (total_compute_time_ms > 0.0) {
                pixels_per_second = (total_pixels * 1000.0) / total_compute_time_ms;
                iterations_per_second = (total_iterations * 1000.0) / total_compute_time_ms;
            }
        }
    };

private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::vector<BenchmarkResult> benchmark_results_;
    bool profiling_enabled_;
    std::string output_dir_;

public:
    PerformanceMeasurement(bool enable_profiling = false, const std::string& output_dir = ".")
        : profiling_enabled_(enable_profiling), output_dir_(output_dir) {

        if (profiling_enabled_) {
            // Set up device profiler output directory
            tt::tt_metal::detail::SetDeviceProfilerDir(output_dir_);
            std::cout << "🔍 Device profiling enabled. Results will be saved to: " << output_dir_ << std::endl;
        }
    }

    // Start timing a specific operation
    void start_timer(const std::string& operation_name) {
        start_times_[operation_name] = std::chrono::high_resolution_clock::now();
    }

    // End timing and return duration in milliseconds
    double end_timer(const std::string& operation_name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto it = start_times_.find(operation_name);
        if (it == start_times_.end()) {
            std::cerr << "Warning: Timer '" << operation_name << "' was not started!" << std::endl;
            return 0.0;
        }

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - it->second);
        double duration_ms = duration.count() / 1000.0;
        start_times_.erase(it);
        return duration_ms;
    }

    // Initialize device profiling for a single device
    void init_device_profiling(IDevice* device) {
        if (profiling_enabled_) {
            tt::tt_metal::detail::InitDeviceProfiler(device);
            tt::tt_metal::detail::ClearProfilerControlBuffer(device);
        }
    }

    // Initialize device profiling for mesh device
    void init_mesh_profiling(MeshDevice* mesh_device) {
        if (profiling_enabled_) {
            for (uint32_t device_id = 0; device_id < mesh_device->num_devices(); ++device_id) {
                auto device = mesh_device->get_device(device_id);
                init_device_profiling(device);
            }
            tt::tt_metal::detail::ProfilerSync(ProfilerSyncState::INIT);
        }
    }

    // Read profiling results for a single device
    void read_device_profiling_results(IDevice* device, const std::string& tag = "") {
        if (profiling_enabled_) {
            // Create empty runtime map for ProfilerOptionalMetadata
            std::map<std::pair<chip_id_t, uint32_t>, std::string> runtime_map;
            if (!tag.empty()) {
                // Add tag to runtime map if provided
                runtime_map[{0, 0}] = tag;
            }
            ProfilerOptionalMetadata metadata(std::move(runtime_map));
            tt::tt_metal::detail::ReadDeviceProfilerResults(device, ProfilerReadState::NORMAL, metadata);
        }
    }

    // Read profiling results for mesh device
    void read_mesh_profiling_results(MeshDevice& mesh_device, const std::string& tag = "") {
        if (profiling_enabled_) {
            // Create empty runtime map for ProfilerOptionalMetadata
            std::map<std::pair<chip_id_t, uint32_t>, std::string> runtime_map;
            if (!tag.empty()) {
                // Add tag to runtime map if provided
                runtime_map[{0, 0}] = tag;
            }
            ProfilerOptionalMetadata metadata(std::move(runtime_map));
            ReadMeshDeviceProfilerResults(mesh_device, ProfilerReadState::NORMAL, metadata);
        }
    }

    // Finalize device profiling
    void finalize_profiling() {
        if (profiling_enabled_) {
            tt::tt_metal::detail::ProfilerSync(ProfilerSyncState::CLOSE_DEVICE);
            tt::tt_metal::detail::FreshProfilerDeviceLog();
        }
    }

    // Create a new benchmark result
    BenchmarkResult& create_benchmark(const std::string& implementation_name) {
        benchmark_results_.emplace_back(implementation_name);
        return benchmark_results_.back();
    }

    // Add timing result to the current benchmark
    void add_timing(BenchmarkResult& benchmark, const std::string& operation_name, double duration_ms, size_t iterations = 1) {
        benchmark.timings.emplace_back(operation_name, duration_ms, iterations);
    }

    // Measure and time a function execution
    template<typename Func>
    double measure_execution_time(const std::string& operation_name, Func&& func) {
        start_timer(operation_name);
        func();
        return end_timer(operation_name);
    }

    // Print timing results for a benchmark
    void print_benchmark_results(const BenchmarkResult& benchmark) {
        std::cout << "\n🚀 Performance Results for: " << benchmark.implementation_name << std::endl;
        std::cout << "================================================================" << std::endl;

        std::cout << std::left << std::setw(25) << "Operation"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Time (μs)"
                  << std::setw(12) << "Iterations"
                  << std::setw(15) << "Avg (ms)"
                  << std::setw(15) << "Avg (μs)" << std::endl;
        std::cout << std::string(97, '-') << std::endl;

        for (const auto& timing : benchmark.timings) {
            std::cout << std::left << std::setw(25) << timing.name
                      << std::setw(15) << std::fixed << std::setprecision(3) << timing.duration_ms
                      << std::setw(15) << std::fixed << std::setprecision(1) << timing.duration_us
                      << std::setw(12) << timing.iterations
                      << std::setw(15) << std::fixed << std::setprecision(3) << timing.avg_duration_ms
                      << std::setw(15) << std::fixed << std::setprecision(1) << timing.avg_duration_us << std::endl;
        }

        std::cout << std::string(97, '=') << std::endl;

        if (benchmark.total_pixels > 0) {
            std::cout << "📊 Computation Metrics:" << std::endl;
            std::cout << "  • Total pixels: " << benchmark.total_pixels << std::endl;
            std::cout << "  • Total iterations: " << benchmark.total_iterations << std::endl;
            std::cout << "  • Devices used: " << benchmark.num_devices << std::endl;
            std::cout << "  • Cores per device: " << benchmark.cores_per_device << std::endl;
            std::cout << "  • Total cores: " << (benchmark.num_devices * benchmark.cores_per_device) << std::endl;

            if (benchmark.pixels_per_second > 0.0) {
                std::cout << "  • Pixels/second: " << std::scientific << std::setprecision(2) << benchmark.pixels_per_second << std::endl;
                std::cout << "  • Iterations/second: " << std::scientific << std::setprecision(2) << benchmark.iterations_per_second << std::endl;
            }
        }

        if (profiling_enabled_) {
            std::cout << "\n📈 Device profiling results saved to: " << output_dir_ << std::endl;
            std::cout << "  Look for CSV files with detailed kernel-level performance data." << std::endl;
        }
        std::cout << std::endl;
    }

    // Save benchmark results to CSV file
    void save_benchmark_to_csv(const BenchmarkResult& benchmark, const std::string& filename = "") {
        std::string csv_filename = filename;
        if (csv_filename.empty()) {
            csv_filename = output_dir_ + "/" + benchmark.implementation_name + "_benchmark.csv";
        }

        std::ofstream csv_file(csv_filename);
        if (!csv_file.is_open()) {
            std::cerr << "Error: Could not open CSV file: " << csv_filename << std::endl;
            return;
        }

        // Write header
        csv_file << "Implementation,Operation,Duration_ms,Duration_us,Iterations,Avg_Duration_ms,Avg_Duration_us\n";

        // Write timing data
        for (const auto& timing : benchmark.timings) {
            csv_file << benchmark.implementation_name << ","
                     << timing.name << ","
                     << timing.duration_ms << ","
                     << timing.duration_us << ","
                     << timing.iterations << ","
                     << timing.avg_duration_ms << ","
                     << timing.avg_duration_us << "\n";
        }

        // Write summary metrics
        csv_file << "\n# Summary Metrics\n";
        csv_file << "Total_Pixels," << benchmark.total_pixels << "\n";
        csv_file << "Total_Iterations," << benchmark.total_iterations << "\n";
        csv_file << "Num_Devices," << benchmark.num_devices << "\n";
        csv_file << "Cores_Per_Device," << benchmark.cores_per_device << "\n";
        csv_file << "Total_Compute_Time_ms," << benchmark.total_compute_time_ms << "\n";
        csv_file << "Pixels_Per_Second," << benchmark.pixels_per_second << "\n";
        csv_file << "Iterations_Per_Second," << benchmark.iterations_per_second << "\n";

        csv_file.close();
        std::cout << "📊 Benchmark results saved to: " << csv_filename << std::endl;
    }

    // Compare multiple benchmark results
    void compare_benchmarks(const std::vector<BenchmarkResult>& benchmarks) {
        if (benchmarks.empty()) return;

        std::cout << "\n🏆 Benchmark Comparison" << std::endl;
        std::cout << "================================================================" << std::endl;

        std::cout << std::left << std::setw(25) << "Implementation"
                  << std::setw(15) << "Total Time (ms)"
                  << std::setw(18) << "Pixels/sec"
                  << std::setw(18) << "Iterations/sec"
                  << std::setw(12) << "Cores" << std::endl;
        std::cout << std::string(88, '-') << std::endl;

        for (const auto& benchmark : benchmarks) {
            double total_time = 0.0;
            for (const auto& timing : benchmark.timings) {
                total_time += timing.duration_ms;
            }

            std::cout << std::left << std::setw(25) << benchmark.implementation_name
                      << std::setw(15) << std::fixed << std::setprecision(3) << total_time
                      << std::setw(18) << std::scientific << std::setprecision(2) << benchmark.pixels_per_second
                      << std::setw(18) << std::scientific << std::setprecision(2) << benchmark.iterations_per_second
                      << std::setw(12) << (benchmark.num_devices * benchmark.cores_per_device) << std::endl;
        }
        std::cout << std::endl;
    }

    // Get all benchmark results
    const std::vector<BenchmarkResult>& get_benchmark_results() const {
        return benchmark_results_;
    }
};

// Utility class for automatic timing (RAII)
class ScopedTimer {
private:
    PerformanceMeasurement& perf_;
    std::string operation_name_;

public:
    ScopedTimer(PerformanceMeasurement& perf, const std::string& operation_name)
        : perf_(perf), operation_name_(operation_name) {
        perf_.start_timer(operation_name_);
    }

    ~ScopedTimer() {
        perf_.end_timer(operation_name_);
    }

    // Get elapsed time without stopping the timer
    double elapsed_ms() const {
        // This is a placeholder implementation
        // In a real implementation, you might want to add a method to PerformanceMeasurement
        return 0.0; // Placeholder
    }
};

// Macro for easy scoped timing
#define SCOPED_TIMER(perf, name) ScopedTimer _timer(perf, name)
