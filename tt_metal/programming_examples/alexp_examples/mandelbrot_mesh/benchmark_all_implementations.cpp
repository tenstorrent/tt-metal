// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Comprehensive Mandelbrot Implementation Benchmark Tool
 *
 * This tool benchmarks all available Mandelbrot implementations:
 * - Single-core mesh implementation
 * - Multi-core mesh implementation
 * - Different core counts and configurations
 * - CPU reference implementation for comparison
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <chrono>
#include <complex>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/mesh_config.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

#include "performance_measurement.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Benchmark configuration
struct BenchmarkConfig {
    int width = 1024;
    int height = 1024;
    int max_iterations = 100;
    double x_min = -2.5;
    double x_max = 1.5;
    double y_min = -2.0;
    double y_max = 2.0;
    std::vector<uint32_t> core_counts = {1, 4, 16, 64};  // Different core configurations to test
    bool enable_profiling = false;
    std::string output_dir = "./benchmark_results";
    bool run_cpu_reference = true;
    int num_warmup_runs = 1;
    int num_benchmark_runs = 3;
};

// CPU reference implementation for comparison
std::vector<uint32_t> cpu_mandelbrot_reference(const BenchmarkConfig& config) {
    std::vector<uint32_t> result(config.width * config.height);

    double dx = (config.x_max - config.x_min) / config.width;
    double dy = (config.y_max - config.y_min) / config.height;

    for (int y = 0; y < config.height; ++y) {
        for (int x = 0; x < config.width; ++x) {
            double cx = config.x_min + x * dx;
            double cy = config.y_min + y * dy;

            std::complex<double> c(cx, cy);
            std::complex<double> z(0, 0);

            uint32_t iterations = 0;
            while (iterations < config.max_iterations && std::abs(z) <= 2.0) {
                z = z * z + c;
                iterations++;
            }

            result[y * config.width + x] = iterations;
        }
    }

    return result;
}

// Function to create single-core program
Program CreateSingleCoreMandelbrotProgram(
    const std::shared_ptr<MeshBuffer>& output_buffer,
    size_t tile_size_bytes,
    uint32_t num_tiles,
    const BenchmarkConfig& config,
    uint32_t device_id) {

    auto program = CreateProgram();
    auto core_range = CoreRange(CoreCoord{0, 0});  // Single core

    // Create circular buffer for output
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(
        num_output_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::UInt32}})
        .set_page_size(output_cb_index, tile_size_bytes);
    CreateCircularBuffer(program, core_range, cb_output_config);

    // Create compute kernel
    std::string compute_kernel_file = std::string(OVERRIDE_KERNEL_PREFIX) + "alexp_examples/mandelbrot_mesh/kernels/compute/mandelbrot_simple.cpp";
    auto compute_kernel = CreateKernel(
        program,
        compute_kernel_file,
        core_range,
        ComputeConfig{.compile_args = {}}
    );

    // Create writer kernel
    std::string writer_kernel_file = std::string(OVERRIDE_KERNEL_PREFIX) + "alexp_examples/mandelbrot_mesh/kernels/dataflow/mandelbrot_writer.cpp";
    auto writer_kernel = CreateKernel(
        program,
        writer_kernel_file,
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
    );

    // Set runtime arguments
    CoreCoord core = {0, 0};

    // Compute kernel args
    std::vector<uint32_t> compute_args = {
        static_cast<uint32_t>(config.width),
        static_cast<uint32_t>(config.height),
        static_cast<uint32_t>(config.max_iterations),
        0,  // start_tile
        num_tiles,  // end_tile
        device_id,
        0   // core_id
    };
    SetRuntimeArgs(program, compute_kernel, core, compute_args);

    // Writer kernel args
    auto buffer_address = output_buffer->get_device_buffer(MeshCoordinate{0, device_id})->address();
    std::vector<uint32_t> writer_args = {
        static_cast<uint32_t>(buffer_address),
        0,  // offset in buffer
        num_tiles * tile_size_bytes  // size to write
    };
    SetRuntimeArgs(program, writer_kernel, core, writer_args);

    return program;
}

// Function to create multi-core program with configurable core count
Program CreateMultiCoreMandelbrotProgram(
    const std::shared_ptr<MeshBuffer>& output_buffer,
    size_t tile_size_bytes,
    uint32_t num_tiles,
    const BenchmarkConfig& config,
    uint32_t device_id,
    uint32_t cores_per_device) {

    auto program = CreateProgram();

    // Calculate core grid based on cores_per_device
    uint32_t cores_x = 1, cores_y = 1;
    if (cores_per_device >= 64) {
        cores_x = cores_y = 8;  // 8x8 = 64 cores
    } else if (cores_per_device >= 16) {
        cores_x = cores_y = 4;  // 4x4 = 16 cores
    } else if (cores_per_device >= 4) {
        cores_x = cores_y = 2;  // 2x2 = 4 cores
    } else {
        cores_x = cores_y = 1;  // 1x1 = 1 core
    }

    uint32_t actual_cores = cores_x * cores_y;
    auto core_range = CoreRange(CoreCoord{0, 0}, CoreCoord{cores_x - 1, cores_y - 1});

    // Create circular buffer for output
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(
        num_output_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::UInt32}})
        .set_page_size(output_cb_index, tile_size_bytes);
    CreateCircularBuffer(program, core_range, cb_output_config);

    // Create compute kernel
    std::string compute_kernel_file = std::string(OVERRIDE_KERNEL_PREFIX) + "alexp_examples/mandelbrot_mesh/kernels/compute/mandelbrot_multi_core.cpp";
    auto compute_kernel = CreateKernel(
        program,
        compute_kernel_file,
        core_range,
        ComputeConfig{.compile_args = {}}
    );

    // Create writer kernel
    std::string writer_kernel_file = std::string(OVERRIDE_KERNEL_PREFIX) + "alexp_examples/mandelbrot_mesh/kernels/dataflow/mandelbrot_writer.cpp";
    auto writer_kernel = CreateKernel(
        program,
        writer_kernel_file,
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
    );

    // Set runtime arguments for each core
    uint32_t tiles_per_core = num_tiles / actual_cores;
    uint32_t core_id = 0;

    for (uint32_t y = 0; y < cores_y; ++y) {
        for (uint32_t x = 0; x < cores_x; ++x) {
            CoreCoord core = {x, y};

            // Calculate which tiles this core should process
            uint32_t start_tile = core_id * tiles_per_core;
            uint32_t end_tile = (core_id == actual_cores - 1) ?
                                num_tiles : (core_id + 1) * tiles_per_core;

            // Compute kernel args
            std::vector<uint32_t> compute_args = {
                static_cast<uint32_t>(config.width),
                static_cast<uint32_t>(config.height),
                static_cast<uint32_t>(config.max_iterations),
                start_tile,
                end_tile,
                device_id,
                core_id
            };
            SetRuntimeArgs(program, compute_kernel, core, compute_args);

            // Writer kernel args
            auto buffer_address = output_buffer->get_device_buffer(MeshCoordinate{0, device_id})->address();
            std::vector<uint32_t> writer_args = {
                static_cast<uint32_t>(buffer_address),
                start_tile * tile_size_bytes,  // offset in buffer
                (end_tile - start_tile) * tile_size_bytes  // size to write
            };
            SetRuntimeArgs(program, writer_kernel, core, writer_args);

            core_id++;
        }
    }

    return program;
}

// Function to run CPU benchmark
PerformanceMeasurement::BenchmarkResult run_cpu_benchmark(const BenchmarkConfig& config, PerformanceMeasurement& perf) {
    std::cout << "\n🖥️  Running CPU Reference Implementation..." << std::endl;

    auto& benchmark = perf.create_benchmark("CPU-Reference");
    benchmark.total_pixels = config.width * config.height;
    benchmark.total_iterations = benchmark.total_pixels * config.max_iterations;
    benchmark.num_devices = 1;
    benchmark.cores_per_device = 1;  // Single-threaded CPU implementation

    // Warmup runs
    for (int i = 0; i < config.num_warmup_runs; ++i) {
        cpu_mandelbrot_reference(config);
    }

    // Benchmark runs
    std::vector<double> run_times;
    for (int i = 0; i < config.num_benchmark_runs; ++i) {
        double run_time = perf.measure_execution_time("cpu_compute_run_" + std::to_string(i), [&]() {
            auto result = cpu_mandelbrot_reference(config);
        });
        run_times.push_back(run_time);
    }

    // Calculate average time
    double total_time = 0.0;
    for (double time : run_times) {
        total_time += time;
    }
    double avg_time = total_time / config.num_benchmark_runs;

    perf.add_timing(benchmark, "cpu_compute", avg_time, config.num_benchmark_runs);
    benchmark.calculate_metrics();

    return benchmark;
}

// Function to run TT-Metal benchmark with specific core count
PerformanceMeasurement::BenchmarkResult run_ttmetal_benchmark(
    const BenchmarkConfig& config,
    PerformanceMeasurement& perf,
    uint32_t cores_per_device,
    const std::string& implementation_name) {

    std::cout << "\n🚀 Running " << implementation_name << " (cores: " << cores_per_device << ")..." << std::endl;

    auto& benchmark = perf.create_benchmark(implementation_name + "-" + std::to_string(cores_per_device) + "cores");
    benchmark.total_pixels = config.width * config.height;
    benchmark.total_iterations = benchmark.total_pixels * config.max_iterations;
    benchmark.cores_per_device = cores_per_device;

    // Initialize mesh device
    MeshShape mesh_shape({1, 8});
    auto mesh_device_config = MeshDeviceConfig(mesh_shape);
    auto mesh_device = MeshDevice::create(mesh_device_config);
    benchmark.num_devices = mesh_device->num_devices();

    if (config.enable_profiling) {
        perf.init_mesh_profiling(mesh_device.get());
    }

    // Calculate buffer requirements
    const size_t tile_size_bytes = 4096;
    const uint32_t tiles_per_row = (config.width + 31) / 32;
    const uint32_t tiles_per_col = (config.height + 31) / 32;
    const uint32_t num_tiles = tiles_per_row * tiles_per_col;
    const size_t distributed_buffer_size_bytes = num_tiles * tile_size_bytes;

    // Create buffer configurations
    Shape2D distributed_buffer_shape = {mesh_device->num_rows(), mesh_device->num_cols()};
    Shape2D shard_shape = {1, 1};
    auto local_buffer_config = DeviceLocalBufferConfig{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM,
        .bottom_up = false
    };
    auto distributed_buffer_config = distributed::ShardedBufferConfig{
        .global_size = distributed_buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR
    };
    auto output_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    // Warmup runs
    for (int warmup = 0; warmup < config.num_warmup_runs; ++warmup) {
        auto mesh_workload = MeshWorkload();
        uint32_t device_id = 0;
        for (uint32_t row = 0; row < mesh_device->num_rows(); ++row) {
            for (uint32_t col = 0; col < mesh_device->num_cols(); ++col) {
                Program program;
                if (cores_per_device == 1) {
                    program = CreateSingleCoreMandelbrotProgram(output_buffer, tile_size_bytes, num_tiles, config, device_id);
                } else {
                    program = CreateMultiCoreMandelbrotProgram(output_buffer, tile_size_bytes, num_tiles, config, device_id, cores_per_device);
                }
                mesh_workload.add_program(MeshCoordinateRange({row, col}, {row, col}), std::move(program));
                device_id++;
            }
        }

        auto& cq = mesh_device->mesh_command_queue();
        EnqueueMeshWorkload(cq, mesh_workload, true);
    }

    // Benchmark runs
    std::vector<double> execution_times;
    for (int run = 0; run < config.num_benchmark_runs; ++run) {
        auto mesh_workload = MeshWorkload();
        uint32_t device_id = 0;
        for (uint32_t row = 0; row < mesh_device->num_rows(); ++row) {
            for (uint32_t col = 0; col < mesh_device->num_cols(); ++col) {
                Program program;
                if (cores_per_device == 1) {
                    program = CreateSingleCoreMandelbrotProgram(output_buffer, tile_size_bytes, num_tiles, config, device_id);
                } else {
                    program = CreateMultiCoreMandelbrotProgram(output_buffer, tile_size_bytes, num_tiles, config, device_id, cores_per_device);
                }
                mesh_workload.add_program(MeshCoordinateRange({row, col}, {row, col}), std::move(program));
                device_id++;
            }
        }

        auto& cq = mesh_device->mesh_command_queue();

        double execution_time = perf.measure_execution_time("kernel_execution_run_" + std::to_string(run), [&]() {
            EnqueueMeshWorkload(cq, mesh_workload, true);
        });
        execution_times.push_back(execution_time);
    }

    // Calculate average execution time
    double total_exec_time = 0.0;
    for (double time : execution_times) {
        total_exec_time += time;
    }
    double avg_exec_time = total_exec_time / config.num_benchmark_runs;

    perf.add_timing(benchmark, "kernel_execution", avg_exec_time, config.num_benchmark_runs);

    // Read back results (only once)
    std::vector<uint32_t> result_data(distributed_buffer_size_bytes / sizeof(uint32_t), 0);
    auto& cq = mesh_device->mesh_command_queue();
    double readback_time = perf.measure_execution_time("result_readback", [&]() {
        EnqueueReadMeshBuffer(cq, result_data, output_buffer, true);
    });
    perf.add_timing(benchmark, "result_readback", readback_time);

    if (config.enable_profiling) {
        perf.read_mesh_profiling_results(*mesh_device, implementation_name);
    }

    benchmark.calculate_metrics();
    return benchmark;
}

int main(int argc, char** argv) {
    std::cout << "🏁 Comprehensive Mandelbrot Implementation Benchmark" << std::endl;
    std::cout << "====================================================" << std::endl;

    BenchmarkConfig config;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--profile") {
            config.enable_profiling = true;
        } else if (arg == "--width" && i + 1 < argc) {
            config.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.height = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--no-cpu") {
            config.run_cpu_reference = false;
        } else if (arg == "--warmup-runs" && i + 1 < argc) {
            config.num_warmup_runs = std::stoi(argv[++i]);
        } else if (arg == "--benchmark-runs" && i + 1 < argc) {
            config.num_benchmark_runs = std::stoi(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --profile              Enable device profiling\n";
            std::cout << "  --width <n>            Image width (default: 1024)\n";
            std::cout << "  --height <n>           Image height (default: 1024)\n";
            std::cout << "  --iterations <n>       Max iterations (default: 100)\n";
            std::cout << "  --no-cpu               Skip CPU reference benchmark\n";
            std::cout << "  --warmup-runs <n>      Number of warmup runs (default: 1)\n";
            std::cout << "  --benchmark-runs <n>   Number of benchmark runs (default: 3)\n";
            std::cout << "  --output-dir <dir>     Output directory (default: ./benchmark_results)\n";
            std::cout << "  --help                 Show this help\n";
            return 0;
        }
    }

    std::cout << "📋 Benchmark Configuration:" << std::endl;
    std::cout << "  • Image size: " << config.width << "x" << config.height << " pixels" << std::endl;
    std::cout << "  • Max iterations: " << config.max_iterations << std::endl;
    std::cout << "  • Core configurations: ";
    for (size_t i = 0; i < config.core_counts.size(); ++i) {
        std::cout << config.core_counts[i];
        if (i < config.core_counts.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "  • Profiling: " << (config.enable_profiling ? "enabled" : "disabled") << std::endl;
    std::cout << "  • CPU reference: " << (config.run_cpu_reference ? "enabled" : "disabled") << std::endl;
    std::cout << "  • Warmup runs: " << config.num_warmup_runs << std::endl;
    std::cout << "  • Benchmark runs: " << config.num_benchmark_runs << std::endl;
    std::cout << "  • Output directory: " << config.output_dir << std::endl;
    std::cout << std::endl;

    PerformanceMeasurement perf(config.enable_profiling, config.output_dir);
    std::vector<PerformanceMeasurement::BenchmarkResult> all_results;

    try {
        // Run CPU reference benchmark
        if (config.run_cpu_reference) {
            auto cpu_result = run_cpu_benchmark(config, perf);
            perf.print_benchmark_results(cpu_result);
            perf.save_benchmark_to_csv(cpu_result);
            all_results.push_back(cpu_result);
        }

        // Run TT-Metal benchmarks with different core counts
        for (uint32_t cores : config.core_counts) {
            std::string impl_name = (cores == 1) ? "Single-Core-Mesh" : "Multi-Core-Mesh";
            auto tt_result = run_ttmetal_benchmark(config, perf, cores, impl_name);
            perf.print_benchmark_results(tt_result);
            perf.save_benchmark_to_csv(tt_result);
            all_results.push_back(tt_result);
        }

        // Compare all results
        perf.compare_benchmarks(all_results);

        // Save comprehensive comparison
        std::string comparison_file = config.output_dir + "/comprehensive_comparison.csv";
        std::ofstream comparison_csv(comparison_file);
        if (comparison_csv.is_open()) {
            comparison_csv << "Implementation,Cores,Total_Time_ms,Pixels_Per_Second,Iterations_Per_Second,Speedup_vs_CPU\n";

            double cpu_baseline = 0.0;
            for (const auto& result : all_results) {
                if (result.implementation_name.find("CPU") != std::string::npos) {
                    for (const auto& timing : result.timings) {
                        cpu_baseline = timing.duration_ms;
                        break;
                    }
                    break;
                }
            }

            for (const auto& result : all_results) {
                double total_time = 0.0;
                for (const auto& timing : result.timings) {
                    total_time += timing.duration_ms;
                }

                double speedup = (cpu_baseline > 0.0) ? cpu_baseline / total_time : 1.0;

                comparison_csv << result.implementation_name << ","
                              << (result.num_devices * result.cores_per_device) << ","
                              << total_time << ","
                              << result.pixels_per_second << ","
                              << result.iterations_per_second << ","
                              << speedup << "\n";
            }
            comparison_csv.close();
            std::cout << "📊 Comprehensive comparison saved to: " << comparison_file << std::endl;
        }

        if (config.enable_profiling) {
            perf.finalize_profiling();
        }

        std::cout << "\n🎉 Comprehensive benchmark completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error during benchmark: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
