// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Multi-Core Mandelbrot Mesh Implementation with Performance Measurement
 *
 * This implementation includes comprehensive performance measurement capabilities:
 * - Host-side timing measurements
 * - Device-side profiling using TT-Metal APIs
 * - Cycle counting and throughput analysis
 * - Comparison between different configurations
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

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

// Mandelbrot set parameters with performance configuration
struct MandelbrotConfig {
    int width = 1024;
    int height = 1024;
    int max_iterations = 100;
    double x_min = -2.5;
    double x_max = 1.5;
    double y_min = -2.0;
    double y_max = 2.0;
    uint32_t cores_per_device = 64;  // 8x8 grid of cores
    bool enable_profiling = true;
    std::string output_dir = "./benchmark_results";
};

// Function to save Mandelbrot set as PPM image
void save_mandelbrot_image(const std::vector<uint32_t>& data, int width, int height,
                          const std::string& filename, int max_iterations) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    file << "P3\n" << width << " " << height << "\n255\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            uint32_t iterations = data[index];

            // Color mapping: black for points in set, colored for others
            if (iterations >= max_iterations) {
                file << "0 0 0 ";  // Black for points in the set
            } else {
                // Create a color gradient based on iteration count
                int r = (iterations * 255) / max_iterations;
                int g = (iterations * 128) / max_iterations;
                int b = (iterations * 64) / max_iterations;
                file << r << " " << g << " " << b << " ";
            }
        }
        file << "\n";
    }
    file.close();
    std::cout << "💾 Mandelbrot image saved as: " << filename << std::endl;
}

Program CreateMultiCoreMandelbrotProgram(
    const std::shared_ptr<MeshBuffer>& output_buffer,
    size_t tile_size_bytes,
    uint32_t num_tiles,
    const MandelbrotConfig& config,
    uint32_t device_id) {

    auto program = CreateProgram();

    // Calculate core grid based on cores_per_device, avoiding dispatch cores
    uint32_t cores_x = static_cast<uint32_t>(std::sqrt(config.cores_per_device));
    uint32_t cores_y = config.cores_per_device / cores_x;

    // Start from (1, 1) to avoid potential dispatch cores at (0, 0)
    auto core_range = CoreRange(CoreCoord{1, 1}, CoreCoord{cores_x, cores_y});

    // Create circular buffer for output
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(
        num_output_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::UInt32}})
        .set_page_size(output_cb_index, tile_size_bytes);
    CreateCircularBuffer(program, core_range, cb_output_config);

    // Create compute kernel with runtime args for each core
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
    uint32_t tiles_per_core = num_tiles / config.cores_per_device;
    uint32_t core_id = 0;

    for (uint32_t y = 0; y < cores_y; ++y) {
        for (uint32_t x = 0; x < cores_x; ++x) {
            CoreCoord core = {x, y};

            // Calculate which tiles this core should process
            uint32_t start_tile = core_id * tiles_per_core;
            uint32_t end_tile = (core_id == config.cores_per_device - 1) ?
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

int main(int argc, char** argv) {
    std::cout << "🚀 Multi-Core Mandelbrot Mesh Benchmark" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Parse command line arguments
    MandelbrotConfig config;
    bool enable_profiling = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--profile") {
            enable_profiling = true;
            config.enable_profiling = true;
        } else if (arg == "--width" && i + 1 < argc) {
            config.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.height = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--cores" && i + 1 < argc) {
            config.cores_per_device = std::stoi(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --profile          Enable device profiling\n";
            std::cout << "  --width <n>        Image width (default: 1024)\n";
            std::cout << "  --height <n>       Image height (default: 1024)\n";
            std::cout << "  --iterations <n>   Max iterations (default: 100)\n";
            std::cout << "  --cores <n>        Cores per device (default: 64)\n";
            std::cout << "  --output-dir <dir> Output directory (default: ./benchmark_results)\n";
            std::cout << "  --help             Show this help\n";
            return 0;
        }
    }

    // Initialize performance measurement
    PerformanceMeasurement perf(enable_profiling, config.output_dir);
    auto& benchmark = perf.create_benchmark("Multi-Core-Mesh");

    // Set benchmark metadata
    benchmark.total_pixels = config.width * config.height;
    benchmark.total_iterations = benchmark.total_pixels * config.max_iterations;
    benchmark.cores_per_device = config.cores_per_device;

    std::cout << "📋 Configuration:" << std::endl;
    std::cout << "  • Image size: " << config.width << "x" << config.height << " pixels" << std::endl;
    std::cout << "  • Max iterations: " << config.max_iterations << std::endl;
    std::cout << "  • Cores per device: " << config.cores_per_device << std::endl;
    std::cout << "  • Profiling: " << (enable_profiling ? "enabled" : "disabled") << std::endl;
    std::cout << "  • Output directory: " << config.output_dir << std::endl;
    std::cout << std::endl;

    try {
        // Initialize mesh device
        MeshShape mesh_shape({1, 8});
        auto mesh_device_config = MeshDeviceConfig(mesh_shape);

        double init_time = perf.measure_execution_time("mesh_device_init", [&]() {
            auto mesh_device = MeshDevice::create(mesh_device_config);
            benchmark.num_devices = mesh_device->num_devices();

            // Initialize profiling if enabled
            if (enable_profiling) {
                perf.init_mesh_profiling(mesh_device.get());
            }

            std::cout << "🔧 Initialized mesh device with " << mesh_device->num_devices() << " devices" << std::endl;
        });
        perf.add_timing(benchmark, "mesh_device_init", init_time);

        // Create mesh device (we need to recreate it since it went out of scope)
        auto mesh_device = MeshDevice::create(mesh_device_config);
        if (enable_profiling) {
            perf.init_mesh_profiling(mesh_device.get());
        }

        // Calculate buffer requirements
        const size_t tile_size_bytes = 4096;  // 32x32 pixels * 4 bytes per pixel
        const uint32_t tiles_per_row = (config.width + 31) / 32;
        const uint32_t tiles_per_col = (config.height + 31) / 32;
        const uint32_t num_tiles = tiles_per_row * tiles_per_col;
        const size_t distributed_buffer_size_bytes = num_tiles * tile_size_bytes;

        std::cout << "📊 Buffer configuration:" << std::endl;
        std::cout << "  • Tiles: " << tiles_per_row << "x" << tiles_per_col << " = " << num_tiles << std::endl;
        std::cout << "  • Tile size: " << tile_size_bytes << " bytes" << std::endl;
        std::cout << "  • Total buffer size: " << distributed_buffer_size_bytes << " bytes" << std::endl;

        // Create buffer configurations
        double buffer_setup_time = perf.measure_execution_time("buffer_setup", [&]() {
            Shape2D distributed_buffer_shape = {
                mesh_device->num_rows(),
                mesh_device->num_cols()
            };
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

            // Create distributed output buffer
            auto output_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
        });
        perf.add_timing(benchmark, "buffer_setup", buffer_setup_time);

        // Recreate buffer (same issue as mesh_device)
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

        // Create and execute mesh workload
        double workload_creation_time = perf.measure_execution_time("workload_creation", [&]() {
            auto mesh_workload = CreateMeshWorkload();

            // Create programs for each device in the mesh
            uint32_t device_id = 0;
            for (uint32_t row = 0; row < mesh_device->num_rows(); ++row) {
                for (uint32_t col = 0; col < mesh_device->num_cols(); ++col) {
                    auto program = CreateMultiCoreMandelbrotProgram(output_buffer, tile_size_bytes, num_tiles, config, device_id);
                    AddProgramToMeshWorkload(mesh_workload, std::move(program), MeshCoordinateRange({row, col}, {row, col}));
                    device_id++;
                }
            }
        });
        perf.add_timing(benchmark, "workload_creation", workload_creation_time);

        // Execute the workload
        auto mesh_workload = CreateMeshWorkload();
        uint32_t device_id = 0;
        for (uint32_t row = 0; row < mesh_device->num_rows(); ++row) {
            for (uint32_t col = 0; col < mesh_device->num_cols(); ++col) {
                auto program = CreateMultiCoreMandelbrotProgram(output_buffer, tile_size_bytes, num_tiles, config, device_id);
                AddProgramToMeshWorkload(mesh_workload, std::move(program), MeshCoordinateRange({row, col}, {row, col}));
                device_id++;
            }
        }

        auto& cq = mesh_device->mesh_command_queue();

        double execution_time = perf.measure_execution_time("kernel_execution", [&]() {
            EnqueueMeshWorkload(cq, mesh_workload, false);
            // Wait for completion
            cq.finish();
        });
        perf.add_timing(benchmark, "kernel_execution", execution_time);

        // Read back results
        std::vector<uint32_t> result_data(distributed_buffer_size_bytes / sizeof(uint32_t), 0);
        double readback_time = perf.measure_execution_time("result_readback", [&]() {
            EnqueueReadMeshBuffer(cq, result_data, output_buffer, true);
        });
        perf.add_timing(benchmark, "result_readback", readback_time);

        // Save results
        double save_time = perf.measure_execution_time("image_save", [&]() {
            std::string output_filename = config.output_dir + "/mandelbrot_multi_core_mesh_benchmark.ppm";
            save_mandelbrot_image(result_data, config.width, config.height, output_filename, config.max_iterations);
        });
        perf.add_timing(benchmark, "image_save", save_time);

        // Read device profiling results if enabled
        if (enable_profiling) {
            double profiling_time = perf.measure_execution_time("profiling_readback", [&]() {
                perf.read_mesh_profiling_results(*mesh_device, "mandelbrot_multi_core");
                perf.finalize_profiling();
            });
            perf.add_timing(benchmark, "profiling_readback", profiling_time);
        }

        // Calculate performance metrics
        benchmark.calculate_metrics();

        // Print results
        perf.print_benchmark_results(benchmark);

        // Save benchmark results to CSV
        perf.save_benchmark_to_csv(benchmark);

        std::cout << "🎉 Multi-core Mandelbrot computation completed successfully!" << std::endl;
        std::cout << "Used " << (mesh_device->num_devices() * config.cores_per_device) << " total cores across " << mesh_device->num_devices() << " devices" << std::endl;

        if (enable_profiling) {
            std::cout << "\n📈 Device profiling data has been saved to CSV files in: " << config.output_dir << std::endl;
            std::cout << "These files contain detailed kernel-level performance metrics including:" << std::endl;
            std::cout << "  • Kernel execution times" << std::endl;
            std::cout << "  • Memory bandwidth utilization" << std::endl;
            std::cout << "  • Core utilization statistics" << std::endl;
            std::cout << "  • Dispatch overhead analysis" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error during execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
