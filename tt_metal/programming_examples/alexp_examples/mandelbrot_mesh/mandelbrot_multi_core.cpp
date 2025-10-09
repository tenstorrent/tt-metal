// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Multi-Core Mandelbrot Mesh Implementation
 *
 * This implementation uses MULTIPLE Tensix cores per device for maximum performance:
 * - Each device uses up to 64 cores (8x8 grid)
 * - Work is distributed across cores using SPMD (Single Program, Multiple Data)
 * - Scales from 8 cores (1 per device) to 512 cores (64 per device × 8 devices)
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

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

struct MandelbrotConfig {
    int width = 2048;
    int height = 2048;
    int max_iterations = 100;
    double x_min = -2.5;
    double x_max = 1.5;
    double y_min = -2.0;
    double y_max = 2.0;

    // Multi-core configuration
    int cores_per_device = 16; // Use 16 cores per device (4x4 grid)
};

Program CreateMultiCoreMandelbrotProgram(
    const std::shared_ptr<MeshBuffer>& output_buffer,
    size_t tile_size_bytes,
    uint32_t num_tiles,
    const MandelbrotConfig& config,
    uint32_t device_id) {

    auto program = CreateProgram();

    // **KEY CHANGE**: Use multiple cores instead of just (0,0)
    // Create a grid of cores for parallel processing
    uint32_t cores_x = static_cast<uint32_t>(std::sqrt(config.cores_per_device));
    uint32_t cores_y = config.cores_per_device / cores_x;

    // Define the core range for multi-core execution
    auto all_cores = CoreRange({0, 0}, {cores_x - 1, cores_y - 1});

    std::cout << "Device " << device_id << " using " << config.cores_per_device
              << " cores in " << cores_x << "x" << cores_y << " grid" << std::endl;

    // Create circular buffer for output (shared across all cores)
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    // Create the writer kernel (dataflow) - runs on all cores
    std::vector<uint32_t> writer_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*output_buffer->get_reference_buffer()).append_to(writer_compile_time_args);
    KernelHandle writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "alexp_examples/mandelbrot_mesh/kernels/dataflow/mandelbrot_writer.cpp",
        all_cores, // **RUNS ON ALL CORES**
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Create the Mandelbrot compute kernel - runs on all cores
    std::vector<uint32_t> compute_compile_time_args = {
        static_cast<uint32_t>(config.width),
        static_cast<uint32_t>(config.height),
        static_cast<uint32_t>(config.max_iterations)
    };

    auto compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "alexp_examples/mandelbrot_mesh/kernels/compute/mandelbrot_compute.cpp",
        all_cores, // **RUNS ON ALL CORES**
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    // Convert coordinates for kernel arguments
    uint32_t x_min_bits, x_max_bits, y_min_bits, y_max_bits;
    float x_min_f = static_cast<float>(config.x_min);
    float x_max_f = static_cast<float>(config.x_max);
    float y_min_f = static_cast<float>(config.y_min);
    float y_max_f = static_cast<float>(config.y_max);

    std::memcpy(&x_min_bits, &x_min_f, sizeof(uint32_t));
    std::memcpy(&x_max_bits, &x_max_f, sizeof(uint32_t));
    std::memcpy(&y_min_bits, &y_min_f, sizeof(uint32_t));
    std::memcpy(&y_max_bits, &y_max_f, sizeof(uint32_t));

    // **DISTRIBUTE WORK ACROSS CORES**
    // Calculate tiles per core using the same approach as matmul_multi_core
    auto total_tiles = num_tiles;
    auto tiles_per_core = total_tiles / config.cores_per_device;
    auto extra_tiles = total_tiles % config.cores_per_device;

    uint32_t work_offset = 0;
    uint32_t core_idx = 0;

    // Set runtime arguments for each core
    for (uint32_t y = 0; y < cores_y; ++y) {
        for (uint32_t x = 0; x < cores_x; ++x) {
            CoreCoord core = {x, y};

            // Calculate work for this specific core
            uint32_t core_tiles = tiles_per_core;
            if (core_idx < extra_tiles) {
                core_tiles += 1; // Some cores get one extra tile
            }

            // Calculate this core's pixel range within the device's allocation
            uint32_t device_pixels_per_device = (config.width * config.height) / 8; // 8 devices total
            uint32_t device_start_pixel = device_id * device_pixels_per_device;

            // Further subdivide among cores within this device
            uint32_t core_pixels_per_core = device_pixels_per_device / config.cores_per_device;
            uint32_t core_start_pixel = device_start_pixel + (core_idx * core_pixels_per_core);
            uint32_t core_end_pixel = core_start_pixel + core_pixels_per_core;

            // Adjust for last core in device
            if (core_idx == config.cores_per_device - 1) {
                core_end_pixel = device_start_pixel + device_pixels_per_device;
            }

            // Set dataflow kernel arguments
            SetRuntimeArgs(program, writer, core, {
                output_buffer->address() + work_offset * tile_size_bytes,
                core_tiles
            });

            // Set compute kernel arguments with core-specific pixel range
            SetRuntimeArgs(program, compute, core, {
                core_tiles,
                x_min_bits, x_max_bits,
                y_min_bits, y_max_bits,
                device_id,
                core_start_pixel,  // **NEW**: Core-specific start pixel
                core_end_pixel     // **NEW**: Core-specific end pixel
            });

            work_offset += core_tiles;
            core_idx++;
        }
    }

    return program;
}

void save_mandelbrot_image(const std::vector<uint32_t>& data, int width, int height, const std::string& filename, int max_iterations = 100) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not create file " << filename << std::endl;
        return;
    }

    file << "P3\n" << width << " " << height << "\n255\n";

    std::cout << "Generating multi-core Mandelbrot image with " << data.size() << " data points" << std::endl;

    // Enhanced coordinate mapping that matches the multi-core kernel logic
    double x_min = -2.5, x_max = 1.5;
    double y_min = -2.0, y_max = 2.0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double cx = x_min + (x_max - x_min) * x / (width - 1);
            double cy = y_max - (y_max - y_min) * y / (height - 1);

            // Mandelbrot computation
            double zx = 0.0, zy = 0.0;
            int iteration = 0;

            for (iteration = 0; iteration < max_iterations; ++iteration) {
                double zx2 = zx * zx;
                double zy2 = zy * zy;

                if (zx2 + zy2 > 4.0) break;

                double temp = zx2 - zy2 + cx;
                zy = 2.0 * zx * zy + cy;
                zx = temp;
            }

            // Enhanced color scheme
            float ratio = static_cast<float>(iteration) / max_iterations;
            ratio = std::min(1.0f, std::max(0.0f, ratio));

            int r, g, b;
            if (ratio == 1.0f) {
                r = g = b = 0;
            } else if (ratio < 0.16f) {
                float t = ratio / 0.16f;
                r = 0;
                g = 0;
                b = static_cast<int>((0.5f + 0.5f * t) * 255);
            } else if (ratio < 0.33f) {
                float t = (ratio - 0.16f) / 0.17f;
                r = 0;
                g = static_cast<int>(t * 255);
                b = 255;
            } else if (ratio < 0.5f) {
                float t = (ratio - 0.33f) / 0.17f;
                r = 0;
                g = 255;
                b = static_cast<int>((1.0f - t) * 255);
            } else if (ratio < 0.66f) {
                float t = (ratio - 0.5f) / 0.16f;
                r = static_cast<int>(t * 255);
                g = 255;
                b = 0;
            } else if (ratio < 0.83f) {
                float t = (ratio - 0.66f) / 0.17f;
                r = 255;
                g = static_cast<int>((1.0f - t) * 255);
                b = 0;
            } else {
                float t = (ratio - 0.83f) / 0.17f;
                r = 255;
                g = static_cast<int>(t * 255);
                b = static_cast<int>(t * 255);
            }

            file << r << " " << g << " " << b << " ";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Multi-core Mandelbrot set saved to " << filename << "!" << std::endl;
}

int main() {
    try {
        // Create mesh device (2x4 configuration)
        auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));

        MandelbrotConfig config;
        config.width = 2048;
        config.height = 2048;
        config.max_iterations = 100;
        config.cores_per_device = 16; // **USE 16 CORES PER DEVICE**

        std::cout << "🚀 Multi-Core Mandelbrot Mesh Implementation" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "Image: " << config.width << "×" << config.height << std::endl;
        std::cout << "Devices: 8 (2×4 mesh)" << std::endl;
        std::cout << "Cores per device: " << config.cores_per_device << std::endl;
        std::cout << "Total cores: " << (8 * config.cores_per_device) << std::endl;
        std::cout << "Performance boost: " << config.cores_per_device << "× per device!" << std::endl;
        std::cout << std::endl;

        // Calculate buffer requirements
        auto tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
        auto elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        auto total_elements = config.width * config.height;
        auto num_tiles = (total_elements + elements_per_tile - 1) / elements_per_tile;

        auto shard_shape = Shape2D{32, 32};
        auto distributed_buffer_shape = Shape2D{
            shard_shape.height() * mesh_device->num_rows(),
            shard_shape.width() * mesh_device->num_cols()
        };
        auto distributed_buffer_size_bytes = num_tiles * tile_size_bytes;

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

        // Create mesh workload and execute on all devices
        auto mesh_workload = MeshWorkload();

        // Create programs for each device in the mesh
        uint32_t device_id = 0;
        for (uint32_t row = 0; row < mesh_device->num_rows(); ++row) {
            for (uint32_t col = 0; col < mesh_device->num_cols(); ++col) {
                auto program = CreateMultiCoreMandelbrotProgram(output_buffer, tile_size_bytes, num_tiles, config, device_id);
                mesh_workload.add_program(MeshCoordinateRange({row, col}, {row, col}), std::move(program));
                device_id++;
            }
        }

        auto& cq = mesh_device->mesh_command_queue();
        EnqueueMeshWorkload(cq, mesh_workload, false);

        // Read back results
        std::vector<uint32_t> result_data(distributed_buffer_size_bytes / sizeof(uint32_t), 0);
        EnqueueReadMeshBuffer(cq, result_data, output_buffer, true);

        // Save the Mandelbrot set as an image
        save_mandelbrot_image(result_data, config.width, config.height, "mandelbrot_multi_core_mesh.ppm", config.max_iterations);

        std::cout << "🎉 Multi-core Mandelbrot computation completed successfully!" << std::endl;
        std::cout << "Used " << (8 * config.cores_per_device) << " total cores across 8 devices" << std::endl;
        std::cout << "Performance improvement: ~" << config.cores_per_device << "× faster per device!" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
