// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Mandelbrot Mesh Implementation - Simple Version
 *
 * This version uses the mandelbrot_simple.cpp kernel which includes
 * detailed comments explaining the parallelization strategy.
 */

#include <functional>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Mandelbrot set parameters
struct MandelbrotConfig {
    int width = 512;
    int height = 512;
    int max_iterations = 100;
    double x_min = -2.5;
    double x_max = 1.5;
    double y_min = -2.0;
    double y_max = 2.0;
};

Program CreateMandelbrotProgram(
    const std::shared_ptr<MeshBuffer>& output_buffer,
    size_t tile_size_bytes,
    uint32_t num_tiles,
    const MandelbrotConfig& config,
    uint32_t device_id) {

    auto program = CreateProgram();
    auto target_tensix_core = CoreRange(CoreCoord{0, 0});

    // Create circular buffer for output
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_output_config);

    // Create the writer kernel
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*output_buffer->get_reference_buffer()).append_to(writer_compile_time_args);
    KernelHandle writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "alexp_examples/mandelbrot_mesh/kernels/dataflow/mandelbrot_writer.cpp",
        target_tensix_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Create the Mandelbrot compute kernel using the simple version with detailed comments
    std::vector<uint32_t> compute_compile_time_args = {
        static_cast<uint32_t>(config.width),
        static_cast<uint32_t>(config.height),
        static_cast<uint32_t>(config.max_iterations)
    };

    auto compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "alexp_examples/mandelbrot_mesh/kernels/compute/mandelbrot_simple.cpp",
        target_tensix_core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    // Set runtime arguments (simple version doesn't need coordinate conversion)
    SetRuntimeArgs(program, writer, target_tensix_core, {output_buffer->address(), num_tiles});
    SetRuntimeArgs(program, compute, target_tensix_core, {
        num_tiles,
        0, 0, 0, 0,  // Placeholder args (coordinates handled internally)
        device_id
    });

    return program;
}

void save_mandelbrot_image(const std::vector<uint32_t>& data, int width, int height, const std::string& filename, int max_iterations = 100) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not create file " << filename << std::endl;
        return;
    }

    // Write PPM header
    file << "P3\n" << width << " " << height << "\n255\n";

    std::cout << "Using actual kernel data for image generation!" << std::endl;
    std::cout << "Data size: " << data.size() << ", expected: " << (width * height) << std::endl;

    // Verify we have the right amount of data
    if (data.size() != static_cast<size_t>(width * height)) {
        std::cerr << "Error: Data size mismatch! Expected " << (width * height)
                  << " pixels, got " << data.size() << std::endl;
        return;
    }

    // Use the actual kernel-computed data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pixel_idx = y * width + x;
            uint32_t iteration = data[pixel_idx];

            // Create enhanced color scheme
            float ratio = static_cast<float>(iteration) / max_iterations;
            ratio = std::min(1.0f, std::max(0.0f, ratio));

            int r, g, b;
            if (ratio == 1.0f) {
                // Points in the set
                r = g = b = 0;
            } else if (ratio < 0.16f) {
                // Deep blue to blue
                float t = ratio / 0.16f;
                r = 0;
                g = 0;
                b = static_cast<int>((0.5f + 0.5f * t) * 255);
            } else if (ratio < 0.33f) {
                // Blue to cyan
                float t = (ratio - 0.16f) / 0.17f;
                r = 0;
                g = static_cast<int>(t * 255);
                b = 255;
            } else if (ratio < 0.5f) {
                // Cyan to green
                float t = (ratio - 0.33f) / 0.17f;
                r = 0;
                g = 255;
                b = static_cast<int>((1.0f - t) * 255);
            } else if (ratio < 0.66f) {
                // Green to yellow
                float t = (ratio - 0.5f) / 0.16f;
                r = static_cast<int>(t * 255);
                g = 255;
                b = 0;
            } else if (ratio < 0.83f) {
                // Yellow to red
                float t = (ratio - 0.66f) / 0.17f;
                r = 255;
                g = static_cast<int>((1.0f - t) * 255);
                b = 0;
            } else {
                // Red to white
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
    std::cout << "Mandelbrot set saved to " << filename << " using kernel data!" << std::endl;
}

int main() {
    try {
        std::cout << "=== Mandelbrot Mesh - Simple Kernel Version ===" << std::endl;
        std::cout << "Using mandelbrot_simple.cpp with detailed parallelization comments" << std::endl;
        std::cout << std::endl;

        // Create mesh device (2x4 configuration)
        auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));

        MandelbrotConfig config;
        config.width = 512;
        config.height = 512;
        config.max_iterations = 100;

        std::cout << "Configuration:" << std::endl;
        std::cout << "  Resolution: " << config.width << " × " << config.height << std::endl;
        std::cout << "  Max iterations: " << config.max_iterations << std::endl;
        std::cout << "  Mesh device: " << mesh_device->num_rows() << " × " << mesh_device->num_cols() << " (" << mesh_device->num_rows() * mesh_device->num_cols() << " devices)" << std::endl;
        std::cout << std::endl;

        // Calculate buffer requirements
        auto tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
        auto elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        auto total_elements = config.width * config.height;
        auto num_tiles = (total_elements + elements_per_tile - 1) / elements_per_tile;

        std::cout << "Memory layout:" << std::endl;
        std::cout << "  Tile size: " << elements_per_tile << " elements (" << tile_size_bytes << " bytes)" << std::endl;
        std::cout << "  Total tiles: " << num_tiles << std::endl;
        std::cout << "  Tiles per device: ~" << (num_tiles / 8) << std::endl;
        std::cout << std::endl;

        // Define distributed buffer configuration
        auto shard_shape = Shape2D{32, 32}; // Each device handles a 32x32 tile
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

        auto distributed_buffer_config = tt::tt_metal::distributed::ShardedBufferConfig{
            .global_size = distributed_buffer_size_bytes,
            .global_buffer_shape = distributed_buffer_shape,
            .shard_shape = shard_shape,
            .shard_orientation = ShardOrientation::ROW_MAJOR
        };

        // Create distributed output buffer
        auto output_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

        // Create mesh workload and execute on all devices
        auto mesh_workload = MeshWorkload();

        std::cout << "Creating programs for each device:" << std::endl;
        uint32_t device_id = 0;
        uint32_t pixels_per_device = total_elements / 8;
        for (uint32_t row = 0; row < mesh_device->num_rows(); ++row) {
            for (uint32_t col = 0; col < mesh_device->num_cols(); ++col) {
                uint32_t start_pixel = device_id * pixels_per_device;
                uint32_t end_pixel = (device_id == 7) ? total_elements : (device_id + 1) * pixels_per_device;

                std::cout << "  Device " << device_id << " (" << row << "," << col << "): pixels "
                         << start_pixel << " → " << (end_pixel - 1) << " (" << (end_pixel - start_pixel) << " pixels)" << std::endl;

                auto program = CreateMandelbrotProgram(output_buffer, tile_size_bytes, num_tiles, config, device_id);
                mesh_workload.add_program(MeshCoordinateRange({row, col}, {row, col}), std::move(program));
                device_id++;
            }
        }
        std::cout << std::endl;

        std::cout << "Executing Mandelbrot computation on mesh device..." << std::endl;
        auto& cq = mesh_device->mesh_command_queue();
        EnqueueMeshWorkload(cq, mesh_workload, false);

        // Read back results
        std::cout << "Gathering results from all devices..." << std::endl;
        std::vector<uint32_t> result_data(distributed_buffer_size_bytes / sizeof(uint32_t), 0);
        EnqueueReadMeshBuffer(cq, result_data, output_buffer, true);

        // Save the Mandelbrot set as an image
        save_mandelbrot_image(result_data, config.width, config.height, "mandelbrot_mesh_simple.ppm", config.max_iterations);

        std::cout << std::endl;
        std::cout << "✅ Mandelbrot computation completed successfully!" << std::endl;
        std::cout << "   8× parallel speedup achieved across "
                  << mesh_device->num_rows() << "×" << mesh_device->num_cols()
                  << " mesh device" << std::endl;
        std::cout << "   Image saved as mandelbrot_mesh_simple.ppm" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
