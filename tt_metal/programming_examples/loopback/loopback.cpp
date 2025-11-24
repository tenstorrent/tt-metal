// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

// This example demonstrates a simple data copy from DRAM into L1(SRAM) and to another place in DRAM.
// The general flow is as follows:
// 1. Initialize the device
// 2. Create the data movement kernel (fancy word of specialized subroutines) on core {0, 0}
//    that will perform the copy
// 3. Create the buffer (both on DRAM And L1) and fill DRAM with data. Point the kernel to the buffers.
// 4. Execute the kernel
// 5. Read the data back from the buffer
// 6. Validate the data
// 7. Clean up the device. Exit

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    bool pass = true;

    try {
        // Create a 1x1 mesh on device 0 (same API scales to multi-device meshes)
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // Submit work via the mesh command queue: uploads/downloads and program execution.
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        // Data on Tensix is stored in tiles. A tile is a 2D array of (usually) 32x32 values. And the Tensix uses
        // BFloat16 as the most well supported data type. Thus the tile size is 32x32x2 = 2048 bytes.
        constexpr uint32_t num_tiles = 50;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
        constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

        // Configure mesh buffers. Use single-tile page size so transfers operate tile-by-tile.
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,  // Number of bytes when round-robin between banks. Usually this is the same
                                           // as the tile size for efficiency.
            .buffer_type = tt::tt_metal::BufferType::DRAM};  // Type of buffer (DRAM or L1(SRAM))
        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = tile_size_bytes, .buffer_type = tt::tt_metal::BufferType::L1};  // This time we allocate on L1

        distributed::ReplicatedBufferConfig dram_buffer_config{
            .size = dram_buffer_size};  // Size per device (replicated across mesh). Since we are operating on a unit
                                        // mesh this is the total size.
        distributed::ReplicatedBufferConfig l1_buffer_config{.size = tile_size_bytes};

        // Allocate the buffers (replicated across mesh; on unit mesh ⇒ single device allocation)
        auto l1_buffer = distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());
        auto input_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
        auto output_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a given time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        Program program = CreateProgram();

        // A MeshWorkload is a collection of programs that will be executed on the mesh. Each workload is
        // local to a single device. Here we create a workload for our single-device mesh.
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

        // This example program will only use 1 Tensix core. So we set the core to {0, 0} (the most top-left core).
        constexpr CoreCoord core = {0, 0};

        // Create the data movement kernel. This kernel will be used to copy data from DRAM to DRAM (see the
        // `loopback_dram_copy.cpp` file for the actual implementation). The kernel is created on the Tensix core
        // {0, 0} and uses the default NoC.
        std::vector<uint32_t> dram_copy_compile_time_args;
        TensorAccessorArgs(*input_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
        TensorAccessorArgs(*output_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "loopback/kernels/loopback_dram_copy.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = dram_copy_compile_time_args});

        // Initialize the input buffer with random data.
        std::vector<bfloat16> input_vec(elements_per_tile * num_tiles);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
        for (auto& val : input_vec) {
            val = bfloat16(distribution(rng));
        }

        // Upload the data from host to the device. The final argument is set to false. This indicates to Metalium that
        // the upload is non-blocking, an upload will be launched, but the function will return immediately, before the
        // upload is complete. This is useful for performance reasons, as it allows the host to continue while the
        // upload is in progress. Note that the host is responsible for ensuring that the upload is complete before the
        // memory holding the data is freed.
        distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, /*blocking=*/false);

        // Set runtime arguments for the kernel.
        const std::vector<uint32_t> runtime_args = {
            l1_buffer->address(), input_dram_buffer->address(), output_dram_buffer->address(), num_tiles};

        SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

        // Add the program to the workload for the mesh.
        workload.add_program(device_range, std::move(program));
        // Enqueue the workload for execution on the mesh (non-blocking) and wait for completion before reading back.
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);
        // NOTE: The above is equivalent to a blocking enqueue of the workload.

        // Read the result back from the shard at mesh coordinate {0,0}. Use blocking=true to wait for completion.
        // The vector is automatically resized to fit the data.
        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, output_dram_buffer, /*blocking*/ true);

        // Compare the result with the input. The result should be the same as the input.
        TT_FATAL(
            result_vec.size() == input_vec.size(),
            "Result vector size {} does not match input vector size {}",
            result_vec.size(),
            input_vec.size());
        for (int i = 0; i < input_vec.size(); i++) {
            if (input_vec[i] != result_vec[i]) {
                pass = false;
                break;
            }
        }

        // Close the device
        if (!mesh_device->close()) {
            pass = false;
        }

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
