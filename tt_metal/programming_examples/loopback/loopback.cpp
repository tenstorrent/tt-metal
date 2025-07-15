// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

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
    // Fast Dispatch = support for async operations. We need it for most applications.
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        // Initialize the device (here we use the 1st device, but you can use any device)
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        // In Metalium, submitting operations to the device is done through a command queue. This includes
        // uploading/downloading data to/from the device, and executing programs.
        CommandQueue& cq = device->command_queue();
        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a given time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        Program program = CreateProgram();

        // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
        constexpr CoreCoord core = {0, 0};

        // Create the data movement kernel. This kernel will be used to copy data from DRAM to DRAM (see the
        // `loopback_dram_copy.cpp` file for the actual implementation). The kernel is created on the Tensix core
        // {0, 0} and uses the default NoC.
        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "loopback/kernels/loopback_dram_copy.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        // Data on Tensix is stored in tiles. A tile is a 2D array of (usually) 32x32 values. And the Tensix uses
        // BFloat16 as the most well supported data type. Thus the tile size is 32x32x2 = 2048 bytes.
        constexpr uint32_t num_tiles = 50;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
        constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

        // Configuration for the buffers.
        tt::tt_metal::InterleavedBufferConfig dram_config{
            .device = device,              // Device which owns the buffer
            .size = dram_buffer_size,      // Size of the buffer in bytes
            .page_size = tile_size_bytes,  // Number of bytes when round-robin between banks. Usually this is the same
                                           // as the tile size for efficiency.
            .buffer_type = tt::tt_metal::BufferType::DRAM};  // Type of buffer (DRAM or L1(SRAM))
        tt::tt_metal::InterleavedBufferConfig l1_config{
            .device = device,
            .size = tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1};  // This time we allocate on L1

        // Allocate the buffers
        auto l1_buffer = CreateBuffer(l1_config);
        auto input_dram_buffer = CreateBuffer(dram_config);
        auto output_dram_buffer = CreateBuffer(dram_config);

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
        EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, /*blocking=*/false);

        // Set the arguments for the kernel.
        const std::vector<uint32_t> runtime_args = {
            l1_buffer->address(), input_dram_buffer->address(), output_dram_buffer->address(), num_tiles};

        SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

        // Run the program. Again blocking is set to false. So the host function returns immediately and can continue
        // executing while the program is running on the device; leading the better performance if the host has other
        // work to do.
        EnqueueProgram(cq, program, /*blocking=*/false);
        Finish(cq);
        // NOTE: The above is equivalent to the following single line:
        // EnqueueProgram(cq, program, /*blocking=*/true);

        // Read the result back from the device. The `blocking` argument is set to true. Telling Metalium to wait for
        // the read to complete before returning. Thus we can be sure the data is ready to be used right after the call.
        // The vector will be automatically resized to fit the data.
        std::vector<bfloat16> result_vec;
        EnqueueReadBuffer(cq, output_dram_buffer, result_vec, /*blocking*/ true);

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
        if (!CloseDevice(device)) {
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
