// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;

/*
 * 1. Host creates one vector of data.
 * 2. Device eltwise performs a unary SFPU operation on the data.
 * 3. Read result back and compare to golden.
 * */
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    bool pass = true;

    try {
        // Initialize the device (here we use the 1st device, but you can use any device)
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        // In Metalium, submitting operations to the device is done through a command queue. This includes
        // uploading/downloading data to/from the device, and executing programs.
        CommandQueue& cq = device->command_queue();
        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a give time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        Program program = CreateProgram();

        // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        // Allocate DRAM buffers for the input and output data.
        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = tile_size_bytes * n_tiles,
            .page_size = tile_size_bytes,
            .buffer_type = tt_metal::BufferType::DRAM};

        // Allocate a input and output buffer on DRAM. We will perform an unary operation on the input buffer and write
        // the result
        std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
        std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

        // Allocate 2 circular buffers for input and output.
        constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
        constexpr uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, tile_size_bytes);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_input_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, tile_size_bytes);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        // Create the 2 data movement kernels and the compute kernel.
        KernelHandle unary_reader_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/read_tile.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        KernelHandle unary_writer_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/write_tile.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        KernelHandle eltwise_sfpu_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = false,
            });

        // Initialize the input data with random values and use as the input to the kernel.
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }

        // Write the data on host to the input buffer on the device.
        // setting blocking to false allows us to overlap the data movement and following host operations (
        // setting kerenel args) in this case
        EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);

        // setup the runtime arguments for the kernels.
        SetRuntimeArgs(program, eltwise_sfpu_kernel_id, core, {n_tiles});
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src0_dram_buffer->address(),
                n_tiles,
            });

        SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), n_tiles});

        // execute the program and wait for it to finish
        EnqueueProgram(cq, program, false);
        Finish(cq);

        // Read the result and compare to our expected result
        std::vector<bfloat16> result_vec;
        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

        // Compute the same thing on CPU for comparison

        // Compare the result to the golden vector. Loose tolerance of 2e-2f because bfloat16 is not as accurate as 32
        // bit float.
        constexpr float eps = 5e-2f;
        for (uint32_t i = 0; i < result_vec.size(); ++i) {
            float expected = bfloat16(std::exp(src0_vec[i].to_float())).to_float();
            float result = result_vec[i].to_float();
            if (std::abs(expected - result) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: {} != {}\n", i, expected, result);
            }
        }

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
