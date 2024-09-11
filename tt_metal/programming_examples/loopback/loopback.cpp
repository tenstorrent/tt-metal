// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "common/bfloat16.hpp"

/*
* 1. Host writes data to buffer in DRAM
* 2. dram_copy kernel on logical core {0, 0} BRISC copies data from buffer
*      in step 1. to buffer in L1 and back to another buffer in DRAM
* 3. Host reads from buffer written to in step 2.
*/

using namespace tt::tt_metal;

int main(int argc, char **argv) {

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        /*
        * Silicon accelerator setup
        */
        constexpr int device_id = 0;
        Device *device =
            CreateDevice(device_id);

        /*
        * Setup program and command queue to execute along with its buffers and kernels to use
        */
        CommandQueue& cq = device->command_queue();
        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
        );

        constexpr uint32_t single_tile_size = 2 * (32 * 32);
        constexpr uint32_t num_tiles = 50;
        constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

        tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device= device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };
        tt::tt_metal::InterleavedBufferConfig l1_config{
                    .device= device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::L1
        };

        auto l1_buffer = CreateBuffer(l1_config);

        auto input_dram_buffer = CreateBuffer(dram_config);
        const uint32_t input_dram_buffer_addr = input_dram_buffer->address();

        auto output_dram_buffer = CreateBuffer(dram_config);
        const uint32_t output_dram_buffer_addr = output_dram_buffer->address();

        /*
        * Create input data and runtime arguments, then execute
        */
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);

        const std::vector<uint32_t> runtime_args = {
            l1_buffer->address(),
            input_dram_buffer->address(),
            static_cast<uint32_t>(input_dram_buffer->noc_coordinates().x),
            static_cast<uint32_t>(input_dram_buffer->noc_coordinates().y),
            output_dram_buffer->address(),
            static_cast<uint32_t>(output_dram_buffer->noc_coordinates().x),
            static_cast<uint32_t>(output_dram_buffer->noc_coordinates().y),
            l1_buffer->size()
        };

        SetRuntimeArgs(
            program,
            dram_copy_kernel_id,
            core,
            runtime_args
        );

        EnqueueProgram(cq, program, false);
        Finish(cq);

        /*
        * Validation & Teardown
        */
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(cq, output_dram_buffer, result_vec, true);

        pass &= input_vec == result_vec;

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
