// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt::tt_metal;

/*
 * This test serves as a simple, stable tt_metal executable that issues both
 * reads and writes from Tensix to the NoC. It is used to do sanity checking of
 * the Device Profiler's NoC event capture feature during CI in
 * test_device_profiler.py.
 */

int main() {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);
        CommandQueue& cq = device->command_queue();
        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        // See kernel cpp code for details on which noc calls are captured
        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_noc_event_profiler/kernels/loopback_dram_copy.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        // boilerplate setup for reading and writing multiple tiles from DRAM
        constexpr uint32_t single_tile_size = 2 * (32 * 32);
        constexpr uint32_t num_tiles = 5;
        constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

        tt::tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt::tt_metal::BufferType::DRAM};
        tt::tt_metal::InterleavedBufferConfig l1_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt::tt_metal::BufferType::L1};

        auto l1_buffer = CreateBuffer(l1_config);

        auto input_dram_buffer = CreateBuffer(dram_config);
        const uint32_t input_dram_buffer_addr = input_dram_buffer->address();

        auto output_dram_buffer = CreateBuffer(dram_config);
        const uint32_t output_dram_buffer_addr = output_dram_buffer->address();

        // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
        const uint32_t input_bank_id = 0;
        const uint32_t output_bank_id = 0;

        const std::vector<uint32_t> runtime_args = {
            l1_buffer->address(),
            input_dram_buffer->address(),
            input_bank_id,
            output_dram_buffer->address(),
            output_bank_id,
            l1_buffer->size()};
        SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

        EnqueueProgram(cq, program, false);
        Finish(cq);

        // It is necessary to explictly dump profile results at the end of the
        // program to get noc traces for standalone tt_metal programs.  For
        // ttnn, this is called _automatically_
        detail::DumpDeviceProfileResults(device);

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Test failed with exception!");
        log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
