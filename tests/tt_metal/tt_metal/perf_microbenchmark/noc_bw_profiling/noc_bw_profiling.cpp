// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/device.hpp"

/*
 * 1. Host writes data to buffer in DRAM
 * 2. dram_copy kernel on logical core {0, 0} BRISC copies data from buffer
 *      in step 1. to buffer in L1 and back to another buffer in DRAM
 * 3. Host reads from buffer written to in step 2.
 */

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        /*
         * Silicon accelerator setup
         */
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        /*
         * Setup program and command queue to execute along with its buffers and kernels to use
         */
        CommandQueue& cq = device->command_queue();
        Program program = CreateProgram();

        const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device->id());
        std::vector<CoreCoord> dram_cores = soc_d.get_dram_cores();
        // for (const auto& core : dram_cores) {
        //     tt::log_info(tt::LogTest, "DRAM core: {}, {}", core.x, core.y);
        // }

        int page_size = 8192;
        int buffer_size = page_size * 100;

        tt::tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = buffer_size,
            .page_size = page_size,
            .buffer_type = tt::tt_metal::BufferType::DRAM};
        auto output_dram_buffer = CreateBuffer(dram_config);

        tt::tt_metal::InterleavedBufferConfig l1_config{
            .device = device, .size = buffer_size, .page_size = page_size, .buffer_type = tt::tt_metal::BufferType::L1};
        auto l1_buffer = CreateBuffer(l1_config);

        std::vector<std::pair<CoreCoord, CoreCoord>> pairs = {
            //{CoreCoord(3, 0), CoreCoord(5, 0)},
            //{CoreCoord(3, 0), CoreCoord(5, 1)},
            //{CoreCoord(3, 1), CoreCoord(5, 2)},
            //{CoreCoord(3, 2), CoreCoord(5, 3)},

            //{CoreCoord(3, 5), CoreCoord(5, 8)},
            //{CoreCoord(5, 6), CoreCoord(5, 9)},
            {CoreCoord(5, 7), CoreCoord(5, 10)},
        };

        fmt::println("dram based address is {:8x}", output_dram_buffer->address());

        for (const auto& [src, dst] : pairs) {
            std::vector<uint32_t> runtime_args = {
                l1_buffer->address(), output_dram_buffer->address(), dst.x, dst.y, page_size};

            KernelHandle kernel_id = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/noc_bw_profiling/kernels/noc_read.cpp",
                src,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1});
            SetRuntimeArgs(program, kernel_id, src, runtime_args);
        }

        EnqueueProgram(cq, program, false);
        Finish(cq);
        DumpDeviceProfileResults(device, program);

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
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
