// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
#include <cstdlib>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

namespace unary_datacopy {
// #include "hlks/eltwise_copy.cpp"
//  FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
};
}  // namespace unary_datacopy

int main() {
    bool pass = true;

    auto* slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();
        auto num_cores_c = 2;
        auto num_cores_r = 2;
        CoreCoord start_core = {0, 0};
        CoreCoord end_core = {(std::size_t)start_core.x + num_cores_c - 1, (std::size_t)start_core.y + num_cores_r - 1};
        CoreRange all_cores(start_core, end_core);

        int num_sticks = 4;
        int num_elements_in_stick = 512;
        int stick_size = num_elements_in_stick * 2;
        uint32_t dram_buffer_size =
            num_sticks * stick_size;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        auto src_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src_addr = src_dram_buffer->address();

        TT_FATAL(
            src_dram_buffer->size() % (num_cores_r * num_cores_c) == 0,
            "DRAM buffer size must be divisible by number of cores");
        uint32_t per_core_l1_size = src_dram_buffer->size() / (num_cores_r * num_cores_c);
        std::unordered_map<CoreCoord, uint32_t> core_to_l1_addr;
        for (int i = start_core.y; i < start_core.y + num_cores_r; i++) {
            for (int j = start_core.x; j < start_core.x + num_cores_c; j++) {
                CoreCoord core = {(std::size_t)j, (std::size_t)i};
                tt_metal::InterleavedBufferConfig l1_config{
                    .device = device,
                    .size = per_core_l1_size,
                    .page_size = per_core_l1_size,
                    .buffer_type = tt_metal::BufferType::L1};
                auto l1_b0 = CreateBuffer(l1_config);
                core_to_l1_addr[core] = l1_b0->address();
            }
        }
        auto unary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_sticks.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

        std::cout << "Num cores " << num_cores_r * num_cores_c << std::endl;
        uint32_t core_index = 0;
        for (int i = start_core.y; i < start_core.y + num_cores_r; i++) {
            for (int j = start_core.x; j < start_core.x + num_cores_c; j++) {
                CoreCoord core = {(std::size_t)j, (std::size_t)i};
                tt_metal::SetRuntimeArgs(
                    program,
                    unary_reader_kernel,
                    core,
                    {core_to_l1_addr.at(core),
                    dram_buffer_src_addr + (core_index * stick_size),
                    0,
                    (std::uint32_t) 1,
                    (std::uint32_t) stick_size});
                core_index++;
            }
        }

        tt_metal::detail::LaunchProgram(device, program);
        // std::vector<uint32_t> result_vec;
        // tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        // pass &= (src_vec == result_vec);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
