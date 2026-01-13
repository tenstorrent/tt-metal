// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
#include <cstdint>
#include <cstdlib>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

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

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size =
            single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        auto src_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src_addr = src_dram_buffer->address();
        auto dst_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        int num_cbs = 1;  // works at the moment
        TT_FATAL(num_tiles % num_cbs == 0, "num_tiles must be divisible by num_cbs");
        int num_tiles_per_cb = num_tiles / num_cbs;

        uint32_t cb0_index = 0;
        uint32_t num_cb_tiles = 8;
        tt_metal::CircularBufferConfig cb0_config =
            tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb0_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb0_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb0_config);

        uint32_t cb1_index = 8;
        tt_metal::CircularBufferConfig cb1_config =
            tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb1_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb1_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb1_config);

        uint32_t cb2_index = 16;
        tt_metal::CircularBufferConfig cb2_config =
            tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb2_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb2_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb2_config);

        uint32_t cb3_index = 24;
        tt_metal::CircularBufferConfig cb3_config =
            tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb3_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb3_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb3_config);

        std::vector<uint32_t> reader_cb_kernel_args = {8, 2};
        std::vector<uint32_t> writer_cb_kernel_args = {8, 4};

        auto reader_cb_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_cb_test.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = reader_cb_kernel_args});

        auto writer_cb_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_cb_test.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = writer_cb_kernel_args});

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

        tt_metal::SetRuntimeArgs(
            program,
            reader_cb_kernel,
            core,
            {dram_buffer_src_addr,
            0,
            (uint32_t)num_tiles_per_cb});

        tt_metal::SetRuntimeArgs(
            program,
            writer_cb_kernel,
            core,
            {dram_buffer_dst_addr,
            0,
            (uint32_t)num_tiles_per_cb});

        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src_vec == result_vec);

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
