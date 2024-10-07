// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/impl/program/program_pool.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto program = tt_metal::CreateScopedProgram();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 256;
        uint32_t dram_buffer_size_bytes = single_tile_size * num_tiles;

        // L1 buffer is double buffered
        // We read and write total_l1_buffer_size_tiles / 2 tiles from and to DRAM
        uint32_t l1_buffer_addr = 400 * 1024;
        uint32_t total_l1_buffer_size_tiles = num_tiles / 2;
        TT_FATAL(total_l1_buffer_size_tiles % 2 == 0, "Error");
        uint32_t total_l1_buffer_size_bytes = total_l1_buffer_size_tiles * single_tile_size;

        tt_metal::InterleavedBufferConfig dram_config{
                                .device=device,
                                .size = dram_buffer_size_bytes,
                                .page_size = dram_buffer_size_bytes,
                                .buffer_type = tt_metal::BufferType::DRAM
                                };

        auto input_dram_buffer = CreateBuffer(dram_config);
        uint32_t input_dram_buffer_addr = input_dram_buffer->address();

        auto output_dram_buffer = CreateBuffer(dram_config);
        uint32_t output_dram_buffer_addr = output_dram_buffer->address();

        auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
        auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

        auto dram_copy_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_db.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::detail::WriteToBuffer(input_dram_buffer, input_vec);



        tt_metal::SetRuntimeArgs(
            program,
            dram_copy_kernel,
            core,
            {input_dram_buffer_addr,
            (std::uint32_t)input_dram_noc_xy.x,
            (std::uint32_t)input_dram_noc_xy.y,
            output_dram_buffer_addr,
            (std::uint32_t)output_dram_noc_xy.x,
            (std::uint32_t)output_dram_noc_xy.y,
            dram_buffer_size_bytes,
            num_tiles,
            l1_buffer_addr,
            total_l1_buffer_size_tiles,
            total_l1_buffer_size_bytes});

        auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
        tt_metal::detail::LaunchProgram(device, *program_ptr);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(output_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass = (input_vec == result_vec);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
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
