// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
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
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t local_buffer_addr = 200 * 1024;

        uint32_t dest_buffer_addr = 500 * 1024;

        tt_metal::InterleavedBufferConfig dram_config{
                                .device=device,
                                .size = dram_buffer_size,
                                .page_size = dram_buffer_size,
                                .buffer_type = tt_metal::BufferType::DRAM
                                };
        auto dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_addr = dram_buffer->address();
        auto dram_noc_xy = dram_buffer->noc_coordinates();

        CoreCoord core_start = {0, 0};
        CoreCoord grid_size = device->logical_grid_size();
        CoreCoord core_end = {core_start.x + (grid_size.x - 1), core_start.y + (grid_size.y - 1)};
        auto core_start_physical = device->worker_core_from_logical_core(core_start);
        auto core_end_physical = device->worker_core_from_logical_core(core_end);
        std::vector<uint32_t> mcast_reader_args = {
            (std::uint32_t)dram_buffer_addr,
            (std::uint32_t)dram_noc_xy.x,
            (std::uint32_t)dram_noc_xy.y,
            (std::uint32_t)dram_buffer_size,
            (std::uint32_t)local_buffer_addr,
            (std::uint32_t)dest_buffer_addr,
            (std::uint32_t)core_end_physical.x,
            (std::uint32_t)core_end_physical.y,
            (std::uint32_t)core_start_physical.x,
            (std::uint32_t)core_start_physical.y,
            (std::uint32_t)(grid_size.x * grid_size.y)
        };
        log_info(LogTest, "Start = {}, {}", core_start_physical.x, core_start_physical.y);
        log_info(LogTest, "End = {}, {}", core_end_physical.x, core_end_physical.y);
        auto mcast_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_multicast_include_src.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, 32, 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
        tt_metal::detail::WriteToBuffer(dram_buffer, activations);


        tt_metal::SetRuntimeArgs(program, mcast_reader_kernel, core, mcast_reader_args);



        log_info(LogTest, "Launching kernels");
        tt_metal::detail::LaunchProgram(device, program);
        log_info(LogTest, "Kernels done");

        for(int i = 0 ; i < grid_size.y; i++) {
            for(int j = 0 ; j < grid_size.x; j++) {
                CoreCoord dest_core = {(std::size_t) core_start.x + j, (std::size_t) core_start.y + i};
                std::vector<uint32_t> dest_core_data;
                tt_metal::detail::ReadFromDeviceL1(device, dest_core, dest_buffer_addr, dram_buffer_size, dest_core_data);
                auto dest_core_data_unpacked = unpack_uint32_vec_into_bfloat16_vec(dest_core_data);
                pass &= (dest_core_data_unpacked == tensor.get_values());
                if(not (dest_core_data_unpacked == tensor.get_values())) {
                    log_info(LogTest, "Mismatch on core {}, {}", dest_core.x, dest_core.y);
                    print_vec_of_bfloat16(dest_core_data_unpacked, 1, "Result");
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
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
