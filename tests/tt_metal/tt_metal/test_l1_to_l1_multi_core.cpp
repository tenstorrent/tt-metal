// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
// #include "tt_gdb/tt_gdb.hpp"


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

        //CoreCoord core = {0, 0};

        uint32_t num_tiles = 10;
        uint32_t tile_size_bytes = 1024 * 2;
        uint32_t total_tiles_size_bytes = num_tiles * tile_size_bytes;
        uint32_t dram_buffer_size = total_tiles_size_bytes;

        std::unordered_map<CoreCoord, uint32_t> core_to_l1_address_map;
        for(uint32_t i = 0; i < 10; i++) {
            for(uint32_t j = 0; j < i; j++) {
                CoreCoord core = {(size_t) j, (size_t) i};
                CoreCoord dst_soc_core = {(size_t) i+1, (size_t) j+1};
                if(j > 5) {
                    dst_soc_core.y += 1;
                }
                std::cout << "Sending from " << j+1 << "," << i+1 << " to " << i+1 << "," << j+1 << std::endl;
                tt_metal::InterleavedBufferConfig l1_config{
                    .device=device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt_metal::BufferType::L1
                };

                auto l1_b0 = CreateBuffer(l1_config);
                uint32_t l1_buffer_addr = l1_b0->address();
                core_to_l1_address_map.insert({core, l1_buffer_addr});

                std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(
                    dram_buffer_size, i * 10 + j);

                tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt_metal::BufferType::DRAM
                };
                auto src_dram_buffer = CreateBuffer(dram_config);
                uint32_t dram_buffer_src_addr = src_dram_buffer->address();
                auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
                tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

                auto l1_to_l1_kernel = tt_metal::CreateKernel(
                        program,
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_l1.cpp",
                        core,
                        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

                tt_metal::SetRuntimeArgs(
                        program,
                        l1_to_l1_kernel,
                        core,
                        {dram_buffer_src_addr,
                        (std::uint32_t)dram_src_noc_xy.x,
                        (std::uint32_t)dram_src_noc_xy.y,
                        l1_buffer_addr,
                        l1_buffer_addr,
                        (uint32_t)dst_soc_core.x,
                        (uint32_t)dst_soc_core.y,
                        num_tiles,
                        tile_size_bytes,
                        total_tiles_size_bytes});
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////



        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        for(uint32_t i = 0; i < 10; i++) {
            for(uint32_t j = i+1; j < 10; j++) {
                CoreCoord core = {(size_t) j, (size_t) i};
                uint32_t l1_buffer_addr = core_to_l1_address_map.at(core);
                tt_metal::detail::ReadFromDeviceL1(device, core, l1_buffer_addr, total_tiles_size_bytes, result_vec);
                std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(
                    dram_buffer_size, j * 10 + i);
                if(src_vec != result_vec) {
                    std::cout << "      Failed on core " << j+1 << "," << i+1 << std::endl;
                }
                else {
                    std::cout << "Passed on core " << j+1 << "," << i+1 << std::endl;
                }
                pass &= (src_vec == result_vec);
            }
        }

        //std::vector<uint32_t> result_vec;
        //tt_metal::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        //pass &= (src_vec == result_vec);

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
