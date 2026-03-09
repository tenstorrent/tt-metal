// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <errno.h>
#include <fmt/base.h>
#include <iomanip>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <tt-metalium/bfloat8.hpp>
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
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/bfloat_utils.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

int main(int argc, char** argv) {
    auto* slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 1;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        CoreCoord grid_size = device->compute_with_storage_grid_size();

        uint32_t single_tile_size = tt::tile_size(tt::DataFormat::Bfp8_b);
        TT_FATAL(single_tile_size == (256 * 4) + (16 * 4), "Error");
        uint32_t num_tiles = 2;
        uint32_t dram_buffer_size = single_tile_size * num_tiles;  // num_tiles of BFP8_B

        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM};
        auto src_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src_addr = src_dram_buffer->address();
        auto dst_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        std::vector<uint32_t> src_vec = test_utils::create_random_vector_of_bfp8(
            dram_buffer_size,
            /*is_exp_a=*/false,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_tiles = 2;
        vector<uint32_t> compute_kernel_args = {
            uint(num_tiles)  // per_core_tile_cnt
        };

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute on each core one by one
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> result_vec;
        for (uint32_t x = 0; x < grid_size.x; x++) {
            for (uint32_t y = 0; y < grid_size.y; y++) {
                CoreCoord core = {x, y};

                tt_metal::Program program = tt_metal::CreateProgram();

                tt_metal::CircularBufferConfig cb_src0_config =
                    tt_metal::CircularBufferConfig(
                        num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Bfp8_b}})
                        .set_page_size(src0_cb_index, single_tile_size);
                tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

                tt_metal::CircularBufferConfig cb_output_config =
                    tt_metal::CircularBufferConfig(
                        num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Bfp8_b}})
                        .set_page_size(ouput_cb_index, single_tile_size);
                tt_metal::CreateCircularBuffer(program, core, cb_output_config);

                auto unary_reader_kernel = tt_metal::CreateKernel(
                    program,
                    "tt_metal/kernels/dataflow/reader_unary.cpp",
                    core,
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

                auto unary_writer_kernel = tt_metal::CreateKernel(
                    program,
                    "tt_metal/kernels/dataflow/writer_unary.cpp",
                    core,
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

                tt_metal::CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
                    core,
                    tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

                tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

                tt_metal::SetRuntimeArgs(program, unary_reader_kernel, core, {dram_buffer_src_addr, 0, num_tiles});

                tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, {dram_buffer_dst_addr, 0, num_tiles});

                tt_metal::detail::LaunchProgram(device, program);

                tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

                bool core_pass = (src_vec == result_vec);
                pass &= core_pass;
                if (!core_pass) {
                    log_error(LogTest, "Datacopy BFP8b failed on core ({}, {})", x, y);
                    std::cout << "Expected (core " << x << "," << y << "):\n";
                    size_t src_half = src_vec.size() / 2;
                    for (size_t i = 0; i < src_vec.size(); ++i) {
                        std::cout << std::hex << std::setw(8) << src_vec[i] << " ";
                        if ((i + 1) % 16 == 0) {
                            std::cout << std::endl;
                        }
                        if ((i + 1) == src_half) {
                            std::cout << std::endl;
                        }
                    }
                    std::cout << std::dec << std::endl;
                    std::cout << "Result (core " << x << "," << y << "):\n";
                    size_t res_half = result_vec.size() / 2;
                    for (size_t i = 0; i < result_vec.size(); ++i) {
                        std::cout << std::hex << std::setw(8) << result_vec[i] << " ";
                        if ((i + 1) % 16 == 0) {
                            std::cout << std::endl;
                        }
                        if ((i + 1) == res_half) {
                            std::cout << std::endl;
                        }
                    }
                    std::cout << std::dec << std::endl;
                } else {
                    log_info(LogTest, "Datacopy BFP8b passed on core ({}, {})", x, y);
                }
            }
        }

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
