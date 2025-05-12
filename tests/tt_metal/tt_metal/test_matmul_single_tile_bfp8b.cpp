// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <errno.h>
#include <fmt/base.h>
#include <stdint.h>
#include <stdlib.h>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/util.hpp>
#include <algorithm>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

int main(int argc, char** argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
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

        uint32_t single_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Bfp8_b);
        TT_FATAL(single_tile_size == (256 * 4) + (16 * 4), "Error");
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles;  // num_tiles of BFP8_B

        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        auto src0_dram_buffer = CreateBuffer(dram_config);
        auto src1_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 1;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Bfp8_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Bfp8_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_tiles = 1;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Bfp8_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        auto mm_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            1,  // block_tile_dim
            1,  // dst_tile_rows
            1,  // dst_tile_cols
            1,  // block_cnt
            1,  // in0_block_tile_cnt
            1,  // in1_block_tile_cnt
            1   // out_block_tile_cnt
        };

        auto mm_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> activations = create_random_vector_of_bfp8(
            dram_buffer_size,
            /*is_exp_a=*/false,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::detail::WriteToBuffer(src0_dram_buffer, activations);

        int num_float_in_tile = 32 * 32;
        std::vector<float> vec(num_float_in_tile, (float)0);
        for (int i = 0; i < 32; i++) {
            vec.at(i * 32 + i) = (float)1;
        }
        std::vector<uint32_t> weights = pack_fp32_vec_as_bfp8_tiles(vec, /*row_major_input=*/true, /*is_exp_a=*/false);

        tt_metal::detail::WriteToBuffer(src1_dram_buffer, weights);

        tt_metal::SetRuntimeArgs(
            program,
            mm_reader_kernel,
            core,
            {src0_dram_buffer->address(),
            0,
            src1_dram_buffer->address(),
            0,
            1,
            1,
            1,
            1 * single_tile_size,
            1 * single_tile_size});

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst_dram_buffer->address(),
            0,
            num_tiles});

        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass &= tt_metal::CloseDevice(device);

        pass &= (activations == result_vec);  // src1 is identity

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
