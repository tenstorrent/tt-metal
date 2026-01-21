// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

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
#include <set>
#include <string>
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
#include "test_gold_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "impl/data_format/bfloat16_utils.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

inline std::vector<uint32_t> gold_standard_untilize(std::vector<uint32_t> src_vec, std::vector<uint32_t> shape) {
    std::vector<uint32_t> dst_vec;

    int num_rows = shape.at(0);
    int num_cols = shape.at(1) / 2;

    int num_tile_rows = num_rows / 32;
    int num_tile_cols = num_cols / 16;

    int face_size = 16 * 8;
    int tile_size = face_size * 4;

    std::set<int> ind;

    // Iterate over tile rows
    for (int t = 0; t < num_tile_rows; t++) {
        int tile_start_index = t * num_tile_cols;

        int physical_start_for_tile_row = tile_start_index * 32 * 16;

        // Iterate over tile columns 32 times (naive, but simple for validation)
        for (int x = 0; x < 2; x++) {
            for (int i = 0; i < 16; i++) {                 // num rows in a face
                for (int j = 0; j < num_tile_cols; j++) {  // num columns top two faces
                    // Left face row copy
                    for (int k = 0; k < 8; k++) {
                        int idx = physical_start_for_tile_row + (i * 8) + k + (j * tile_size);
                        TT_FATAL(!ind.contains(idx), "{}", t);
                        ind.insert(idx);
                        dst_vec.push_back(src_vec.at(idx));
                    }

                    // Right face row copy
                    for (int k = 0; k < 8; k++) {
                        int idx = physical_start_for_tile_row + (i * 8) + k + face_size + (j * tile_size);
                        TT_FATAL(!ind.contains(idx), "{}", t);
                        ind.insert(idx);
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            physical_start_for_tile_row += 2 * face_size;  // Move to bottom faces
        }
    }

    return dst_vec;
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
TEST_F(MeshDeviceSingleCardFixture, UntilizeEltwiseBinary) {
    IDevice* dev = devices_[0]->get_devices()[0];
    bool pass = true;
    bool multibank = true;

    const char* op_id_to_op_define[] = {"add_tiles"};
    const char* op_id_to_op_name[] = {"ADD"};

    auto eltwise_op = EltwiseOp::ADD;
    log_info(LogTest, "====================================================================");
    log_info(LogTest, "======= Running eltwise_binary test for op={}", op_id_to_op_name[eltwise_op]);
    try {
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};

        uint32_t num_blocks = 1;
        uint32_t num_tiles_r = 2;
        uint32_t num_tiles_c = 2;

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = num_blocks * num_tiles_r * num_tiles_c;
        uint32_t dram_buffer_size =
            single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t page_size = single_tile_size;
        if (not multibank) {
            page_size = dram_buffer_size;
        }

        tt_metal::InterleavedBufferConfig dram_config{
            .device = dev, .size = dram_buffer_size, .page_size = page_size, .buffer_type = tt_metal::BufferType::DRAM};

        auto src0_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

        auto src1_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();

        auto dst_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = num_tiles_c;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t untilized_src0_cb_index = 24;
        tt_metal::CircularBufferConfig cb_untilized_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{untilized_src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(untilized_src0_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_untilized_src0_config);

        uint32_t src1_cb_index = 1;
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_tiles = num_tiles_c;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> reader_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);
        auto binary_reader_kernel = tt_metal::CreateKernel(
            program,
            multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp"
                      : "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"
                      : "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        vector<uint32_t> compute_kernel_args = {num_blocks, num_tiles_r, num_tiles_c};

        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/untilA_elwbin_3m.cpp",
            core,
            tt_metal::ComputeConfig{
                .compile_args = compute_kernel_args, .defines = {{"ELTWISE_OP", op_id_to_op_define[eltwise_op]}}});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);

        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);

        tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_vec);

        tt_metal::SetRuntimeArgs(
            program,
            binary_reader_kernel,
            core,
            {dram_buffer_src0_addr, (uint32_t)0, num_tiles, dram_buffer_src1_addr, (uint32_t)0, num_tiles, 0});

        tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, {dram_buffer_dst_addr, (uint32_t)0, num_tiles});

        tt_metal::detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        vector<uint32_t> golden = gold_standard_untilize(src0_vec, {num_tiles_r * 32, num_tiles_c * 32});

        if (not pass) {
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles, "result");
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles, "golden");
        }

        pass &= (golden == result_vec);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    ASSERT_TRUE(pass);
}
