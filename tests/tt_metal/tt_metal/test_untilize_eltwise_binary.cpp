// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"

using namespace tt;

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
            for (int i = 0; i < 16; i++) { // num rows in a face
                for (int j = 0; j < num_tile_cols; j++) { // num columns top two faces
                    // Left face row copy
                    for (int k = 0; k < 8; k++) {
                        int idx = physical_start_for_tile_row + i * 8 + k + j * tile_size;
                        TT_FATAL(ind.find(idx) == ind.end(), t);
                        ind.insert(idx);
                        dst_vec.push_back(src_vec.at(idx));
                    }

                    // Right face row copy
                    for (int k = 0; k < 8; k++) {
                        int idx = physical_start_for_tile_row + i * 8 + k + face_size + j * tile_size;
                        TT_FATAL(ind.find(idx) == ind.end(), t);
                        ind.insert(idx);
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            physical_start_for_tile_row += 2 * face_size; // Move to bottom faces
        }
    }

    return dst_vec;
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    bool pass = true;
    bool multibank = true;

    const char* op_id_to_op_define[] = {"add_tiles"};
    const char* op_id_to_op_name[] = {"ADD"};

    auto eltwise_op = EltwiseOp::ADD;
    log_info(LogTest, "====================================================================");
    log_info(LogTest, "======= Running eltwise_binary test for op={}", op_id_to_op_name[eltwise_op]);
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

        uint32_t num_blocks = 1;
        uint32_t num_tiles_r = 2;
        uint32_t num_tiles_c = 2;

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = num_blocks * num_tiles_r * num_tiles_c;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t page_size = single_tile_size;
        if (not multibank) {
            page_size = dram_buffer_size;
        }

        tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = dram_buffer_size,
                    .page_size = page_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

        auto src0_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

        auto src1_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();

        auto dst_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();


        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = num_tiles_c;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t untilized_src0_cb_index = 24;
        tt_metal::CircularBufferConfig cb_untilized_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{untilized_src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(untilized_src0_cb_index, single_tile_size);
        auto cb_untilized_src0 = tt_metal::CreateCircularBuffer(program, core, cb_untilized_src0_config);

        uint32_t src1_cb_index = 1;
        tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = num_tiles_c;
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        auto binary_reader_kernel = tt_metal::CreateKernel(
            program,
            multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp"
                      : "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"
                      : "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            num_blocks,
            num_tiles_r,
            num_tiles_c
        };

        auto eltwise_binary_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/untilA_elwbin_3m.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = {{"ELTWISE_OP", op_id_to_op_define[eltwise_op]}}}
        );

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
            {dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            num_tiles,
            dram_buffer_src1_addr,
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            num_tiles, 0});

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles});

        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        vector<uint32_t> golden = gold_standard_untilize(src0_vec, {num_tiles_r * 32, num_tiles_c * 32});

        if (not pass){
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles, "result");
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles, "golden");
        }

        pass &= (golden == result_vec);

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

    TT_FATAL(pass);

    return 0;
}
