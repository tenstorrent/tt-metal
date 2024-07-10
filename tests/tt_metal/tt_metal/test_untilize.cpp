// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"

#include "llrt/llrt.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

inline vector<uint32_t> gold_standard_untilize(std::vector<uint32_t> src_vec, vector<uint32_t> shape) {
    vector<uint32_t> dst_vec;

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

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    std::vector<string> untilize_types = {"unpack", "pack"};
    Program programs[] = {tt_metal::CreateProgram(), tt_metal::CreateProgram()};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(device_id);

    CommandQueue& cq = device->command_queue();


    for (uint untilize_idx = 0;  untilize_idx < untilize_types.size(); untilize_idx++) {
        try {
            log_info(LogTest, "====================================================================");
            tt::log_info(tt::LogTest, "Running untilize test for type={}", untilize_types[untilize_idx]);
            // ////////////////////////////////////////////////////////////////////////////
            //                      Application Setup
            ////////////////////////////////////////////////////////////////////////////
            tt_metal::Program& program = programs[untilize_idx];

            CoreCoord core = {0, 0};

            uint32_t single_tile_size = 2 * 1024;

            uint32_t num_tiles_r = 1;
            uint32_t num_tiles_c = 4;
            uint32_t num_tiles = num_tiles_r * num_tiles_c;

            uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

            int dram_src_channel_id = 0;
            int dram_dst_channel_id = 0;

            tt_metal::InterleavedBufferConfig dram_config{
                        .device=device,
                        .size = dram_buffer_size,
                        .page_size = dram_buffer_size,
                        .buffer_type = tt_metal::BufferType::DRAM
            };


            auto src_dram_buffer = CreateBuffer(dram_config);
            uint32_t dram_buffer_src_addr = src_dram_buffer->address();

            auto dst_dram_buffer = CreateBuffer(dram_config);
            uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

            auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
            auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

            // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
            // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
            uint32_t src0_cb_index = 0;
            uint32_t num_input_tiles = 8;
            tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
            auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t ouput_cb_index = 16; // output operands start at index 16
            uint32_t num_output_tiles = 8;
            tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
            auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

            auto unary_reader_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/kernels/dataflow/reader_unary.cpp",
                core,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

            auto unary_writer_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/kernels/dataflow/writer_unary.cpp",
                core,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

            vector<uint32_t> compute_kernel_args = {
                1, // per_core_block_cnt
                uint(num_tiles_c) // per_core_block_tile_cnt
            };

            string untilize_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/" + untilize_types[untilize_idx] + "_untilize.cpp";

            auto eltwise_unary_kernel = tt_metal::CreateKernel(
                program,
                untilize_kernel,
                core,
                tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
            );

            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel,
                core,
                {dram_buffer_src_addr,
                (std::uint32_t)dram_src_noc_xy.x,
                (std::uint32_t)dram_src_noc_xy.y,
                num_tiles});

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel,
                core,
                {dram_buffer_dst_addr,
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                num_tiles});

            ////////////////////////////////////////////////////////////////////////////
            //                      Compile & Execute Application
            ////////////////////////////////////////////////////////////////////////////
            std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(
                dram_buffer_size, false);

            pass &= tt_metal::detail::WriteToDeviceDRAMChannel(device, dram_src_channel_id, src_dram_buffer->address(), src_vec);


            tt_metal::detail::LaunchProgram(device, program);

            std::vector<uint32_t> result_vec;
            tt_metal::detail::ReadFromDeviceDRAMChannel(
                device, dram_dst_channel_id, dst_dram_buffer->address(), dst_dram_buffer->size(), result_vec);
            ////////////////////////////////////////////////////////////////////////////
            //                      Validation & Teardown
            ////////////////////////////////////////////////////////////////////////////
            vector<uint32_t> golden = gold_standard_untilize(src_vec, {num_tiles_r * 32, num_tiles_c * 32});

            TT_FATAL(golden.size() == result_vec.size());
            pass &= (golden == result_vec);
            tt::log_info(tt::LogTest, "Test {} for type={}", pass ? "Passed" : "Failed", untilize_types[untilize_idx]);

            if (not pass) {
                std::cout << "GOLDEN" << std::endl;
                print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles);

                std::cout << "RESULT" << std::endl;
                print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles);
            }

        } catch (const std::exception &e) {
            pass = false;
            // Capture the exception error message
            log_error(LogTest, "{}", e.what());
            // Capture system call errors that may have returned from driver/kernel
            log_error(LogTest, "System error message: {}", std::strerror(errno));
        }
    }

    pass &= tt_metal::CloseDevice(device);

    if (pass) {
        log_info(LogTest, "All Tests Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
