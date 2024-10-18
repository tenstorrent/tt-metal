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
#include "test_tiles.hpp"

using namespace tt;

// Given a tensor that is row-major datums, make it tilized
// so that its row major within a tile, and each tile's data
// is contiguous
template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    int num_tiles_r = rows / 32;
    int num_tiles_c = cols / 32;
    std::vector<T> result;
    for(auto r = 0; r < num_tiles_r; r++) {
        for(auto c = 0; c < num_tiles_c; c++) {
            for(auto j = 0; j < 32; j++) { // tile rows
                for(auto i = 0; i < 32; i++) { // tile cols
                    // each row of tiles is 32x32 * num_tiles_c
                    // each row within the row of tiles is cols
                    // each col of tiles is 32
                    // pick row of tiles, pick the row within the tile, pick col tile
                    int index = r * 32 * 32 * num_tiles_c + j * cols + c * 32 + i;
                    result.push_back(data.at(index));
                }
            }
        }
    }
    return result;
}


// Given a tilized data (each tile's data is contiguous and row major within the tile)
// transform it back to row major full tensor. (This function inverts the tilize() function)
template <typename T>
std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
    int num_tiles_r = rows / 32;
    int num_tiles_c = cols / 32;
    std::vector<T> result;
    for(auto r = 0; r < num_tiles_r; r++) {
        for(auto i = 0; i < 32; i++) {
            for(auto c = 0; c < num_tiles_c; c++) {
                int offset = r * 32 * 32 * num_tiles_c + c * 32 * 32 + i * 32;
                for(auto j = 0; j < 32; j++) {
                    result.push_back(data.at(offset + j));
                }
            }
        }
    }

    return result;
}

void print_vec(std::vector<bfloat16> data, int rows, int cols, string name) {
    std::cout<<name<<": "<<std::endl;
    int index = 0;
    for(int i = 0 ; i < rows ; i++) {
        for(int j = 0 ; j < cols; j++) {
            std::cout<<data.at(index).to_float()<<", ";
            index++;
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

// creates a bfloat16 diagonal matrix with dim(rowsXrows)
inline std::vector<bfloat16> create_diagonal_matrix(int rows, float val) {
    std::vector<bfloat16> vec(rows * rows, (float)0);
    for(int i = 0; i < rows; i++) {
        vec.at(i * rows + i) = bfloat16((float)val);
    }
    return vec;
}

// creates a bfloat16 diagonal matrix with dim(rowsXrows)
// each 2 cols will be packed as a single uint32_t
inline std::vector<uint32_t> create_diagonal_matrix_in_tile_layout(int rows, float val) {
    auto vec = create_diagonal_matrix(rows, val);
    auto vec_tilized = tilize(vec, rows, rows);
    auto vec_tile_layout = convert_to_tile_layout(vec_tilized);
    return pack_bfloat16_vec_into_uint32_vec(vec_tile_layout);
}

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
        CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});

        uint32_t single_tile_size = 2 * 1024;
        tt_metal::InterleavedBufferConfig single_tile_config{
                    .device=device,
                    .size = single_tile_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

        auto q_buffer = CreateBuffer(single_tile_config);
        auto k_buffer = CreateBuffer(single_tile_config);
        auto v_buffer = CreateBuffer(single_tile_config);
        auto out_buffer = CreateBuffer(single_tile_config);

        uint32_t q_addr = q_buffer->address();
        uint32_t k_addr = k_buffer->address();
        uint32_t v_addr = v_buffer->address();
        uint32_t out_addr = out_buffer->address();

        // Q input
        auto c_in0_config = tt::tt_metal::CircularBufferConfig(single_tile_size, {{CB::c_in0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in0, single_tile_size);
        auto cb_in0_id = CreateCircularBuffer(program, cores, c_in0_config);

        // K input
        auto c_in1_config = tt::tt_metal::CircularBufferConfig(single_tile_size, {{CB::c_in1, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in1, single_tile_size);
        auto cb_in1_id = CreateCircularBuffer(program, cores, c_in1_config);

        // V input
        auto c_in2_config = tt::tt_metal::CircularBufferConfig(single_tile_size, {{CB::c_in2, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in2, single_tile_size);
        auto cb_in2_id = CreateCircularBuffer(program, cores, c_in2_config);

        // identity scale input
        auto c_in5_config = tt::tt_metal::CircularBufferConfig(single_tile_size, {{CB::c_in5, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in5, single_tile_size);
        auto cb_in5_id = CreateCircularBuffer(program, cores, c_in5_config);

        // cb_qk_im
        auto c_intermed0_config = tt::tt_metal::CircularBufferConfig(single_tile_size, {{CB::c_intermed0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed0, single_tile_size);
        auto cb_intermed0_id = CreateCircularBuffer(program, cores, c_intermed0_config);

        // cb_cur_sum
        auto c_intermed5_config = tt::tt_metal::CircularBufferConfig(single_tile_size, {{CB::c_intermed5, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed5, single_tile_size);
        auto cb_intermed5_id = CreateCircularBuffer(program, cores, c_intermed5_config);

        // Output
        auto c_out0_config = tt::tt_metal::CircularBufferConfig(single_tile_size, {{CB::c_out0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_out0, single_tile_size);
        auto cb_out0_id = CreateCircularBuffer( program, cores, c_out0_config );

        std::map<string, string> defines;
        std::map<string, string> compute_defines;
        std::vector<uint32_t> compile_time_args;

        auto reader_kernels_id = CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/reader_interleaved.cpp", cores,
            tt_metal::ReaderDataMovementConfig(
               compile_time_args,
                defines
        ));

        auto writer_kernels_id = CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/writer_interleaved.cpp", cores,
            tt_metal::WriterDataMovementConfig(
                compile_time_args,
                defines
        ));

        const bool enable_nops = std::getenv("TT_NOP_INSERT");
        if(enable_nops) {
            const uint32_t num_nops = enable_nops ? std::stoi( std::getenv("TT_NOP_INSERT") ) : 0;
            std::cout << "Should insert nops # " << num_nops << std::endl;
            compute_defines["MM_ADD_NOPS"] = "1";
            compute_defines["MM_NUM_NOPS"] = std::to_string(num_nops);
        }

        auto compute_kernels_id = CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/sdpa.cpp", cores,
            tt_metal::ComputeConfig{
                .compile_args = compile_time_args,
                .defines = compute_defines
        });


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        auto Q = create_diagonal_matrix_in_tile_layout(32, 5);
        auto K = create_diagonal_matrix_in_tile_layout(32, 7);
        auto V = create_diagonal_matrix_in_tile_layout(32, 9);
        tt_metal::detail::WriteToBuffer(q_buffer, Q);
        tt_metal::detail::WriteToBuffer(k_buffer, K);
        tt_metal::detail::WriteToBuffer(v_buffer, V);

        SetRuntimeArgs(program, reader_kernels_id, core, { q_addr, k_addr, v_addr});
        SetRuntimeArgs(program, writer_kernels_id, core, { out_addr});
        SetRuntimeArgs(program, compute_kernels_id, core, {});

        log_info(LogTest, "Launching kernels");
        tt_metal::detail::LaunchProgram(device, program);
        log_info(LogTest, "Kernels done");
        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(out_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        auto result_flat_layout = convert_to_flat_layout(result_bfp16);
        auto result_untilized = untilize(result_flat_layout, 32, 32);
        auto golden = create_diagonal_matrix(32, 316.0f);
        if (std::equal(golden.begin(), golden.end(), result_untilized.begin())) {
	    std::cout << "Result matched with Golden" << std::endl;
	} else {
	    std::cout << "XXX Result DID NOT match with  Golden" << std::endl;
            print_vec(result_untilized, 32, 32, "Output");
	}

        //pass &= check_bug(result_untilized);
        pass &= tt_metal::CloseDevice(device);;
        log_info(LogTest, "Closing device");

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

    return 0;
}
