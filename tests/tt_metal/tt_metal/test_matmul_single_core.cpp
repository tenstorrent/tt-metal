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

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

// Given a tensor that is row-major datums, make it tilized
// so that its row major within a tile, and each tile's data
// is contiguous
template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    TT_FATAL(rows % 32 == 0);
    TT_FATAL(cols % 32 == 0);
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
    TT_FATAL(rows % 32 == 0);
    TT_FATAL(cols % 32 == 0);
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

// Transpose 2D matrix of tiles so that its column major of tiles instead of row major.
// this is usually used for activation so that blocks data is contiguous in memory
// until we have a more generalized read kernel that can read tiles from different
// location in memory to make up a block in the activations CB
std::vector<std::uint32_t> transpose_tiles(std::vector<std::uint32_t> data, int row_tiles, int col_tiles, int in0_block_w) {
    std::vector<std::uint32_t> result;
    int tile_size = 512;
    for(int c = 0; c < col_tiles; c+=in0_block_w) {
        for(int r = 0 ; r < row_tiles; r++) {
            for(int k = 0; k < in0_block_w; k++) {
                int offset = tile_size * col_tiles * r + c * tile_size + k * tile_size;
                for(int i = 0; i < tile_size; i++) {
                    result.push_back(data.at(offset + i));
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

void print_faces(std::vector<bfloat16> data, string name) {
    std::cout<<name<<": "<<std::endl;
    int index = 0;

    int tile_index = 0;
    int face_index = 0;
    for(int i = 0; i < data.size(); i++) {
        if(i % 256 == 0 ){
            std::cout<<"Tile "<<tile_index / 4<<std::endl;
            std::cout<<"Face = "<<face_index<<std::endl;
            face_index++;
            tile_index++;
            if(face_index == 4) {
                face_index = 0;
            }
        }
        std::cout<<data.at(i).to_float()<<", ";
        if( (i+1) % 16 == 0) {
            std::cout<<std::endl;
        }
    }
    std::cout<<std::endl;
}

std::vector<bfloat16> select_columns(std::vector<bfloat16> data, int M, int K, int min_K_N) {
    if(min_K_N == K) {
        return data;
    }
    if(min_K_N > K) {
        TT_FATAL(false);
    }
    std::vector<bfloat16> result;
    for(int i = 0; i < M * 32; i++) {
        for(int j = 0; j < min_K_N * 32; j++) {
            int offset = i * K * 32;
            result.push_back(data.at(offset + j));
        }
    }
    return result;
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
        uint32_t M = 16;
        uint32_t K = 16 * 12;
        uint32_t N = 16;
        int out_subblock_h = 4;
        int out_subblock_w = 2;
        int in0_block_w = 2;
        log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
        log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
        log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
        log_info(LogTest, "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}", out_subblock_h, in0_block_w, K / in0_block_w, M / out_subblock_h);
        log_info(LogTest, "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}", out_subblock_w, in0_block_w, K / in0_block_w, N / out_subblock_w);

        uint32_t single_tile_size = 2 * 1024;
        TT_FATAL(M * in0_block_w * single_tile_size * 2 <= 130*1024);
        TT_FATAL(N * in0_block_w * single_tile_size * 2 <= 130*1024);
        TT_FATAL(M * N * single_tile_size <= 540*1024);
        uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        tt_metal::InterleavedBufferConfig act_config{
                    .device=device,
                    .size = dram_buffer_size_act,
                    .page_size = dram_buffer_size_act,
                    .buffer_type = tt_metal::BufferType::DRAM
        };
        tt_metal::InterleavedBufferConfig weights_config{
                    .device=device,
                    .size = dram_buffer_size_weights,
                    .page_size = dram_buffer_size_weights,
                    .buffer_type = tt_metal::BufferType::DRAM
        };
        tt_metal::InterleavedBufferConfig dst_config{
                    .device=device,
                    .size = dram_buffer_size_out,
                    .page_size = dram_buffer_size_out,
                    .buffer_type = tt_metal::BufferType::DRAM
        };


        auto src0_dram_buffer = CreateBuffer(act_config);
        auto src1_dram_buffer = CreateBuffer(weights_config);
        auto dst_dram_buffer = CreateBuffer(dst_config);

        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t cb0_tiles = M * in0_block_w * 2;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        uint32_t cb1_tiles = N * in0_block_w * 2;
        tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t interm0_cb_index = 24;
        std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
            {ouput_cb_index, tt::DataFormat::Float16_b},
            {interm0_cb_index, tt::DataFormat::Float16_b}
        };
        uint32_t num_output_tiles = M * N;
        CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_data_format_spec)
            .set_page_size(interm0_cb_index, single_tile_size)
            .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_output_config);

        std::vector<uint32_t> mm_reader_rt_args{
            src0_dram_buffer->address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            src1_dram_buffer->address(),
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            (std::uint32_t)(K/in0_block_w), // num_blocks
            M * in0_block_w, // input 0 block num tiles
            N * in0_block_w, // input 1 block num tiles
            M * in0_block_w * single_tile_size, // input 0 block bytes
            N * in0_block_w * single_tile_size}; // input 1 block bytes

        std::vector<uint32_t> writer_rt_args{
            dst_dram_buffer->address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            (std::uint32_t)out_subblock_h, // num tiles per sub block m
            (std::uint32_t)out_subblock_w, // num tiles per sub block n
            (std::uint32_t)M/out_subblock_h, // num sub blocks m
            (std::uint32_t)N/out_subblock_w, // num sub blocks n
            (std::uint32_t)out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row within sub-block
            (std::uint32_t)out_subblock_h * out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row of sub-blocks
            (std::uint32_t)out_subblock_w*single_tile_size}; // bytes offset to next sub-block

        auto mm_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        int num_blocks = (K/in0_block_w);

        int in0_num_subblocks = (M/out_subblock_h);
        int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
        int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        int in1_num_subblocks = (N/out_subblock_w);
        int in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
        int in1_per_core_w = out_subblock_w * in1_num_subblocks;

        int out_subblock_num_tiles = out_subblock_h*out_subblock_w;

        vector<uint32_t> compute_kernel_args = {
            uint(in0_block_w),
            uint(in0_num_subblocks),
            uint(in0_block_num_tiles),
            uint(in0_subblock_num_tiles),

            uint(in1_num_subblocks),
            uint(in1_block_num_tiles),
            uint(in1_per_core_w),

            uint(num_blocks),

            uint(out_subblock_h),
            uint(out_subblock_w),
            uint(out_subblock_num_tiles)
        };

        auto matmul_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////



        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, M * 32, K * 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto activations_tilized = tilize(tensor.get_values(), M * 32, K * 32);
        auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
        auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        auto activations_tile_transposed = transpose_tiles(activations, M, K, in0_block_w);
        tt_metal::detail::WriteToBuffer(src0_dram_buffer, activations_tile_transposed);

        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 32x32 identity
        auto identity_tilized = tilize(identity, K * 32, N * 32);
        auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        tt_metal::detail::WriteToBuffer(src1_dram_buffer, weights);



        tt_metal::SetRuntimeArgs(
            program,
            mm_reader_kernel,
            core,
            mm_reader_rt_args);

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            writer_rt_args);


        log_info(LogTest, "Launching kernels");
        tt_metal::detail::LaunchProgram(device, program);
        log_info(LogTest, "Kernels done");
        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        auto result_flat_layout = convert_to_flat_layout(result_bfp16);
        auto result_untilized = untilize(result_flat_layout, M*32, N*32);
        // print_vec(result_bfp16, 128, 128, "Result bfp16");
        // print_faces(unpack_uint32_vec_into_bfloat16_vec(activations_tile_transposed), "Activations tile transpose");
        // print_faces(unpack_uint32_vec_into_bfloat16_vec(weights), "Weights tile transposed");
        // print_faces(result_bfp16, "Result bfp16");
        // print_vec_of_uint32_as_packed_bfloat16(weights, 16, "weights tile transposed");
        // print_vec(result_untilized, M*32, N*32, "Result");
        // print_vec(tensor.get_values(), 128, 128, "Golden");
        auto golden = select_columns(tensor.get_values(), M, K, std::min(K, N));
        // auto golden = tensor.get_values();
        pass &= (golden == result_untilized);
        pass &= tt_metal::CloseDevice(device);
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

    TT_FATAL(pass);

    return 0;
}
