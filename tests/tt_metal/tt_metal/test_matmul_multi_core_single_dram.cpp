// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"
#include "tt_metal/impl/program/program_pool.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

// Given a tensor that is row-major datums, make it tilized
// so that its row major within a tile, and each tile's data
// is contiguous
template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    TT_FATAL(rows % 32 == 0, "Error");
    TT_FATAL(cols % 32 == 0, "Error");
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
    TT_FATAL(rows % 32 == 0, "Error");
    TT_FATAL(cols % 32 == 0, "Error");
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

std::vector<bfloat16> select_columns(std::vector<bfloat16> data, int M, int K, int N) {
    if(N == K) {
        return data;
    }
    std::vector<bfloat16> result;
    if(N > K) {
        for(int i = 0; i < M * 32; i++) {
            for(int j = 0; j < K * 32; j++) {
                int offset = i * K * 32;
                result.push_back(data.at(offset + j));
            }
            for(int j = 0; j < (N - K) * 32; j++) {
                result.push_back((float)0);
            }
        }
    } else {
        for(int i = 0; i < M * 32; i++) {
            for(int j = 0; j < N * 32; j++) {
                int offset = i * K * 32;
                result.push_back(data.at(offset + j));
            }
        }
    }

    return result;
}

std::tuple<tt_metal::ScopedProgramHandle, tt_metal::KernelHandle , tt_metal::KernelHandle> create_program(tt_metal::Device *device, int num_cores_r, int num_cores_c, int per_core_M, int per_core_N, int K, int in0_block_w, int out_subblock_h, int out_subblock_w) {
    auto program = tt_metal::CreateScopedProgram();
    uint32_t single_tile_size = 2 * 1024;
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;
    TT_FATAL(in0_CB_size <= 130*1024, "Error");
    TT_FATAL(in1_CB_size <= 130*1024, "Error");
    TT_FATAL(out_CB_size <= 540*1024, "Error");

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};
    CoreRange all_cores(start_core, end_core);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
        {ouput_cb_index, tt::DataFormat::Float16_b},
        {interm0_cb_index, tt::DataFormat::Float16_b}
    };

    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            int core_index = i * num_cores_c + j;
            CoreCoord core = {(std::size_t) j, (std::size_t) i};

            uint32_t src0_cb_index = 0;
            uint32_t cb0_tiles = in0_block_tiles * 2; // double buffer
            tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
            auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t src1_cb_index = 1;
            uint32_t cb1_tiles = in1_block_tiles * 2; // double buffer
            tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
            auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

            CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});
            tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(out_CB_size, partials_and_out_data_format_spec)
                .set_page_size(ouput_cb_index, single_tile_size)
                .set_page_size(interm0_cb_index, single_tile_size);
            auto cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_output_config);
        }
    }

    auto mm_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    int num_blocks = (K/in0_block_w);

    int in0_num_subblocks = (per_core_M/out_subblock_h);
    int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    int in1_num_subblocks = (per_core_N/out_subblock_w);
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

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    return {program, mm_reader_kernel, unary_writer_kernel};
}

std::vector<bfloat16> get_row_slice(std::vector<bfloat16> data, int total_row_slices, int row_slice_index, int rows, int cols) {
    std::vector<bfloat16> result;
    int rows_per_slice = rows / total_row_slices;
    for(int i = rows_per_slice * row_slice_index * cols; i < rows_per_slice * (row_slice_index + 1) * cols; i++) {
        result.push_back(data.at(i));
    }
    return result;
}

std::vector<bfloat16> get_col_slice(std::vector<bfloat16> data, int total_col_slices, int col_slice_index, int rows, int cols) {
    std::vector<bfloat16> result;
    int cols_per_slice = cols / total_col_slices;
    for(int r = 0; r < rows; r++) {
        for(int c = cols_per_slice * col_slice_index; c < cols_per_slice * (col_slice_index + 1); c++) {
            result.push_back(data.at(r * cols + c));
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

        CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        int num_cores_r = compute_with_storage_grid_size.y;
        int num_cores_c = compute_with_storage_grid_size.x;
        log_info(LogTest, "Num cores r = {}, Num cores c = {}", num_cores_r, num_cores_c);
        uint32_t M = 16 * num_cores_r;
        uint32_t K = 16 * 12;
        uint32_t N = 16 * num_cores_c;
        int out_subblock_h = 4;
        int out_subblock_w = 2;
        int in0_block_w = 2;
        int per_core_M = M / num_cores_r;
        int per_core_N = N / num_cores_c;
        uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_unreserved_base = device->get_base_allocator_addr(HalMemType::DRAM);
        log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
        log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
        log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
        log_info(LogTest, "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}", out_subblock_h, in0_block_w, K / in0_block_w, M / out_subblock_h);
        log_info(LogTest, "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}", out_subblock_w, in0_block_w, K / in0_block_w, N / out_subblock_w);
        SHAPE shape = {1, 1, M * 32, K * 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 identity
        auto golden = select_columns(tensor.get_values(), M, K, N);


        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [program, mm_reader_kernel, unary_writer_kernel] = create_program(device, num_cores_r, num_cores_c, per_core_M, per_core_N, K, in0_block_w, out_subblock_h, out_subblock_w);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        log_info(LogTest, "Slicing input tensors and copying them to dram along with sending runtime args to device");
        for(int i = 0; i < num_cores_r; i++) {
            std::vector<bfloat16> activation_slice = get_row_slice(tensor.get_values(), num_cores_r, i, M * 32, K * 32);
            for(int j = 0; j < num_cores_c; j++) {
                std::vector<bfloat16> weights_slice = get_col_slice(identity, num_cores_c, j, K * 32, N * 32);
                int core_index = i * num_cores_c + j;
                CoreCoord core = {(std::size_t) j, (std::size_t) i};

                uint32_t dram_buffer_src0_addr = (  core_index * per_core_M * K * single_tile_size) + dram_unreserved_base;
                int dram_src0_channel_id = 0;
                uint32_t dram_buffer_src1_addr = (core_index * K * per_core_N * single_tile_size) + dram_unreserved_base;
                int dram_src1_channel_id = 1;
                uint32_t dram_buffer_dst_addr = (core_index * per_core_M * per_core_N * single_tile_size) + dram_unreserved_base;
                int dram_dst_channel_id = 2;

                uint32_t dram_buffer_size_act = single_tile_size * per_core_M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
                uint32_t dram_buffer_size_weights = single_tile_size * K * per_core_N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
                uint32_t dram_buffer_size_out = single_tile_size * per_core_M * per_core_N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

                TT_FATAL(dram_buffer_src0_addr + dram_buffer_size_act < 1024 * 1024 * 1024, "Error");
                TT_FATAL(dram_buffer_src1_addr + dram_buffer_size_weights < 1024 * 1024 * 1024, "Error");
                TT_FATAL(dram_buffer_dst_addr + dram_buffer_size_out < 1024 * 1024 * 1024, "Error");

                auto dram_src0_noc_xy = device->dram_core_from_dram_channel(dram_src0_channel_id);
                auto dram_src1_noc_xy = device->dram_core_from_dram_channel(dram_src1_channel_id);
                auto dram_dst_noc_xy = device->dram_core_from_dram_channel(dram_dst_channel_id);

                auto activations_tilized = tilize(activation_slice, per_core_M * 32, K * 32);
                auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
                auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
                auto activations_tile_transposed = transpose_tiles(activations, per_core_M, K, in0_block_w);
                pass &= tt_metal::detail::WriteToDeviceDRAMChannel(device, dram_src0_channel_id, dram_buffer_src0_addr, activations_tile_transposed);

                auto identity_tilized = tilize(weights_slice, K * 32, per_core_N * 32);
                auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
                auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
                pass &= tt_metal::detail::WriteToDeviceDRAMChannel(device, dram_src1_channel_id, dram_buffer_src1_addr, weights);

                std::vector<uint32_t> mm_reader_args = {
                    (std::uint32_t) dram_buffer_src0_addr,
                    (std::uint32_t) dram_src0_noc_xy.x,
                    (std::uint32_t) dram_src0_noc_xy.y,
                    (std::uint32_t) dram_buffer_src1_addr,
                    (std::uint32_t) dram_src1_noc_xy.x,
                    (std::uint32_t) dram_src1_noc_xy.y,
                    (std::uint32_t) (K/in0_block_w), // num_blocks
                    (std::uint32_t) per_core_M * in0_block_w, // input 0 block num tiles
                    (std::uint32_t) per_core_N * in0_block_w, // input 1 block num tiles
                    (std::uint32_t) per_core_M * in0_block_w * single_tile_size, // input 0 block bytes
                    (std::uint32_t) per_core_N * in0_block_w * single_tile_size};

                std::vector<uint32_t> writer_args = {
                    (std::uint32_t) dram_buffer_dst_addr,
                    (std::uint32_t) dram_dst_noc_xy.x,
                    (std::uint32_t) dram_dst_noc_xy.y,
                    (std::uint32_t) out_subblock_h, // num tiles per sub block m
                    (std::uint32_t) out_subblock_w, // num tiles per sub block n
                    (std::uint32_t) per_core_M/out_subblock_h, // num sub blocks m
                    (std::uint32_t) per_core_N/out_subblock_w, // num sub blocks n
                    (std::uint32_t) out_subblock_w * single_tile_size * (per_core_N/out_subblock_w), // bytes offset to next row within sub-block
                    (std::uint32_t) out_subblock_h * out_subblock_w * single_tile_size * (per_core_N/out_subblock_w), // bytes offset to next row of sub-blocks
                    (std::uint32_t) out_subblock_w * single_tile_size};

                tt_metal::SetRuntimeArgs(program, mm_reader_kernel, core, mm_reader_args);
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
            }
        }

        log_info(LogTest, "Copying inputs to dram and runtime args to cores complete");

        log_info(LogTest, "Running Matmul {} core test", num_cores_c * num_cores_r);

        auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
        tt_metal::detail::LaunchProgram(device, *program_ptr);

        log_info(LogTest, "Matmul test done");
        log_info(LogTest, "Gathering data back from dram and checking against golden");
        for(int i = 0; i < num_cores_r; i++) {
            auto golden_row = get_row_slice(golden, num_cores_r, i, M * 32, N * 32);
            for(int j = 0; j < num_cores_c; j++) {
                auto per_core_golden = get_col_slice(golden_row, num_cores_c, j, per_core_M * 32, N * 32);
                std::vector<uint32_t> result_vec;
                int core_index = i * num_cores_c + j;
                uint32_t dram_buffer_dst_addr = (core_index * per_core_M * per_core_N * single_tile_size) + dram_unreserved_base;
                int dram_dst_channel_id = 2;
                tt_metal::detail::ReadFromDeviceDRAMChannel(device, dram_dst_channel_id, dram_buffer_dst_addr, per_core_M * per_core_N * single_tile_size, result_vec);
                auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
                auto result_flat_layout = convert_to_flat_layout(result_bfp16);
                auto result_untilized = untilize(result_flat_layout, per_core_M*32, per_core_N*32);
                pass &= (per_core_golden == result_untilized);
            }
        }
        log_info(LogTest, "Golden check complete");
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
