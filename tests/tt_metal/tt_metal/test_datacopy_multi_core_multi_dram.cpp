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

std::tuple<tt_metal::Program, tt_metal::KernelHandle, tt_metal::KernelHandle> create_program(
    tt_metal::Device *device,
    int num_cores_r,
    int num_cores_c,
    int tensor_num_tiles,
    int block_num_tiles) {

    tt_metal::Program program = tt_metal::CreateProgram();

    int num_cores = num_cores_r * num_cores_c;

    TT_FATAL("Error: tensor's tiles don't even distributed across cores." && tensor_num_tiles % num_cores == 0, "Error");
    int num_tiles_per_core = tensor_num_tiles / num_cores;

    TT_FATAL("Error: block must fit to half-dst" && block_num_tiles <= 8, "Error"); // half-dst in GS
    TT_FATAL("Error: num tiles per core needs to be divisible by block size." && num_tiles_per_core % block_num_tiles == 0, "Error");
    int num_blocks_per_core = num_tiles_per_core / block_num_tiles;

    uint32_t single_tile_size = 2 * 1024; // bfloat16
    uint32_t in0_CB_tiles = block_num_tiles * 2; // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
    uint32_t in0_CB_index = 0;

    uint32_t out_CB_tiles = block_num_tiles * 2; // double buffer
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;
    uint32_t out_CB_index = 16;

    TT_FATAL(in0_CB_size <= 130*1024, "Error");
    TT_FATAL(out_CB_size <= 540*1024, "Error");

    CoreCoord start_core{0, 0};
    CoreCoord end_core{(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};
    CoreRange all_cores(start_core, end_core);

    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            CoreCoord core = {(std::size_t) j, (std::size_t) i};
            uint32_t l1_valid_address = 200 * 1024;

            uint32_t in0_CB_addr = l1_valid_address;
            l1_valid_address += in0_CB_size;
            tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(in0_CB_size, {{in0_CB_index, tt::DataFormat::Float16_b}}, in0_CB_addr)
                .set_page_size(in0_CB_index, single_tile_size);
            auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t out_CB_addr = l1_valid_address;
            l1_valid_address += out_CB_size;
            tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(out_CB_size, {{out_CB_index, tt::DataFormat::Float16_b}}, out_CB_addr)
                .set_page_size(out_CB_index, single_tile_size);
            auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

            TT_FATAL(l1_valid_address < 1024 * 1024, "Error");
        }
    }

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_copy_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_copy_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {
        uint(block_num_tiles),
        uint(num_blocks_per_core)
    };

    auto compute_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    return {program, reader_kernel, writer_kernel};
}

bool write_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program &program,
    int num_cores_r,
    int num_cores_c,
    tt_metal::KernelHandle reader_kernel,
    tt_metal::KernelHandle writer_kernel,
    int tensor_num_tiles,
    int block_num_tiles,
    int Ht,
    int Wt,
    uint32_t src0_dram_addr,
    uint32_t dst_dram_addr) {

    bool pass = true;
    uint32_t single_tile_size = 2 * 1024;


    // this doesn't seem to be correct
    /*
    uint32_t dram_buffer_size_act = single_tile_size * Ht * Wt;
    uint32_t dram_buffer_size_out = single_tile_size * Ht * Wt;

    uint dram_channel_size = 1024 * 1024 * 1024;

    TT_FATAL(src0_dram_addr + dram_buffer_size_act < dram_channel_size, "Error");
    TT_FATAL(dst_dram_addr + dram_buffer_size_out < dram_channel_size, "Error");
    */

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) core_idx_x, (std::size_t) core_idx_y};

            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t) src0_dram_addr, // src0_addr
                (std::uint32_t)  K * per_core_M * core_idx_y, // src0_start_tile_id
                (std::uint32_t)  block_tile_dim, // src0_block_x
                (std::uint32_t)  per_core_M, // src0_block_y
                (std::uint32_t)  1, // src0_stride_x
                (std::uint32_t)  K, // src0_stride_y
                (std::uint32_t)  block_tile_dim, // src0_block_stride
                (std::uint32_t)  block_tile_dim * per_core_M, //src0_block_num_tiles
                (std::uint32_t)  src1_dram_addr, // src1_addr
                (std::uint32_t)  per_core_N * core_idx_x, //src1_start_tile_id
                (std::uint32_t)  per_core_N, // src1_block_x
                (std::uint32_t)  block_tile_dim, //src1_block_y
                (std::uint32_t)  1, // src1_stride_x
                (std::uint32_t)  N, // src1_stride_y
                (std::uint32_t)  block_tile_dim * N, //src1_block_stride
                (std::uint32_t)  per_core_N * block_tile_dim, // src1_block_num_tiles
                (std::uint32_t)  K / block_tile_dim // num_blocks
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t) dst_dram_addr, // dst_addr
                (std::uint32_t) core_idx_x * per_core_N + core_idx_y * per_core_M * N, // dst_start_tile_id
                (std::uint32_t) num_tiles_per_sub_block_n, // dst_sub_block_x
                (std::uint32_t) num_tiles_per_sub_block_m, // dst_sub_block_y
                (std::uint32_t) (per_core_N / num_tiles_per_sub_block_n), // dst_num_sub_blocks_x
                (std::uint32_t) (per_core_M / num_tiles_per_sub_block_m), // dst_num_sub_blocks_y
                (std::uint32_t) num_tiles_per_sub_block_n, // dst_sb_stride_x
                (std::uint32_t) num_tiles_per_sub_block_m * N, // dst_sb_stride_y
                (std::uint32_t) 1, // dst_stride_x
                (std::uint32_t) N  //dst_stride_y
            };

            // log_info(LogTest, "Core = {}, {}", core.x, core.y);

            // log_info(LogTest, "Reader args");
            // log_info(LogTest, "src0_addr = {}", src0_dram_addr);
            // log_info(LogTest, "src0_start_tile_id = {}", K * per_core_M * core_idx_y);
            // log_info(LogTest, "src0_block_x = {}", block_tile_dim);
            // log_info(LogTest, "src0_block_y = {}", per_core_M);
            // log_info(LogTest, "src0_stride_x = {}", 1);
            // log_info(LogTest, "src0_stride_y = {}", K);
            // log_info(LogTest, "src0_block_stride = {}", block_tile_dim);
            // log_info(LogTest, "src0_block_num_tiles = {}", block_tile_dim * per_core_M);
            // log_info(LogTest, "src1_addr = {}", src1_dram_addr);
            // log_info(LogTest, "src1_start_tile_id = {}", per_core_N * core_idx_x);
            // log_info(LogTest, "src1_sub_block_x = {}", num_tiles_per_sub_block_n);
            // log_info(LogTest, "src1_sub_block_y = {}", block_tile_dim);
            // log_info(LogTest, "src1_num_sub_blocks = {}", (per_core_N / num_tiles_per_sub_block_n));
            // log_info(LogTest, "src1_stride_x = {}", 1);
            // log_info(LogTest, "src1_stride_y = {}", N);
            // log_info(LogTest, "src1_block_stride = {}", block_tile_dim * N);
            // log_info(LogTest, "src1_block_num_tiles = {}", per_core_N * block_tile_dim);
            // log_info(LogTest, "num_blocks = {}", K / block_tile_dim);

            // log_info(LogTest, "Writer args");
            // log_info(LogTest, "dst_addr = {}", dst_dram_addr);
            // log_info(LogTest, "dst_start_tile_id = {}", core_idx_x * per_core_N + core_idx_y * per_core_M * N);
            // log_info(LogTest, "dst_sub_block_x = {}", num_tiles_per_sub_block_n);
            // log_info(LogTest, "dst_sub_block_y = {}", num_tiles_per_sub_block_m);
            // log_info(LogTest, "dst_num_sub_blocks_x = {}", (per_core_N / num_tiles_per_sub_block_n));
            // log_info(LogTest, "dst_num_sub_blocks_y = {}", (per_core_M / num_tiles_per_sub_block_m));
            // log_info(LogTest, "dst_sb_stride_x = {}", num_tiles_per_sub_block_n);
            // log_info(LogTest, "dst_sb_stride_y = {}", num_tiles_per_sub_block_m * N);
            // log_info(LogTest, "dst_stride_x = {}", 1);
            // log_info(LogTest, "dst_stride_y = {}", N);

            tt_metal::SetRuntimeArgs(program, mm_reader_kernel, core, mm_reader_args);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
        }
    }

    return pass;
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

bool move_tiles_to_dram(tt_metal::Device *device, std::vector<uint32_t> tensor, int tiles_r, int tiles_c, uint32_t dram_buffer_addr) {
    bool pass = true;
    int tile_size = 512; // 32*32 packed into u32
    int tile_size_bytes = 32 * 32 * 2;
    int start_index = 0;
    int tile_id = 0;
    for(int i = 0; i < tiles_r; i++) {
        for(int j = 0; j < tiles_c; j++) {
            std::vector<uint32_t> tile;
            tile.insert(tile.end(), tensor.begin() + start_index, tensor.begin() + start_index + tile_size);
            uint32_t dram_addr = (tile_id / 8) * tile_size_bytes + dram_buffer_addr;
            int dram_channel = tile_id % 8;

            pass &= tt_metal::detail::WriteToDeviceDRAMChannel(device, dram_channel, dram_addr, tile);
            start_index += tile_size;
            tile_id++;
        }
    }
    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {
        int num_cores_r = 10;
        int num_cores_c = 12;
        uint32_t M = 16 * num_cores_r;
        uint32_t K = 16 * 12;
        uint32_t N = 16 * num_cores_c;
        int num_tiles_per_sub_block_m = 4;
        int num_tiles_per_sub_block_n = 2;
        int block_tile_dim = 2;
        int per_core_M = M / num_cores_r;
        int per_core_N = N / num_cores_c;
        uint32_t single_tile_size = 2 * 1024;
        uint32_t src0_dram_addr = 0;
        uint32_t src1_dram_addr = 400 * 1024 * 1024;
        uint32_t dst_dram_addr = 800 * 1024 * 1024;
        log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
        log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
        log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
        log_info(LogTest, "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}", per_core_M, block_tile_dim, K / block_tile_dim, per_core_M / num_tiles_per_sub_block_m);
        log_info(LogTest, "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}", block_tile_dim, per_core_N, K / block_tile_dim, per_core_N / num_tiles_per_sub_block_n);
        SHAPE shape = {1, 1, M * 32, K * 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 identity
        auto golden = select_columns(tensor.get_values(), M, K, N);

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [program, mm_reader_kernel, unary_writer_kernel]  = create_program(device, num_cores_r, num_cores_c, M, N, K, block_tile_dim, num_tiles_per_sub_block_m, num_tiles_per_sub_block_n, per_core_M, per_core_N);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        log_info(LogTest, "Scattering inputs (activation & weights) to dram channels using tiled layout");
        auto activations_tilized = tilize(tensor.get_values(), M * 32, K * 32);
        auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
        auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        pass &= move_tiles_to_dram(device, activations, M, K, src0_dram_addr);

        auto identity_tilized = tilize(identity, K * 32, N * 32);
        auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        pass &= move_tiles_to_dram(device, weights, K, N, src1_dram_addr);
        log_info(LogTest, "Copying inputs to dram complete");

        log_info(LogTest, "Writing kernel runtime args to device");
        pass &= write_runtime_args_to_device(
            device,
            program,
            num_cores_r, num_cores_c,
            mm_reader_kernel, unary_writer_kernel,
            M, N, K,
            block_tile_dim,
            num_tiles_per_sub_block_m, num_tiles_per_sub_block_n,
            per_core_M, per_core_N,
            src0_dram_addr, src1_dram_addr, dst_dram_addr
        );
        log_info(LogTest, "Writing kernel runtime args to device complete");

        log_info(LogTest, "Running Matmul {} core test", num_cores_r * num_cores_c);

        tt_metal::detail::LaunchProgram(device, program);

        log_info(LogTest, "Matmul test done");
        log_info(LogTest, "Gathering data back from dram and checking against golden");

        for(int i = 0; i < M; i++) {
            auto row = get_row_slice(golden, M, i, M * 32, N * 32);
            for(int j = 0; j < N; j++) {
                auto golden_tile = get_col_slice(row, N, j, 32, N * 32);
                int tile_id = i * N + j;
                int dram_bank = tile_id % 8;
                uint32_t dram_address = ((tile_id / 8) * single_tile_size) + dst_dram_addr;
                std::vector<uint32_t> result_vec;
                tt_metal::detail::ReadFromDeviceDRAMChannel(device, dram_bank, dram_address, single_tile_size, result_vec);
                auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
                auto result_flat_layout = convert_to_flat_layout(result_bfp16);

                // log_info(LogTest, "Tile id {} on dram bank {}, address {}", tile_id, dram_bank, dram_address);
                // print_vec(result_flat_layout, 32, 32, "Result - tile#" + std::to_string(tile_id));
                pass &= (golden_tile == result_flat_layout);
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
