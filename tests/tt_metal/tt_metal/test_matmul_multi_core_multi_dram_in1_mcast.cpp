// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"
#include "hostdevcommon/common_values.hpp"
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

std::tuple<tt_metal::ScopedProgramHandle, tt_metal::KernelHandle, tt_metal::KernelHandle, tt_metal::KernelHandle> create_program(
    tt_metal::Device *device,
    int start_core_x,
    int start_core_y,
    int num_cores_r,
    int num_cores_c,
    int M, int N, int K,
    int in0_block_w,
    int out_subblock_h,
    int out_subblock_w,
    int per_core_M, int per_core_N) {

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

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    CoreRange mcast_senders(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});
    CoreRange mcast_receivers(
        {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
        {ouput_cb_index, tt::DataFormat::Float16_b},
        {interm0_cb_index, tt::DataFormat::Float16_b}
    };

    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            CoreCoord core = {(std::size_t) start_core_x + j, (std::size_t) start_core_y + i};

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

    auto mm_reader_kernel_sender = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in1_mcast_sender.cpp",
        mcast_senders,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto mm_reader_kernel_receiver = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in1_mcast_receiver.cpp",
        mcast_receivers,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_matmul_tile_layout.cpp",
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

    return {std::move(program), mm_reader_kernel_sender, mm_reader_kernel_receiver, unary_writer_kernel};
}

bool write_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::ProgramHandle program,
    int start_core_x,
    int start_core_y,
    int num_cores_r,
    int num_cores_c,
    tt_metal::KernelHandle mm_reader_kernel_sender,
    tt_metal::KernelHandle mm_reader_kernel_receiver,
    tt_metal::KernelHandle unary_writer_kernel,
    int M,
    int N,
    int K,
    int in0_block_w,
    int out_subblock_h,
    int out_subblock_w,
    int per_core_M,
    int per_core_N,
    uint32_t in0_dram_addr,
    uint32_t in1_dram_addr,
    uint32_t out_dram_addr,
    uint32_t in1_mcast_sender_semaphore_addr,
    uint32_t in1_mcast_receiver_semaphore_addr) {

    bool pass = true;
    uint32_t single_tile_size = 2 * 1024;

    uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    TT_FATAL(in0_dram_addr + dram_buffer_size_act < 1024 * 1024 * 1024, "Error");
    TT_FATAL(in1_dram_addr + dram_buffer_size_weights < 1024 * 1024 * 1024, "Error");
    TT_FATAL(out_dram_addr + dram_buffer_size_out < 1024 * 1024 * 1024, "Error");

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};
            log_info(LogTest, "Runtime kernel args for core {}, {}", core.x, core.y);
            CoreCoord mcast_sender = {core.x, (std::size_t) start_core_y};
            CoreCoord core_start = {core.x, (std::size_t) start_core_y + 1};
            CoreCoord core_end = {core.x, (std::size_t) start_core_y + (num_cores_r - 1)};
            auto mcast_sender_physical = device->worker_core_from_logical_core(mcast_sender);
            auto core_start_physical = device->worker_core_from_logical_core(core_start);
            auto core_end_physical = device->worker_core_from_logical_core(core_end);

            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t) in0_dram_addr, // in0_tensor_addr
                (std::uint32_t)  K * per_core_M * core_idx_y, // in0_tensor_start_tile_id
                (std::uint32_t)  1, // in0_tensor_stride_w
                (std::uint32_t)  K, // in0_tensor_stride_h
                (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride

                (std::uint32_t)  in0_block_w, // in0_block_w
                (std::uint32_t)  per_core_M, // in0_block_h
                (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles

                (std::uint32_t)  in1_dram_addr, // in1_tensor_addr
                (std::uint32_t)  per_core_N * core_idx_x, //in1_tensor_start_tile_id
                (std::uint32_t)  1, // in1_tensor_stride_w
                (std::uint32_t)  N, // in1_tensor_stride_h
                (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride

                (std::uint32_t)  per_core_N, // in1_block_w
                (std::uint32_t)  in0_block_w, //in1_block_h
                (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

                (std::uint32_t)  K / in0_block_w, // num_blocks

                (std::uint32_t)  core_end_physical.x, // in1_mcast_dest_noc_start_x
                (std::uint32_t)  core_end_physical.y, // in1_mcast_dest_noc_start_y
                (std::uint32_t)  core_start_physical.x, // in1_mcast_dest_noc_end_x
                (std::uint32_t)  core_start_physical.y, // in1_mcast_dest_noc_end_y
                (std::uint32_t)  num_cores_r - 1, // in1_mcast_num_dests
                (std::uint32_t)  mcast_sender_physical.x, //in1_mcast_sender_noc_x
                (std::uint32_t)  mcast_sender_physical.y, //in1_mcast_sender_noc_y
                (std::uint32_t)  in1_mcast_sender_semaphore_addr,
                (std::uint32_t)  in1_mcast_receiver_semaphore_addr
            };
            std::vector<uint32_t> writer_args = {
                (std::uint32_t) out_dram_addr, // out_tensor_addr
                (std::uint32_t) core_idx_x * per_core_N + core_idx_y * per_core_M * N, // out_tensor_start_tile_id
                (std::uint32_t) 1, // out_tensor_stride_w
                (std::uint32_t) N,  // out_tensor_stride_h
                (std::uint32_t) out_subblock_w, // out_tensor_next_subblock_stride_w
                (std::uint32_t) out_subblock_h * N, // out_tensor_next_subblock_stride_h

                (std::uint32_t) out_subblock_w, // out_subblock_w
                (std::uint32_t) out_subblock_h, // out_subblock_h
                (std::uint32_t) (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
                (std::uint32_t) (per_core_N / out_subblock_w), // out_num_subblocks_w
                (std::uint32_t) (per_core_M / out_subblock_h), // out_num_subblocks_h
            };

            if(core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_sender, core, mm_reader_args);
            } else {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_receiver, core, mm_reader_args);
            }
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
            uint32_t dram_addr = (tile_id / device->num_dram_channels()) * tile_size_bytes + dram_buffer_addr;
            int dram_channel = tile_id % device->num_dram_channels();

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
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);
        int start_core_x = 0;
        int start_core_y = 0;
        int num_cores_r = device->compute_with_storage_grid_size().y;
        int num_cores_c = device->compute_with_storage_grid_size().x;
        uint32_t M = 16 * num_cores_r;
        uint32_t K = 16 * 12;
        uint32_t N = 16 * num_cores_c;
        int out_subblock_h = 4;
        int out_subblock_w = 2;
        int in0_block_w = 2;
        int per_core_M = M / num_cores_r;
        int per_core_N = N / num_cores_c;
        uint32_t single_tile_size = 2 * 1024;
        uint32_t in0_dram_addr = device->get_base_allocator_addr(HalMemType::DRAM);
        uint32_t in1_dram_addr = 400 * 1024 * 1024;
        uint32_t out_dram_addr = 800 * 1024 * 1024;
        uint32_t in1_mcast_sender_semaphore_addr = 109600;
        uint32_t in1_mcast_receiver_semaphore_addr = 109632;

        log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
        log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
        log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
        log_info(LogTest, "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}", per_core_M, in0_block_w, K / in0_block_w, per_core_M / out_subblock_h);
        log_info(LogTest, "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}", in0_block_w, per_core_N, K / in0_block_w, per_core_N / out_subblock_w);
        SHAPE shape = {1, 1, M * 32, K * 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 identity
        auto golden = select_columns(tensor.get_values(), M, K, N);


        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [program, mm_reader_kernel_sender, mm_reader_kernel_receiver, unary_writer_kernel]  = create_program(device, start_core_x, start_core_y, num_cores_r, num_cores_c, M, N, K, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N);


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
        pass &= move_tiles_to_dram(device, activations, M, K, in0_dram_addr);

        auto identity_tilized = tilize(identity, K * 32, N * 32);
        auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        pass &= move_tiles_to_dram(device, weights, K, N, in1_dram_addr);
        log_info(LogTest, "Copying inputs to dram complete");

        for(int i = 0; i < num_cores_r; i++) {
            for(int j = 0; j < num_cores_c; j++) {
                std::vector<uint32_t> invalid = {INVALID};
                CoreCoord core = {(std::size_t) start_core_x + j, (std::size_t) start_core_y + i};
                tt_metal::detail::WriteToDeviceL1(device, core, in1_mcast_sender_semaphore_addr, invalid);
            }
        }

        log_info(LogTest, "Writing kernel runtime args to device");
        pass &= write_runtime_args_to_device(
            device,
            program,
            start_core_x, start_core_y,
            num_cores_r, num_cores_c,
            mm_reader_kernel_sender, mm_reader_kernel_receiver, unary_writer_kernel,
            M, N, K,
            in0_block_w,
            out_subblock_h, out_subblock_w,
            per_core_M, per_core_N,
            in0_dram_addr, in1_dram_addr, out_dram_addr,
            in1_mcast_sender_semaphore_addr, in1_mcast_receiver_semaphore_addr
        );
        log_info(LogTest, "Writing kernel runtime args to device complete");

        log_info(LogTest, "Running Matmul {} core test", num_cores_r * num_cores_c);

        auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
        tt_metal::detail::LaunchProgram(device, *program_ptr);
        log_info(LogTest, "Matmul test done");

        log_info(LogTest, "Gathering data back from dram and checking against golden");

        for(int i = 0; i < M; i++) {
            auto row = get_row_slice(golden, M, i, M * 32, N * 32);
            for(int j = 0; j < N; j++) {
                auto golden_tile = get_col_slice(row, N, j, 32, N * 32);
                int tile_id = i * N + j;
                int dram_bank = tile_id % device->num_dram_channels();
                uint32_t dram_address = ((tile_id / device->num_dram_channels()) * single_tile_size) + out_dram_addr;
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
