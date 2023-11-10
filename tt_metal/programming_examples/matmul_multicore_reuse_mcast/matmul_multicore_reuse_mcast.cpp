// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include <algorithm>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;


template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    TT_ASSERT(rows % 32 == 0);
    TT_ASSERT(cols % 32 == 0);
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
    return convert_to_tile_layout(result);
}


template <typename T>
std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
    int TileWidth = 32;
    int TileHeight = 32;
    TT_ASSERT(rows % TileHeight == 0);
    TT_ASSERT(cols % TileWidth == 0);
    int elements_in_tile = TileHeight*TileWidth;
    int num_tiles_r = rows / TileHeight;
    int num_tiles_c = cols / TileWidth;
    std::vector<T> result;
    for(auto r = 0; r < num_tiles_r; r++) {
        for(auto i = 0; i < TileHeight; i++) {
            for(auto c = 0; c < num_tiles_c; c++) {
                int offset = r * elements_in_tile * num_tiles_c + c * elements_in_tile + i * TileWidth;
                for(auto j = 0; j < TileWidth; j++) {
                    result.push_back(data.at(offset + j));
                }
            }
        }
    }
    return convert_to_flat_layout(result);
}


void golden_matmul(vector<bfloat16> a, vector<bfloat16> b, vector<uint32_t>& output,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    //vector<float> c_f(M * N, 0);
    float c_f;
    float float_tmp;
    vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j+ (i * N);
            //c_f.at(idx_c) = 0;
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                //c_f.at(idx_c) += a[idx_a].to_float() * b[idx_b].to_float();
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                // uint32_t* int_tmp = (uint32_t*) &float_tmp;
                // *int_tmp &= 0xffff0000 ;
                c_f += float_tmp;
                idx_a += 1;
                idx_b += K;
            }
            c_bf.at(idx_c) = bfloat16(c_f);
            //if (idx_c < 128) {
            //    cout << "GG " << c_f << " .. " << c_bf.at(idx_c) << endl;
            //}
            //output[idx_c] = (uint32_t)c_bf.to_uint16() | ((uint32_t)c_bf.to_uint16() << 16);
        }
    }
    output = pack_bfloat16_vec_into_uint32_vec(c_bf);
}

void matmul_multicore_reuse_mcast(vector<uint32_t>& a, vector<uint32_t>& b, vector<uint32_t>& output, bool bcast_batch,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B, Device* device) {

    /*
    * Setup program to execute along with its buffers and kernels to use
    * Core range is just single core
    */
    CommandQueue& cq = *detail::GLOBAL_CQ;
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);
    //uint32_t single_tile_size = 2 * 1024;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    /*
    * EXtracting Matrix dimensions from input/output vectors
    */
    // C = A*B
    // MN = MK*KN
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])2
    uint32_t in0_block_w = 2;
    //uint32_t out_subblock_h = 4;
    //uint32_t out_subblock_w = 2;
    //uint32_t per_core_M = 16;
    //uint32_t per_core_N = 16;

    // Get large matmul params
    auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

    cout << "========= " << per_core_M << " == " << per_core_N << " == " <<  out_subblock_h << " == " << out_subblock_w << endl;

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2; // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2; // double buffer
    uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles; // No double buffer
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    // Compute kernel compile time args
    uint32_t num_blocks = (Kt/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w, // in0_block_w
        in0_num_subblocks, // in0_num_subblocks
        in0_block_num_tiles, // in0_block_num_tiles
        in0_subblock_num_tiles, // in0_subblock_num_tiles

        in1_num_subblocks, // in1_num_subblocks
        in1_block_num_tiles, // in1_block_num_tiles
        in1_per_core_w, // in1_per_core_w

        num_blocks, // num_blocks

        out_subblock_h, // out_subblock_h
        out_subblock_w, // out_subblock_w
        out_subblock_num_tiles, // out_subblock_num_tiles
        B // batch
    };


    /*
    * Multi-Core prep
    */
    uint32_t num_blocks_y = Mt / per_core_M;
    uint32_t num_blocks_x = Nt / per_core_N;
    uint32_t num_blocks_total =  num_blocks_y * num_blocks_x;
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
    CoreCoord start_core = {0, 0};
    CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);

    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange left_column{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange all_except_left_column{
        .start={(std::size_t) start_core_x + 1, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange in0_sender_in1_sender{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y}};

    CoreRange in0_sender_in1_receiver{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y + 1},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange in0_receiver_in1_sender{
        .start={(std::size_t) start_core_x + 1, (std::size_t) start_core_y},
        .end{(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y}};

    CoreRange in0_receiver_in1_receiver{
        .start={(std::size_t) start_core_x + 1, (std::size_t) start_core_y + 1},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    //////////////////////////////////////////////////
    /*
    * Create DRAM Buffers for input and output vectors
    * Writing data from input vectors to source buffers
    */

    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels


    Buffer src0_dram_buffer = CreateBuffer(device, dram_buffer_A_size, single_tile_size, BufferType::DRAM);
    Buffer src1_dram_buffer = CreateBuffer(device, dram_buffer_B_size, single_tile_size, BufferType::DRAM);
    Buffer dst_dram_buffer = CreateBuffer(device, dram_buffer_C_size, single_tile_size, BufferType::DRAM);
    uint32_t src0_addr = src0_dram_buffer.address();
    uint32_t src1_addr = src1_dram_buffer.address();
    uint32_t dst_addr = dst_dram_buffer.address();

    /*
    * Config of Circular Buffer in the device L1
    * input tiles count is = 2 because it's single tile process, and double-buffer
    */
    uint32_t src0_cb_index = CB::c_in0; //0
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1; // 1
    CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
        {output_cb_index, cb_data_format},
        {interm0_cb_index, cb_data_format}
    };
    CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
		.set_page_size(output_cb_index, single_tile_size)
        .set_page_size(interm0_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_output_config);

    ////////////////////////////
    /*
    * Compile time arguments
    */
    bool src0_is_dram = src0_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    //std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t) output_cb_index, (uint32_t)dst_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    /*
    * Create Kernels (Reader, Writer, Compute)
    */
    // Create reader and writer kernels per core group

    auto mm_reader_kernel_in0_sender_in1_sender_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
        in0_sender_in1_sender,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = reader_compile_time_args});

    auto mm_reader_kernel_in0_sender_in1_receiver_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
        in0_sender_in1_receiver,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = reader_compile_time_args});

    auto mm_reader_kernel_in0_receiver_in1_sender_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
        in0_receiver_in1_sender,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto mm_reader_kernel_in0_receiver_in1_receiver_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
        in0_receiver_in1_receiver,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto unary_writer_kernel_noc0_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
        all_except_left_column,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    auto unary_writer_kernel_noc1_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
        left_column,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = writer_compile_time_args});

    // Create compute kernel
    auto mm_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/compute/bmm_large_block_zm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args}
    );

    auto in0_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);


    /*
    * Kernels - Runtime arguments
    */
    std::vector<KernelID> reader_kernel_ids;
    std::vector<KernelID> writer_kernel_ids;
    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            CoreCoord left_core    = {(std::size_t) start_core_x, (std::size_t) core.y};
            CoreCoord left_core_plus_one    = {(std::size_t) start_core_x + 1, (std::size_t) core.y};
            CoreCoord right_core   = {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) core.y};
            CoreCoord top_core     = {(std::size_t) core.x, (std::size_t) start_core_y};
            CoreCoord top_core_plus_one     = {(std::size_t) core.x, (std::size_t) start_core_y + 1};
            CoreCoord bottom_core  = {(std::size_t) core.x, (std::size_t) start_core_y + num_cores_r - 1};

            auto left_core_physical = device->worker_core_from_logical_core(left_core);
            auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
            auto right_core_physical = device->worker_core_from_logical_core(right_core);
            auto top_core_physical = device->worker_core_from_logical_core(top_core);
            auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
            auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)  src0_dram_buffer.address(), // in0_buffer_addr
                (std::uint32_t)  Kt * per_core_M * core_idx_y, // in0_buffer_start_tile_id
                (std::uint32_t)  1, // in0_buffer_stride_w
                (std::uint32_t)  Kt, // in0_buffer_stride_h
                (std::uint32_t)  in0_block_w, // in0_buffer_next_block_stride

                (std::uint32_t)  in0_block_w, // in0_block_w
                (std::uint32_t)  per_core_M, // in0_block_h
                (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles

                (std::uint32_t)  src1_dram_buffer.address(), // in1_buffer_addr
                (std::uint32_t)  per_core_N * core_idx_x, //in1_buffer_start_tile_id
                (std::uint32_t)  1, // in1_buffer_stride_w
                (std::uint32_t)  Nt, // in1_buffer_stride_h
                (std::uint32_t)  in0_block_w * Nt, //in1_buffer_next_block_stride

                (std::uint32_t)  per_core_N, // in1_block_w
                (std::uint32_t)  in0_block_w, //in1_block_h
                (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

                (std::uint32_t)  Kt / in0_block_w, // num_blocks

                (std::uint32_t)  right_core_physical.x, // in0_mcast_dest_noc_start_x
                (std::uint32_t)  right_core_physical.y, // in0_mcast_dest_noc_start_y
                (std::uint32_t)  left_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
                (std::uint32_t)  left_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y
                (std::uint32_t)  (num_cores_c - 1), // in0_mcast_num_dests
                (std::uint32_t)  left_core_physical.x, // in0_mcast_sender_noc_x
                (std::uint32_t)  left_core_physical.y, // in0_mcast_sender_noc_y
                (std::uint32_t)  in0_mcast_sender_semaphore,
                (std::uint32_t)  in0_mcast_receiver_semaphore,

                (std::uint32_t)  bottom_core_physical.x, // in0_mcast_dest_noc_start_x
                (std::uint32_t)  bottom_core_physical.y, // in0_mcast_dest_noc_start_y
                (std::uint32_t)  top_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
                (std::uint32_t)  top_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y
                (std::uint32_t)  (num_cores_r - 1), // in0_mcast_num_dests
                (std::uint32_t)  top_core_physical.x, // in0_mcast_sender_noc_x
                (std::uint32_t)  top_core_physical.y, // in0_mcast_sender_noc_y
                (std::uint32_t)  in1_mcast_sender_semaphore,
                (std::uint32_t)  in1_mcast_receiver_semaphore,

                (std::uint32_t)  Mt * Kt, // MtKt
                (std::uint32_t)  Kt * Nt, // KtNt
                (std::uint32_t)  B, // batch
                (std::uint32_t)  bcast_batch // bcast_B
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t) dst_dram_buffer.address(), // out_buffer_addr
                (std::uint32_t) core_idx_x * per_core_N + core_idx_y * per_core_M * Nt, // out_buffer_start_tile_id
                (std::uint32_t) 1, // out_buffer_stride_w
                (std::uint32_t) Nt,  // out_buffer_stride_h
                (std::uint32_t) out_subblock_w, // out_buffer_next_subblock_stride_w
                (std::uint32_t) out_subblock_h * Nt, // out_buffer_next_subblock_stride_h

                (std::uint32_t) out_subblock_w, // out_subblock_w
                (std::uint32_t) out_subblock_h, // out_subblock_h
                (std::uint32_t) (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
                (std::uint32_t) (per_core_N / out_subblock_w), // out_num_subblocks_w
                (std::uint32_t) (per_core_M / out_subblock_h), // out_num_subblocks_h

                (std::uint32_t) Mt * Nt, // MtNt
                (std::uint32_t) B // batch
            };

            if(core_idx_x == 0 and core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_sender_id, core, mm_reader_args); // RISCV_0_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args); // RISCV_1_default
                reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_sender_id);
                writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
            } else if (core_idx_x == 0 and core_idx_y != 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_receiver_id, core, mm_reader_args); // RISCV_0_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args); // RISCV_1_default
                reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_receiver_id);
                writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
            } else if (core_idx_x != 0 and core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_sender_id, core, mm_reader_args); // RISCV_1_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args); // RISCV_0_default
                reader_kernel_ids.push_back(mm_reader_kernel_in0_receiver_in1_sender_id);
                writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
            } else {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_receiver_id, core, mm_reader_args); // RISCV_1_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args); // RISCV_0_default
                reader_kernel_ids.push_back(mm_reader_kernel_in0_receiver_in1_receiver_id);
                writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
            }

        }
    }

    /* Launch program & read in output buffer result into the host vector */

    EnqueueWriteBuffer(cq, src0_dram_buffer, a, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b, false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output, true);
}


///////////////////////////////////////



int main(int argc, char **argv) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Matmul Parameters Setup
        ////////////////////////////////////////////////////////////////////////////
        // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
        // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])

        /* Create source data */
        constexpr uint32_t M = 3200;  // user-defined
        constexpr uint32_t N = 3200;  // user-defined
        constexpr uint32_t K = 3200;  // user-defined
        constexpr uint32_t B = 1;  // user-defined

        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B

        /* input vectors */
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_A_size, 1, 123, -0.4);
        std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(dram_buffer_B_size, 1, 12522, -0.3);

        //std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(dram_buffer_A_size, false);
        //std::vector<uint32_t> src1_vec = pack_bfloat16_vec_into_uint32_vec(create_identity_matrix(K, N, K));

        //constexpr float val_to_add = 2.0f;
        //std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(dram_buffer_A_size, val_to_add);
        //std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_B_size, val_to_add);

        /* Input vector tilizing */
        std::vector<uint32_t> tilized_src0_vec = pack_bfloat16_vec_into_uint32_vec(tilize(unpack_uint32_vec_into_bfloat16_vec(src0_vec), M, K));
        std::vector<uint32_t> tilized_src1_vec = pack_bfloat16_vec_into_uint32_vec(tilize(unpack_uint32_vec_into_bfloat16_vec(src1_vec), K, N));

        cout << "-input size- " << src0_vec.size() << " -- " << src1_vec.size() << endl;


        cout << "----orig input 0--" << endl;
        for (int i = 0; i < src0_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(src0_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();

            if (i % 1280 == 0){
                //cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << src0_vec.at(i) << endl;
            }
        }
        /*
        cout << "----orig input 1--" << endl;
        for (int i = 0; i < 512; i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(src1_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << src1_vec.at(i) << endl;
        }
        */

        /*
        cout << "----tiled input--" << endl;
        for (int i = 0; i < src0_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(tilized_src0_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            if (i % 1280 == 0){
                cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << tilized_src0_vec.at(i) << endl;
            }
        }
        */

        /* Calling the MatMul host program. Read in result into a host vector */
        vector<uint32_t> result_vec;
        matmul_multicore_reuse_mcast(tilized_src0_vec, tilized_src1_vec, result_vec, false, M, N, K, B, device);

        /*
        cout << "----metal--" << endl;
        cout << result_vec.size() << endl;
        for (int i = 0; i < result_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(result_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            if (i % 1280 == 0){
                cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << result_vec.at(i) << endl;
            }
        }
        */


        vector<uint32_t> result_vec_untilized = pack_bfloat16_vec_into_uint32_vec(untilize(unpack_uint32_vec_into_bfloat16_vec(result_vec), M, N));
        cout << "----metal_untilized--" << endl;
        cout << result_vec.size() << endl;
        for (int i = 0; i < result_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(result_vec_untilized.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            if (i % 1280 == 0){
                //cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << result_vec_untilized.at(i) << endl;
            }
        }

        /* Golden Matmul running on CPU (Float)*/
        vector<uint32_t> golden_vec;
        golden_matmul(unpack_uint32_vec_into_bfloat16_vec(src0_vec), unpack_uint32_vec_into_bfloat16_vec(src1_vec), golden_vec, M, N, K, B);
        vector<uint32_t>  golden_vec_tilized = pack_bfloat16_vec_into_uint32_vec(tilize(unpack_uint32_vec_into_bfloat16_vec(golden_vec), M, N));

        cout << "----golden--" << endl;
        cout << golden_vec.size() << endl;
        for (int i = 0; i < golden_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(golden_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            if (i % 1280 == 0){
                //cout << "-- " << i << " -- " << a1 << "  " << a2 << "---" << golden_vec.at(i) << endl;
            }
        }

        /* Comparison: Golden vs. METAL Matmul*/
        constexpr float abs_tolerance = 0.01f;
        constexpr float rel_tolerance = 0.001f;
        std::function<bool(const float, const float)> comparison_function = [](const float a, const float b) {
            return is_close(a, b, rel_tolerance, abs_tolerance);
        };

        float pearson = packed_uint32_t_vector_pcc_v2(golden_vec_tilized, result_vec);
        cout << "PCC= " << pearson << endl;

        TT_FATAL(pearson > 0.98, "PCC not high");

        //pass &= packed_uint32_t_vector_comparison(golden_vec, result_vec_untilized, comparison_function);
        //pass &= packed_uint32_t_vector_comparison(golden_vec_tilized, result_vec, comparison_function);

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
