// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <bmm_op.hpp>

#include <cstdint>
#include <vector>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

void golden_matmul(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t /*B*/) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float c_f = 0;
            for (uint32_t k_m = 0; k_m < K; k_m++) {
                c_f += static_cast<float>(a[i * K + k_m]) * static_cast<float>(b[k_m * N + j]);
            }
            output.at(j + i * N) = bfloat16(c_f);
        }
    }
}

void matmul_multicore_reuse(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t B,
    DeviceContext& ctx) {
    /*
     * The EZ API's DeviceContext and ProgramBuilder handle device setup, command queue,
     * and program management. We'll distribute work across multiple cores using the device's compute grid.
     */

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    auto compute_with_storage_grid_size = ctx.device().compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    /*
     * Extracting Matrix dimensions from input/output vectors
     */
    // C = A*B
    // MN = MK*KN
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])2
    uint32_t in0_block_w = 2;

    // Get large matmul params
    auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

    fmt::print(" -- Metalium Core Sizing --\n");
    fmt::print(
        " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --\n",
        per_core_M,
        per_core_N,
        out_subblock_h,
        out_subblock_w);

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    // Compute kernel compile time args
    uint32_t num_blocks = (Kt / in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B                        // batch
    };

    /*
     * Multi-Core prep
     */
    uint32_t num_blocks_y = Mt / per_core_M;
    uint32_t num_blocks_x = Nt / per_core_N;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
    CoreRangeSet all_cores(
        tt::tt_metal::num_cores_to_corerangeset(num_blocks_x * num_blocks_y, compute_with_storage_grid_size, true));

    //////////////////////////////////////////////////
    /*
     * Create DRAM buffers for input and output matrices.
     * We'll upload input vectors into these buffers prior to launching the program.
     */
    auto src0_dram_buffer = ctx.dram_tile_buffer(Mt * Kt);
    auto src1_dram_buffer = ctx.dram_tile_buffer(Kt * Nt);
    auto dst_dram_buffer = ctx.dram_tile_buffer(Mt * Nt);

    /*
     * Config of Circular Buffer in the device L1
     * input tiles count is = 2 because it's single tile process, and double-buffer
     */
    auto builder = ProgramBuilder(all_cores);
    builder.cb(tt::CBIndex::c_0, in0_CB_tiles)
        .cb(tt::CBIndex::c_1, in1_CB_tiles);

    // Output and intermediate accumulation circular buffers (c_16 and c_24) share the same
    // L1 memory. This is a hardware optimization: during computation, results are accumulated
    // in the intermediate buffer (c_24), and the final output is read from c_16. Because they
    // alias the same memory, no explicit copy is needed between pipeline stages.
    uint32_t output_cb_index = CBIndex::c_16;
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
    builder.cb(CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                   .set_page_size(output_cb_index, single_tile_size)
                   .set_page_size(interm0_cb_index, single_tile_size));

    /*
     * Create Kernels (Reader, Writer, Compute)
     * The EZ API auto-generates TensorAccessorArgs from the buffer lists.
     */
    // Create reader and writer kernels per core
    auto& reader_ref = builder.reader(
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout.cpp",
        {src0_dram_buffer, src1_dram_buffer});

    auto& writer_ref = builder.writer(
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
        {dst_dram_buffer});

    // Create compute kernel
    builder.compute(
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp",
        math_fidelity,
        compute_kernel_args);

    /*
     * Kernels - Runtime arguments
     */
    uint32_t num_blocks_read = 0;
    for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
        for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
            int core_idx_x = num_blocks_read % num_cores_x;
            int core_idx_y = num_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            // Write runtime args to device
            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
                (std::uint32_t)Kt * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
                (std::uint32_t)1,                               // in0_tensor_stride_w
                (std::uint32_t)Kt,                              // in0_tensor_stride_h
                (std::uint32_t)in0_block_w,                     // in0_tensor_next_block_stride

                (std::uint32_t)in0_block_w,               // in0_block_w
                (std::uint32_t)per_core_M,                // in0_block_h
                (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

                (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
                (std::uint32_t)per_core_N * output_idx_x,    // in1_tensor_start_tile_id
                (std::uint32_t)1,                            // in1_tensor_stride_w
                (std::uint32_t)Nt,                           // in1_tensor_stride_h
                (std::uint32_t)in0_block_w * Nt,             // in1_tensor_next_block_stride

                (std::uint32_t)per_core_N,                // in1_block_w
                (std::uint32_t)in0_block_w,               // in1_block_h
                (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

                (std::uint32_t)Kt / in0_block_w,  // num_blocks

                (std::uint32_t)Mt * Kt,     // MtKt
                (std::uint32_t)Kt * Nt,     // KtNt
                (std::uint32_t)B,           // batch
                (std::uint32_t)bcast_batch  // bcast_B
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t)dst_dram_buffer->address(),  // out_buffer_addr
                ((std::uint32_t)output_idx_x * per_core_N) +
                    (output_idx_y * per_core_M * Nt),  // out_tensor_start_tile_id
                (std::uint32_t)1,                      // out_tensor_stride_w
                (std::uint32_t)Nt,                     // out_tensor_stride_h
                (std::uint32_t)out_subblock_w,         // out_tensor_next_subblock_stride_w
                (std::uint32_t)out_subblock_h * Nt,    // out_tensor_next_subblock_stride_h

                (std::uint32_t)out_subblock_w,                     // out_subblock_w
                (std::uint32_t)out_subblock_h,                     // out_subblock_h
                (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
                (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
                (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

                (std::uint32_t)Mt * Nt,  // MtNt
                (std::uint32_t)B         // batch
            };

            reader_ref.runtime_args_at(core, mm_reader_args);
            writer_ref.runtime_args_at(core, writer_args);

            num_blocks_read++;
        }
    }

    /* Launch program & read back results */
    ctx.write(src0_dram_buffer, a);
    ctx.write(src1_dram_buffer, b);
    ctx.run(builder.build());
    output = ctx.read<bfloat16>(dst_dram_buffer);
}

///////////////////////////////////////

int main() {
    bool pass = true;

    try {
        /* Silicon accelerator setup */
        // DeviceContext wraps MeshDevice creation, command queue, and teardown in RAII.
        constexpr int device_id = 0;
        DeviceContext ctx(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Matmul Parameters Setup
        ////////////////////////////////////////////////////////////////////////////
        // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
        // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])

        /* Create source data */
        constexpr uint32_t M = 640;  // user-defined
        constexpr uint32_t N = 640;  // user-defined
        constexpr uint32_t K = 640;  // user-defined
        constexpr uint32_t B = 1;    // user-defined

        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;  // num_tiles of FP16_B

        /* input vectors */
        std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_native(single_tile_size * Mt * Kt, 1, 123, -0.4);
        std::vector<bfloat16> src1_vec = create_random_vector_of_bfloat16_native(single_tile_size * Nt * Kt, 1, 12522, -0.3);

        /* Golden Matmul running on CPU (Float)*/
        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K, B);

        /* Input vector tilizing */
        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        /* Calling the MatMul host program. Read in result into a host vector */
        std::vector<bfloat16> result_vec(dram_buffer_C_size / sizeof(bfloat16));
        matmul_multicore_reuse(src0_vec, src1_vec, result_vec, false, M, N, K, B, ctx);
        result_vec = untilize_nfaces(result_vec, M, N);

        fmt::print("Output vector of size {}\n", result_vec.size());

        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        TT_FATAL(pearson > 0.99, "PCC not high enough. Result PCC: {}, Expected PCC: 0.99", pearson);

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
