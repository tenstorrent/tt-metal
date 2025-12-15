// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_COL)
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/softplus.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "tools/profiler/kernel_profiler.hpp"
// #include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

void matmul_blocks(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& matmul_out_cb,
    const uint32_t& reduce_out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& transpose) {
    DeviceZoneScopedN("matmul_blocks");
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
    mm_block_init_short(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    // reconfig_data_format(in1_cb, in0_cb);
    // pack_reconfig_data_format(matmul_out_cb);
    cb_reserve_back(matmul_out_cb, output_num_tiles);
    // Reserve space for reduced outputs: one tile per output column (final result)
    uint32_t total_reduce_tiles = in1_num_subblocks * subblock_w;  // = N
    cb_reserve_back(reduce_out_cb, total_reduce_tiles);

    sfpu_reduce_max_sdpa_init();

    // Width-first traversal: iterate column subblocks outer, row subblocks inner
    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
        // Initialize reduce once per column (before processing any row blocks)
        // sfpu_reduce_max_load_initial_values();
        sfpu_reduce_max_prologue();

        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_subblock * subblock_h * in0_block_w;
            uint32_t in1_index = in1_subblock * subblock_w;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }

            tile_regs_commit();

            tile_regs_wait();

            uint32_t pack_dst_idx = dst_index;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = (r + subblock_h * in0_subblock) * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(pack_dst_idx, matmul_out_cb, out_row_offset + out_col_offset + c);
                    pack_dst_idx++;
                }
            }

            sfpu_reduce_max_sdpa(dst_index, subblock_h, (int)VectorMode::RC_custom);

            // Only finalize and pack after processing all row blocks for this column
            if (in0_subblock == (in0_num_subblocks - 1)) {
                sfpu_reduce_max_col_epilogue();
                // Epilogue always writes result to dst[0], so pack from dst[0]
                pack_tile<true>(0, reduce_out_cb, out_col_offset++);
                pack_tile<true>(1, reduce_out_cb, out_col_offset);
            }

            tile_regs_release();
        }
    }
    cb_push_back(matmul_out_cb, output_num_tiles);
    cb_push_back(reduce_out_cb, total_reduce_tiles);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t k_in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t q_in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t prev_max_cb = get_compile_time_arg_val(2);
    constexpr uint32_t cb_matmul_out = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t identity_scale_cb = get_compile_time_arg_val(5);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(6);
    constexpr uint32_t q_chunk_size = get_compile_time_arg_val(7);
    constexpr uint32_t head_dim = get_compile_time_arg_val(8);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(10);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(11);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(12);
    constexpr uint32_t do_eltwise = get_compile_time_arg_val(13);

    mm_init(k_in_cb, q_in_cb, cb_matmul_out);

    matmul_blocks(
        k_in_cb,
        q_in_cb,
        cb_matmul_out,
        reduce_out_cb,
        k_chunk_size,
        q_chunk_size,
        head_dim,
        in0_num_subblocks,
        in1_num_subblocks,
        head_dim,
        subblock_h,
        subblock_w,
        true);

    // Ensure outputs are produced before exiting
    cb_wait_front(cb_matmul_out, k_chunk_size * q_chunk_size);
    cb_wait_front(reduce_out_cb, q_chunk_size);  // Expect q_chunk_size tiles from on-the-fly reduction

    if (do_eltwise) {
        copy_tile_to_dst_init_short(prev_max_cb);
        for (uint32_t i = 0; i < q_chunk_size; i++) {
            tile_regs_acquire();
            copy_tile(reduce_out_cb, 0, 0);
            copy_tile(prev_max_cb, i, 1);
            max_tile(0, 1, static_cast<int>(VectorMode::R));
            tile_regs_commit();

            cb_pop_front(reduce_out_cb, 1);
            cb_reserve_back(reduce_out_cb, 1);
            tile_regs_wait();
            pack_tile(0, reduce_out_cb);
            tile_regs_release();
            cb_push_back(reduce_out_cb, 1);
        }
        cb_wait_front(reduce_out_cb, q_chunk_size);
    }
}
}  // namespace NAMESPACE
