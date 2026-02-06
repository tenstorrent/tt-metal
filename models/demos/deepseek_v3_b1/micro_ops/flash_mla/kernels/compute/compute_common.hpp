// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "tools/profiler/kernel_profiler.hpp"

/******************************************************************************
 *                                                                             *
 *                   Common Functions for Compute Kernels                      *
 *                                                                             *
 ******************************************************************************/

/**
 * in0_cb += in1_cb
 */
template <bool pop_in1>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
    if (pop_in1) {
        cb_pop_front(in1_cb, num_tiles);
    }
}

/**
 * in_cb -> out_cb
 */
template <bool pop_in_cb>
void move_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced

    copy_tile_to_dst_init_short(in_cb);

    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0 /*dst*/);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
    if (pop_in_cb) {
        cb_pop_front(in_cb, num_tiles);
    }
}

/**
 * out_cb = in0_cb @ in1_cb
 *
 * @param transpose: If false, in1 is expected in [K, N] layout (row-major), no tile transpose.
 *                   If true, in1 is in [N, K] layout and both layout indexing and tile-level
 *                   transpose are applied.
 * @param skip_in1_pop: If true, skip cb_pop_front on in1_cb (caller manages CB lifecycle)
 */
ALWI void cb_matmul_blocks(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& num_blocks,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& transpose,
    const bool& add_mask,
    const uint32_t& mask_cb,
    const uint32_t& zero_cb,
    const bool& skip_in1_pop = false) {
    // precondition: in0_cb has M*K produced
    // precondition: in1_cb has K*N produced (or N*K if transpose)
    // postcondition: in0_cb is full, in1_cb is empty (unless skip_in1_pop=true)
    // postcondition: out_cb has M*N produced

    // For [K, N] layout (transpose=false): in1 tiles indexed as (k, n) -> k * N + n
    //   - Inner loop (over K): stride by N to next K row
    //   - Subblock offset (over N): stride by subblock_w
    //   - MOP iterates over subblock_w with stride 1, which matches [K, N] layout
    // For [N, K] layout (transpose=true): in1 tiles indexed as (n, k) -> n * K + k
    //   - Inner loop (over K): stride by 1 to next K column
    //   - Column offset (over N): stride by K to next N row
    //   - MOP assumes stride 1, so we must call matmul_block per output column
    const uint32_t in1_inner_stride = transpose ? 1 : N;
    const uint32_t in1_subblock_stride = transpose ? subblock_w * K : subblock_w;
    // When transpose=true, MOP's internal stride=1 is wrong for [N,K] layout.
    // We call matmul_block with effective_subblock_w=1 and loop over columns explicitly.
    const uint32_t effective_subblock_w = transpose ? 1 : subblock_w;
    const uint32_t num_col_loops = transpose ? subblock_w : 1;
    const uint32_t in1_col_stride = K;  // Stride between output columns in [N, K] layout

    mm_block_init_short(
        in0_cb,
        in1_cb,
        transpose /*transpose*/,
        effective_subblock_w /*ct_dim*/,
        subblock_h /*rt_dim*/,
        in0_block_w /*kt_dim*/);

    reconfig_data_format(in1_cb, in0_cb);
    {
        DeviceZoneScopedN("matmul-wait-in1");
        cb_wait_front(in1_cb, K * N);
    }

    uint32_t output_num_tiles = M * N;
    cb_reserve_back(out_cb, output_num_tiles);
    uint32_t in0_index_offset = 0;

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        uint32_t in1_index_offset = 0;

        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            // When transpose=true, we process one output column at a time
            for (uint32_t col_loop = 0; col_loop < num_col_loops; ++col_loop) {
                tile_regs_acquire();
                uint32_t dst_index = 0;
                uint32_t in0_index = in0_index_offset;
                // For transpose: offset by col_loop * K to get to correct N row
                uint32_t in1_index = in1_index_offset + (transpose ? col_loop * in1_col_stride : 0);

                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                    matmul_block(
                        in0_cb,
                        in1_cb,
                        in0_index,
                        in1_index,
                        dst_index,
                        transpose,
                        effective_subblock_w,
                        subblock_h,
                        in0_block_w);
                    in0_index++;
                    in1_index += in1_inner_stride;
                }
                uint32_t col_out_tiles = subblock_h * effective_subblock_w;
                if (add_mask) {
                    cb_wait_front(mask_cb, col_out_tiles);
                    cb_wait_front(zero_cb, 1);
                    add_tiles_init(zero_cb, mask_cb, true);
                    for (uint32_t i = 0; i < col_out_tiles; i++) {
                        add_tiles(zero_cb, mask_cb, 0, i, i);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(out_cb, col_out_tiles);
                for (uint32_t i = 0; i < col_out_tiles; i++) {
                    pack_tile(i, out_cb);
                }
                cb_push_back(out_cb, col_out_tiles);
                tile_regs_release();
            }
            in1_index_offset += in1_subblock_stride;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
    if (!skip_in1_pop) {
        cb_pop_front(in1_cb, K * N);
    }
}

/**
 * Strided matmul: out_cb = in0_cb @ in1_cb with custom in1 row stride.
 *
 * Used for MLA where V is the first vDHt columns of K (DHt columns total).
 * in1 is in [K, DHt] layout but we only want [K, N] where N < DHt.
 *
 * @param in1_row_stride: Actual row width of in1 (DHt), stride between K rows
 * @param skip_in1_wait: If true, skip cb_wait_front (caller already ensured tiles available)
 * @param skip_in1_pop: If true, skip cb_pop_front (caller manages CB lifecycle)
 */
ALWI void cb_matmul_blocks_strided(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& in1_row_stride,  // Actual width of in1 rows (DHt)
    const uint32_t& num_blocks,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& skip_in1_wait,
    const bool& skip_in1_pop) {
    // precondition: in0_cb has M*K produced
    // precondition: in1_cb has K*in1_row_stride tiles available (or skip_in1_wait=true)
    // postcondition: in0_cb is full
    // postcondition: out_cb has M*N produced

    // in1 is in [K, in1_row_stride] layout, but we only read [K, N] portion
    // Inner stride is in1_row_stride (not N) to skip extra columns
    const uint32_t in1_inner_stride = in1_row_stride;
    const uint32_t in1_subblock_stride = subblock_w;  // Adjacent columns in output

    mm_block_init_short(
        in0_cb, in1_cb, false /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    reconfig_data_format(in1_cb, in0_cb);

    if (!skip_in1_wait) {
        DeviceZoneScopedN("matmul-wait-in1");
        cb_wait_front(in1_cb, K * in1_row_stride);
    }

    uint32_t output_num_tiles = M * N;
    cb_reserve_back(out_cb, output_num_tiles);
    uint32_t in0_index_offset = 0;

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        uint32_t in1_index_offset = 0;

        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();
            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index,
                    in1_index,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    in0_block_w);
                in0_index++;
                in1_index += in1_inner_stride;  // Stride by actual row width
            }

            uint32_t col_out_tiles = subblock_h * subblock_w;
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(out_cb, col_out_tiles);
            for (uint32_t i = 0; i < col_out_tiles; i++) {
                pack_tile(i, out_cb);
            }
            cb_push_back(out_cb, col_out_tiles);
            tile_regs_release();

            in1_index_offset += in1_subblock_stride;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }

    if (!skip_in1_pop) {
        cb_pop_front(in1_cb, K * in1_row_stride);
    }
}
