// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/softplus.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/reduce_custom.h"

#include <tools/profiler/kernel_profiler.hpp>

using std::uint32_t;
/**
 * in0_cb = exp((in0_cb - in1_cb) * scale_fp32) - Row-pair re-entrant version
 *
 * Processes exactly 2 rows per invocation, allowing fine-grained pipelining.
 * Must be called total_rows/2 times with row_pair_index = 0, 1, 2, ...
 * Precondition: in0_cb has at least (row_pair_index + 1) * 2 * cols tiles produced
 *               (previous row pairs must have completed their in-place rotation)
 * Precondition: in1_cb has at least (row_pair_index + 1) * 2 tiles produced
 * Postcondition: in0_cb has same tile count, with processed row pair rotated to back
 * Postcondition: in1_cb unchanged (never popped)
 * Postcondition: reduce_cb has 2 more tiles produced (if do_reduce=true)
 */
template <
    uint32_t in0_cb,
    uint32_t total_rows,
    uint32_t scale_fp32,
    bool write_result_inplace = true,
    bool do_reduce = true,
    int vector_mode = (int)VectorMode::RC>
void sub_exp_block_bcast_cols_inplace_row_pair(
    uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols, uint32_t row_pair_index) {
    // Constants for 2-row processing
    constexpr uint32_t rows_per_pair = 2;

    // Calculate global row indices for in1_cb access
    uint32_t global_row_base = row_pair_index * rows_per_pair;

    // Initialize operation
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    // exp_tile_init<true, true, scale_fp32>();
    exp_packthread_tile_init<true, true, scale_fp32>();

    // Wait for tiles:
    // - in0_cb: need 2*cols tiles (previous pairs already popped their tiles via in-place rotation)
    // - in1_cb: cumulative wait since we never pop it
    cb_wait_front(in0_cb, rows_per_pair * cols);
    cb_wait_front(in1_cb, (row_pair_index + 1) * rows_per_pair);

    if constexpr (do_reduce) {
        cb_reserve_back(reduce_cb, rows_per_pair);
    }

#ifdef SUB_EXP_GRANULARITY
    uint32_t dst_tiles = (cols < SUB_EXP_GRANULARITY) ? cols : SUB_EXP_GRANULARITY;
    uint32_t granularity = (cols >= SUB_EXP_GRANULARITY) ? (cols >> LOG2_SUB_EXP_GRANULARITY) : 1;
#else
    uint32_t dst_tiles = cols;
    uint32_t granularity = 1;
#endif

    // Process exactly 2 rows
    for (uint32_t local_row = 0; local_row < rows_per_pair; ++local_row) {
        uint32_t global_row = global_row_base + local_row;  // For in1_cb indexing

        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();

            for (uint32_t j = 0; j < dst_tiles; ++j) {
                // in0_cb: index j (relative to current front, which shifts after each pop)
                // in1_cb: index global_row (absolute position, since in1_cb is never popped)
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, global_row, j);
                // exp_tile<true, true>(j, vector_mode);
            }
            tile_regs_commit();

            if constexpr (write_result_inplace) {
                cb_pop_front(in0_cb, dst_tiles);
                cb_reserve_back(in0_cb, dst_tiles);
            }

            tile_regs_wait();

            for (uint32_t j = 0; j < dst_tiles; ++j) {
                exp_packthread_tile<true, true>(j, vector_mode);
            }
            PACK(TTI_STALLWAIT(p_stall ::STALL_PACK, p_stall ::WAIT_SFPU));

            if constexpr (write_result_inplace) {
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, in0_cb);
                }
                // Granular write output to enable following matmul unpack to start early.
                cb_push_back(in0_cb, dst_tiles);
            }

            if constexpr (do_reduce) {
                // While we have results in DST, take advantage of L1 accumulation
                // to reduce row x cols tiles to rows x 1 tiles.
                if (u > 0) {
                    // If on the same row, keep accumulating
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    // Pack to local_row's position in reduce_cb (0 or 1 within this pair)
                    pack_tile<true>(j, reduce_cb, local_row);
                    if (u == 0 && j == 0) {
                        // If this was the first tile of a row, start accumulating
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
            tile_regs_release();

            if constexpr (do_reduce) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
        }
    }

    if constexpr (do_reduce) {
        cb_push_back(reduce_cb, rows_per_pair);
    }
}

ALWI void sdpa_reduce_copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose = 0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, true /*transpose within 16x16 face*/, cbid)));

    MATH((llk_math_eltwise_unary_datacopy_init<
          A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false,  // is_int_fpu_en
          false   // tilize
          >(cbid)));
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t cols,
    int vector_mode = static_cast<int>(VectorMode::C)>
void reduce_c_row_pair(uint32_t out_cb, uint32_t prev_cb, uint32_t row_pair_index, bool do_eltwise_max = false) {
    // Precondition: in0_cb has at least (row_pair_index + 1) * 2 * cols tiles produced (row-major order)
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free (reserved by caller)
    // Precondition: prev_cb has at least (row_pair_index + 1) * 2 tiles produced (if do_eltwise_max)
    // Postcondition: in0_cb unchanged (no pop)
    // Postcondition: out_cb has 2 more tiles written at positions [row_pair_index*2, row_pair_index*2+1]

    constexpr uint32_t PAIR_SIZE = 2;
    const uint32_t row_start = row_pair_index * PAIR_SIZE;

    // Cumulative tile counts for cb_wait_front
    const uint32_t cumulative_input_tiles = (row_pair_index + 1) * PAIR_SIZE * cols;
    const uint32_t cumulative_prev_tiles = (row_pair_index + 1) * PAIR_SIZE;

    // Wait for scale (always needed, returns immediately if already available)
    cb_wait_front(scale_cb, 1);

    // Wait for input tiles up to and including this row pair
    cb_wait_front(in0_cb, cumulative_input_tiles);

    acquire_dst();

    if (do_eltwise_max) {
        cb_wait_front(prev_cb, cumulative_prev_tiles);
        /**
         * Copy previous max values into DST register.
         * Note that this special invocation of copy_tile is necessary to produce
         * tiles in DST with transposed faces, as `reduce_block_max_row` expects.
         */
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < PAIR_SIZE; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
    }

    /**
     * For the 2 rows in this pair, compute the max/sum into DST registers.
     */
    reduce_block_max_row_init<cols>();
    for (uint32_t i = 0; i < PAIR_SIZE; i++) {
        const uint32_t input_tile_start = (row_start + i) * cols;
        const uint32_t reduce_dst_idx = i;
        reduce_block_max_row<cols>(in0_cb, scale_cb, input_tile_start, reduce_dst_idx);
    }
    reduce_block_max_row_uninit(in0_cb);

    // Pack results to output at the correct positions
    cb_reserve_back(out_cb, PAIR_SIZE);
    for (uint32_t i = 0; i < PAIR_SIZE; i++) {
        const uint32_t dst_idx = i;
        pack_tile<true>(dst_idx, out_cb, row_start + i);
    }
    cb_push_back(out_cb, PAIR_SIZE);

    release_dst();
}

// /**
//  * Hacked - this should be written in a better way, so it processes two tiles in a single call.
//  */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t cols,
    int vector_mode = static_cast<int>(VectorMode::C)>
void reduce_row_incremental(uint32_t out_cb, uint32_t prev_cb, uint32_t row_to_reduce, bool do_eltwise_max = false) {
    // Precondition: in0_cb has at least row_to_reduce*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has the same
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced
    // !!! If do_eltwise_max == true, prev_cb has rows produced.

    constexpr uint32_t num_tiles = cols;

    cb_wait_front(scale_cb, 1);
    cb_reserve_back(out_cb, 1);

    const uint32_t num_tiles_to_wait = row_to_reduce * cols + cols;

    cb_wait_front(in0_cb, num_tiles_to_wait);

    tile_regs_acquire();

    if (do_eltwise_max) {
        cb_wait_front(prev_cb, row_to_reduce);
        /**
         * Copy previous max values into DST register.
         * Note that this special invocation of copy_tile is necessary to produce
         * tiles in DST with transposed faces, as `reduce_block_max_row` expects.
         */
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        copy_tile(prev_cb, row_to_reduce, 1);
    }

    reduce_block_max_row_init<cols>();
    reduce_block_max_row<cols>(in0_cb, scale_cb, row_to_reduce * cols, 0);
    reduce_block_max_row_uninit(in0_cb);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile<true>(0, out_cb, row_to_reduce);
    tile_regs_release();

    cb_push_back(out_cb, 1);
}

template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t head_dim_t,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t scale_fp32>
void sdpa_inner_loop(
    const uint32_t cb_max_A, const uint32_t cb_max_B, const uint32_t cb_sum_A, const uint32_t cb_sum_B) {
    // Set up ping pong buffers
    // To be used (and swapped) later on, when we loop over Q chunks.
    uint32_t alias_prev_sum = cb_sum_A;
    uint32_t alias_cur_sum = cb_sum_B;
    uint32_t alias_prev_max = cb_max_A;
    uint32_t alias_cur_max = cb_max_B;

    DPRINT << "alias_prev_sum CB ID: " << alias_prev_sum << ENDL();
    DPRINT << "alias_cur_sum CB ID: " << alias_cur_sum << ENDL();
    DPRINT << "alias_prev_max CB ID: " << alias_prev_max << ENDL();
    DPRINT << "alias_cur_max CB ID: " << alias_cur_max << ENDL();

    // Hardcoded for Q[8x4], Kt[4x16]
    const uint32_t in0_block_w = 4;
    const uint32_t subblock_h = 2;
    const uint32_t subblock_w = 4;
    const uint32_t q_num_subblocks = 4;
    const uint32_t kt_num_subblocks = 4;

    const uint32_t q_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    const uint32_t output_num_tiles_per_row = Sq_chunk_t * subblock_h;

    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    pack_reconfig_data_format(cb_qkt_im);
    reconfig_data_format(cb_kt_in, cb_q_in);
    cb_reserve_back(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);

    cb_wait_front(cb_kt_in, head_dim_t * Sk_chunk_t);
    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        bool is_first_row_pair = (q_subblock == 0);
        if (!is_first_row_pair) {
            MATH(DPRINT << "SUB_EXP for Q[" << q_subblock - 1 << "]" << ENDL());
            {
                DeviceZoneScopedN("SUB_EXP");
                sub_exp_block_bcast_cols_inplace_row_pair<cb_qkt_im, Sq_chunk_t, scale_fp32, true>(
                    alias_cur_max, alias_cur_sum, Sk_chunk_t, q_subblock - 1 /*row_pair_index*/);
            }
        }
        cb_wait_front(cb_q_in, q_wait_tiles);
        kt_index_offset = 0;
        mm_block_init_short(
            cb_q_in,
            cb_kt_in,
            true /*transpose*/,
            subblock_w /*ct_dim*/,
            subblock_h /*rt_dim*/,
            in0_block_w /*kt_dim*/);

        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
            MATH(DPRINT << "Matmul for Q[" << q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());
            {
                DeviceZoneScopedN("matmul_blocks 2x4");
                tile_regs_acquire();
                uint32_t dst_index = 0;
                uint32_t q_index = q_index_offset;
                uint32_t kt_index = kt_index_offset;
                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                    matmul_block(
                        cb_q_in,
                        cb_kt_in,
                        q_index,
                        kt_index,
                        dst_index,
                        true /*transpose*/,
                        subblock_w,
                        subblock_h,
                        in0_block_w);
                    q_index++;
                    kt_index += Sq_chunk_t;
                }
                tile_regs_commit();
            }
            {
                DeviceZoneScopedN("Pack 2x4");
                // Pack the subblock
                tile_regs_wait();
                uint32_t dst_idx = 0;
                uint32_t out_col_offset = kt_subblock * subblock_w;
                for (uint32_t r = 0; r < subblock_h; r++) {
                    uint32_t out_row_offset = (r + q_subblock * subblock_h) * Sk_chunk_t;
                    for (uint32_t c = 0; c < subblock_w; c++) {
                        pack_tile<true>(dst_idx, cb_qkt_im, out_row_offset + out_col_offset + c);
                        dst_idx++;
                    }
                }
                tile_regs_release();
                cb_push_back(cb_qkt_im, subblock_h * Sk_chunk_t);
            }
            kt_index_offset += subblock_w;
        }

        // Max reduce
        MATH(DPRINT << "Max reduce for Q[" << q_subblock << ", :]" << ENDL());
        //  JUST TO ENSURE THE WORST-CASE IS MEASURED, DO THE ELTWISE MAX EVERY TIME
        static_assert(subblock_h == 2, "subblock_h must be 2");
        {
            DeviceZoneScopedN("Reduce max 2x16");
            reduce_c_row_pair<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qkt_im, cb_identity_scale_in, Sk_chunk_t>(
                alias_cur_max, alias_prev_max, q_subblock, true /*do_eltwise_max*/);
        }

        q_index_offset += subblock_h * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }
    MATH(DPRINT << "DRAIN: SUB_EXP for Q[" << 3 << "]" << ENDL());
    {
        DeviceZoneScopedN("SUB_EXP");
        sub_exp_block_bcast_cols_inplace_row_pair<cb_qkt_im, Sq_chunk_t, scale_fp32, true>(
            alias_cur_max, alias_cur_sum, Sk_chunk_t, 3 /*row_pair_index*/);
    }

    cb_pop_front(cb_q_in, head_dim_t * Sq_chunk_t);
    cb_pop_front(cb_kt_in, head_dim_t * Sk_chunk_t);
    cb_pop_front(alias_cur_max, Sq_chunk_t);
    cb_pop_front(alias_cur_sum, Sq_chunk_t);

    // dummy: use QKT somehow
    cb_wait_front(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);
    cb_pop_front(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);
}

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(2);
    constexpr uint32_t num_iter = get_compile_time_arg_val(3);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(4);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_kt_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_qkt_im = tt::CBIndex::c_2;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;

    mm_init(cb_q_in, cb_kt_in, cb_qkt_im);

    for (uint32_t iter = 0; iter < num_iter; iter++) {
        MATH(DPRINT << "Iteration " << iter << ENDL());
        sdpa_inner_loop<
            Sq_chunk_t,
            Sk_chunk_t,
            head_dim_t,
            cb_q_in,
            cb_kt_in,
            cb_qkt_im,
            cb_identity_scale_in,
            scale_fp32>(cb_max_A, cb_max_B, cb_sum_A, cb_sum_B);
    }
}
