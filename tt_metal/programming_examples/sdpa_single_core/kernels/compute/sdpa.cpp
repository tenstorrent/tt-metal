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

// Define this macro to enable high-granularity DeviceZoneScopedN profiling in sdpa_inner_loop_8x4x16
#define SDPA_HIGH_GRANULARITY_PROFILING

#ifdef SDPA_HIGH_GRANULARITY_PROFILING
#define SDPA_DeviceZoneScopedN(name) DeviceZoneScopedN(name)
#else
#define SDPA_DeviceZoneScopedN(name)
#endif

using std::uint32_t;
template <uint32_t in0_cb, uint32_t scale_fp32, bool do_reduce = true, int vector_mode = (int)VectorMode::RC>
void sub_exp_block_bcast_cols_inplace_2x4(
    uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols, uint32_t q_subblock, uint32_t kt_subblock) {
    constexpr uint32_t tiles_per_row = 2;
    constexpr uint32_t tiles_per_column = 4;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    const uint32_t global_col_base = kt_subblock * tiles_per_column;

    // Initialize operation
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    // exp_packthread_tile_init<true, true, scale_fp32>();  // todo: move outside.

    // Wait for tiles:
    // - in0_cb: cumulative wait since we never pop it
    // - in1_cb: cumulative wait since we never pop it
    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row * cols);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    // if constexpr (do_reduce) {
    //     cb_reserve_back(reduce_cb, tiles_per_row);
    // }

    {
        SDPA_DeviceZoneScopedN("SUB");
        tile_regs_acquire();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                uint32_t in0_tile_index =
                    (global_row_base + i) * cols + (global_col_base + j);  // Absolute tile index for in0_cb
                sub_tiles_bcast_cols(in0_cb, in1_cb, in0_tile_index, global_row_base + i, dst_index++);
            }
        }
        tile_regs_commit();
    }

    {
        SDPA_DeviceZoneScopedN("EXP");
        tile_regs_wait();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                exp_packthread_tile<true, true>(dst_index++, vector_mode);
            }
        }
        PACK(TTI_STALLWAIT(p_stall ::STALL_PACK, p_stall ::WAIT_SFPU));
    }

    {
        SDPA_DeviceZoneScopedN("EXP PACK");
        tile_regs_wait();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; ++j) {
                uint32_t in0_tile_index =
                    (global_row_base + i) * cols + (global_col_base + j);  // Absolute tile index for in0_cb
                pack_tile<true>(dst_index++, in0_cb, in0_tile_index);      // Pack back to original position in in0_cb
            }
        }

        if constexpr (do_reduce) {
            dst_index = 0;
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                // While we have results in DST, take advantage of L1 accumulation
                // to reduce row x cols tiles to rows x 1 tiles.
                if (global_col_base > 0) {
                    // If on the same row, keep accumulating
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
                for (uint32_t j = 0; j < tiles_per_column; ++j) {
                    // Pack to local_row's position in reduce_cb (0 or 1 within this pair)
                    pack_tile<true>(dst_index++, reduce_cb, global_row_base + i);
                    if (global_col_base == 0 && j == 0) {
                        // If this was the first tile of a row, start accumulating
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
        }
        tile_regs_release();
        if constexpr (do_reduce) {  // todo: move up?
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
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
     * For the 2 rows in this pair, compute the max into DST registers.
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

template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Sv_chunk_t,
    uint32_t head_dim_t,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_out,
    uint32_t scale_fp32>
void sdpa_inner_loop_8x4x16(
    const uint32_t cb_max_A, const uint32_t cb_max_B, const uint32_t cb_sum_A, const uint32_t cb_sum_B) {
    DeviceZoneScopedN("sdpa_inner_loop_8x4x16");
    // Set up ping pong buffers
    // To be used (and swapped) later on, when we loop over Q chunks.
    uint32_t alias_prev_sum = cb_sum_A;
    uint32_t alias_cur_sum = cb_sum_B;
    uint32_t alias_prev_max = cb_max_A;
    uint32_t alias_cur_max = cb_max_B;

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

    exp_packthread_tile_init<true, true, scale_fp32>();  // todo: move outside.
    pack_reconfig_data_format(cb_qkt_im);
    reconfig_data_format(cb_kt_in, cb_q_in);
    cb_reserve_back(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);

    cb_wait_front(cb_kt_in, head_dim_t * Sk_chunk_t);
    cb_reserve_back(alias_cur_sum, Sq_chunk_t);
    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        SDPA_DeviceZoneScopedN("Softmax(Q@KT) 2x4x16");
        cb_wait_front(cb_q_in, q_wait_tiles);
        kt_index_offset = 0;

        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
            if (q_subblock > 0) {
                uint32_t prev_q_subblock = q_subblock - 1;
                MATH(DPRINT << "SUB EXP for Q[" << prev_q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());
                sub_exp_block_bcast_cols_inplace_2x4<cb_qkt_im, scale_fp32, true>(
                    alias_cur_max, alias_cur_sum, Sk_chunk_t, prev_q_subblock, kt_subblock);
            }

            {
                {
                    SDPA_DeviceZoneScopedN("matmul_blocks 2x4 init");
                    mm_block_init_short(
                        cb_q_in,
                        cb_kt_in,
                        true /*transpose*/,
                        subblock_w /*ct_dim*/,
                        subblock_h /*rt_dim*/,
                        in0_block_w /*kt_dim*/);
                }
                MATH(DPRINT << "Matmul for Q[" << q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());

                {
                    SDPA_DeviceZoneScopedN("matmul_blocks 2x4");

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
                        kt_index += Sk_chunk_t;
                    }
                    // tensix_sync();
                    tile_regs_commit();
                }
            }
            {
                SDPA_DeviceZoneScopedN("Pack 2x4");
                // Pack the subblock
                tile_regs_wait();
                PACK(DPRINT << "Pack for Q[" << q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());
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
                MATH(
                    DPRINT << "Packing " << subblock_h * subblock_w << " tiles to cb_qkt_im for Q[" << q_subblock
                           << "] Kt[" << kt_subblock << "]" << ENDL());
            }
            kt_index_offset += subblock_w;
        }
        cb_push_back(cb_qkt_im, subblock_h * Sk_chunk_t);

        // Max reduce
        MATH(DPRINT << "Max reduce for Q[" << q_subblock << ", :]" << ENDL());
        //  JUST TO ENSURE THE WORST-CASE IS MEASURED, DO THE ELTWISE MAX EVERY TIME
        static_assert(subblock_h == 2, "subblock_h must be 2");
        {
            SDPA_DeviceZoneScopedN("Reduce max 2x16");
            reduce_c_row_pair<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qkt_im, cb_identity_scale_in, Sk_chunk_t>(
                alias_cur_max, alias_prev_max, q_subblock, true /*do_eltwise_max*/);
        }

        q_index_offset += subblock_h * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }

    cb_pop_front(cb_q_in, head_dim_t * Sq_chunk_t);
    cb_pop_front(cb_kt_in, head_dim_t * Sk_chunk_t);

    // QKT @ V: compute attention output
    // in0 = cb_qkt_im: Sq_chunk_t × Sk_chunk_t (M × K) — already produced
    // in1 = cb_v_in:   Sv_chunk_t × head_dim_t  (K × N)
    // out = cb_out:     Sq_chunk_t × head_dim_t  (M × N)
    // Drain sub_exp for the last Q@KT row is interleaved with the first QKT@V q_subblock.
    // sub_exp uses SFPU for exp, matmul uses FPU — they overlap on different hardware units.
    MATH(DPRINT << "Starting QKT @ V computation" << ENDL());
    {
        constexpr uint32_t qktv_subblock_h = 2;
        constexpr uint32_t qktv_subblock_w = 4;
        constexpr uint32_t qktv_in0_block_w = Sv_chunk_t;
        constexpr uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_subblock_h;
        constexpr uint32_t qktv_v_num_subblocks = head_dim_t / qktv_subblock_w;
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * head_dim_t;
        constexpr uint32_t qktv_in0_subblock_num_tiles = qktv_subblock_h * qktv_in0_block_w;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_subblock_num_tiles;

        MATH(DPRINT << "Waiting for cb_v_in: " << Sv_chunk_t * head_dim_t << " tiles" << ENDL());
        cb_wait_front(cb_v_in, Sv_chunk_t * head_dim_t);
        MATH(DPRINT << "Reserving cb_out: " << qktv_output_num_tiles << " tiles" << ENDL());
        cb_reserve_back(cb_out, qktv_output_num_tiles);

        for (uint32_t q_subblock = 0; q_subblock < qktv_q_num_subblocks; ++q_subblock) {
            MATH(DPRINT << "QKT@V: Processing Q_subblock " << q_subblock << ENDL());
            SDPA_DeviceZoneScopedN("Softmax(Q@KT)@V 2x16x4");
            cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);

            // Drain: interleave sub_exp for last Q@KT row with first QKT@V matmul
            // if (q_subblock == 0)
            {
                MATH(
                    DPRINT << "DRAIN: SUB_EXP for Q[" << q_num_subblocks - 1 << "] during QKT@V q_subblock 0"
                           << ENDL());
                // for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
                //  Gradually drain the last Q@KT subblock's sub_exp across all 4 QKT@V subblocks, to best overlap exp
                //  computation with matmul.
                sub_exp_block_bcast_cols_inplace_2x4<cb_qkt_im, scale_fp32, true>(
                    alias_cur_max, alias_cur_sum, Sk_chunk_t, q_num_subblocks - 1, q_subblock);
                //}
                // Reconfigure for matmul after sub_exp changed pack/unpack config
                // pack_reconfig_data_format(cb_out);
                // reconfig_data_format(cb_v_in, cb_qkt_im);
            }

            {
                mm_block_init_short(
                    cb_qkt_im,
                    cb_v_in,
                    false /*transpose*/,
                    qktv_subblock_w /*ct_dim*/,
                    qktv_subblock_h /*rt_dim*/,
                    qktv_in0_block_w /*kt_dim*/);
            }

            uint32_t v_index_offset = 0;
            for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                MATH(DPRINT << "QKT@V Matmul for Q[" << q_subblock << "] V[" << v_subblock << "]" << ENDL());
                {
                    SDPA_DeviceZoneScopedN("QKT@V matmul 2x4");
                    tile_regs_acquire();

                    uint32_t dst_index = 0;
                    uint32_t in0_index = qktv_in0_index_offset;
                    uint32_t in1_index = v_index_offset;

                    for (uint32_t inner = 0; inner < qktv_in0_block_w; ++inner) {
                        matmul_block(
                            cb_qkt_im,
                            cb_v_in,
                            in0_index,
                            in1_index,
                            dst_index,
                            false /*transpose*/,
                            qktv_subblock_w,
                            qktv_subblock_h,
                            qktv_in0_block_w);
                        in0_index++;
                        in1_index += head_dim_t;
                    }
                    tile_regs_commit();
                }

                {
                    SDPA_DeviceZoneScopedN("QKT@V pack 2x4");
                    tile_regs_wait();
                    PACK(DPRINT << "QKT@V Pack for Q[" << q_subblock << "] V[" << v_subblock << "]" << ENDL());
                    uint32_t dst_idx = 0;
                    uint32_t out_col_offset = v_subblock * qktv_subblock_w;
                    for (uint32_t r = 0; r < qktv_subblock_h; r++) {
                        uint32_t out_row_offset = (r + q_subblock * qktv_subblock_h) * head_dim_t;
                        for (uint32_t c = 0; c < qktv_subblock_w; c++) {
                            pack_tile<true>(dst_idx, cb_out, out_row_offset + out_col_offset + c);
                            dst_idx++;
                        }
                    }
                    tile_regs_release();
                    MATH(
                        DPRINT << "Packed " << qktv_subblock_h * qktv_subblock_w << " tiles to cb_out for Q["
                               << q_subblock << "] V[" << v_subblock << "]" << ENDL());
                }

                v_index_offset += qktv_subblock_w;
            }

            qktv_in0_index_offset += qktv_subblock_h * qktv_in0_block_w;
            qktv_in0_wait_tiles += qktv_in0_subblock_num_tiles;
            MATH(
                DPRINT << "Pushing " << qktv_subblock_h * head_dim_t << " tiles to cb_out for Q_subblock " << q_subblock
                       << ENDL());
            cb_push_back(cb_out, qktv_subblock_h * head_dim_t);
        }

        MATH(DPRINT << "Popping cb_v_in: " << Sv_chunk_t * head_dim_t << " tiles" << ENDL());
        cb_pop_front(cb_v_in, Sv_chunk_t * head_dim_t);
        MATH(DPRINT << "Popping cb_qkt_im: " << Sq_chunk_t * Sk_chunk_t << " tiles" << ENDL());
        cb_pop_front(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);
    }

    cb_pop_front(alias_cur_max, Sq_chunk_t);
    cb_push_back(alias_cur_sum, Sq_chunk_t);
    cb_pop_front(alias_cur_sum, Sq_chunk_t);

    cb_wait_front(cb_out, Sq_chunk_t * head_dim_t);
    cb_pop_front(cb_out, Sq_chunk_t * head_dim_t);
    MATH(DPRINT << "Finished QKT @ V computation" << ENDL());
    // tensix_sync();
}

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(3);
    constexpr uint32_t num_iter = get_compile_time_arg_val(4);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(5);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_kt_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_qkt_im = tt::CBIndex::c_2;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;

    mm_init(cb_q_in, cb_kt_in, cb_qkt_im);

    for (uint32_t iter = 0; iter < num_iter; iter++) {
        MATH(DPRINT << "Iteration " << iter << ENDL());
        {
            sdpa_inner_loop_8x4x16<
                Sq_chunk_t,
                Sk_chunk_t,
                Sv_chunk_t,
                head_dim_t,
                cb_q_in,
                cb_kt_in,
                cb_v_in,
                cb_qkt_im,
                cb_identity_scale_in,
                cb_out,
                scale_fp32>(cb_max_A, cb_max_B, cb_sum_A, cb_sum_B);
        }
    }
}
