// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// cp_inner_loop_step: the stock streaming sdpa_inner_loop_step (compute_streaming.hpp) specialised to the one
// configuration qwen_cp_sdpa runs: causal, lightweight causal mask, no ring / padding / sliding window / sink /
// provided mask / KV-pad rotation / in-place V / saved stats, KT_stride == Sk_chunk_t. Every branch that is dead for
// that configuration is removed. Two changes cut packer stalls (each L1-acc / ReLU reconfiguration waits for the
// packer to go idle) without changing any arithmetic, so the output stays bit-identical to the stock step
// (tests/test_qwen_cp_sdpa_op.py):
// - the row-sum packs of a sub_exp block are grouped so the block needs one L1-acc switch instead of 3-5;
// - the causal mask stamp (two L1-acc switches + a copy init per row group) is skipped for row groups that lie
//   wholly below the diagonal, where the stock stamp writes only columns past active_Sk that nothing reads;
// - the unpack side of the drain -> main-loop pack->unpack barrier is taken right before the only P@V that reads
//   the drain's in-place writes (the last row group's), not before the first one.
// 40k unit (scripts/cp_sdpa_split.py, pipelined): 12.490 -> 12.230 ms per call (stall cuts), -> 12.176 (barrier).
//
// Compile-time knobs (op.py QWEN_CP_SDPA_DEFINES, measurement only unless stated):
//   CP_PROF          per-TRISC cycle accumulators per stage, emitted once per kernel as DeviceTimestampedData
//                    (device profiler on; see cp_prof_* below).
//   CP_SKIP_EXP      no exp in the score sub_exp (wrong output; time split only).
//   CP_SKIP_QK_MM    no Q@K^T matmul instructions (wrong output; time split only).
//   CP_SKIP_PV_MM    no P@V matmul instructions (wrong output; time split only).
//   CP_SKIP_SALAD    no SALAD correction math (wrong output; time split only).
//   CP_SKIP_SUB      no max subtraction before the exp (wrong output; time split only).
//   CP_SKIP_SUMPACK  no L1-accumulated row-sum packs of the exp tiles (wrong output; time split only).
//   CP_SKIP_MASK     no causal mask stamp (wrong output; time split only).

#pragma once

#include <cstdint>

// ----------------------------------------------------------------------------------------------------------------
// Stage cycle accumulators (CP_PROF). Each TRISC accumulates the wall-clock cycles it spends inside each stage;
// a TRISC only blocks on its own waits (CB fronts, DEST handshakes, semaphores, a full instruction FIFO), so a
// stage's count on one thread is that thread's issue + wait time there, not engine time.
// ----------------------------------------------------------------------------------------------------------------
enum CpStage : uint32_t {
    CP_ST_QK = 0,   // phase 1 Q@K^T matmul + pack
    CP_ST_SUBEXP1,  // phase 1 in-place sub + exp + row sum of the previous row group
    CP_ST_MASK,     // phase 1 causal mask stamp
    CP_ST_REDUCE,   // phase 1 row max (+ eltwise max with prev)
    CP_ST_DRAIN,    // phase 2 last row group sub + exp interleaved with its partial P@V
    CP_ST_PV,       // phase 2 P@V matmul + pack, row groups 1..N-1
    CP_ST_SALAD,    // phase 2 exp(prev max - cur max) + output / sum correction
    CP_ST_NORM,     // phase 2 1/sum + normalise (last K chunk only)
    CP_ST_INIT1,    // phase 1 per-row-group reconfig / matmul init + K / Q waits
    CP_ST_STEP,     // whole step
    CP_ST_N
};

#if defined(CP_PROF) && defined(PROFILE_KERNEL)
static uint32_t cp_prof_acc[CP_ST_N];
static uint32_t cp_prof_iters;
ALWI uint32_t cp_now() { return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }
struct CpProfScope {
    uint32_t stage, t0;
    ALWI CpProfScope(uint32_t s) : stage(s), t0(cp_now()) {}
    ALWI ~CpProfScope() { cp_prof_acc[stage] += cp_now() - t0; }
};
#define CP_PROF_SCOPE(stage) CpProfScope cp_prof_scope_##stage(stage)
#define CP_PROF_ITER() cp_prof_iters++
ALWI void cp_prof_emit() {
    DeviceTimestampedData("CP_QK", cp_prof_acc[CP_ST_QK]);
    DeviceTimestampedData("CP_SUBEXP1", cp_prof_acc[CP_ST_SUBEXP1]);
    DeviceTimestampedData("CP_MASK", cp_prof_acc[CP_ST_MASK]);
    DeviceTimestampedData("CP_REDUCE", cp_prof_acc[CP_ST_REDUCE]);
    DeviceTimestampedData("CP_DRAIN", cp_prof_acc[CP_ST_DRAIN]);
    DeviceTimestampedData("CP_PV", cp_prof_acc[CP_ST_PV]);
    DeviceTimestampedData("CP_SALAD", cp_prof_acc[CP_ST_SALAD]);
    DeviceTimestampedData("CP_NORM", cp_prof_acc[CP_ST_NORM]);
    DeviceTimestampedData("CP_INIT1", cp_prof_acc[CP_ST_INIT1]);
    DeviceTimestampedData("CP_STEP", cp_prof_acc[CP_ST_STEP]);
    DeviceTimestampedData("CP_ITERS", cp_prof_iters);
}
#else
#define CP_PROF_SCOPE(stage)
#define CP_PROF_ITER()
ALWI void cp_prof_emit() {}
#endif

// ----------------------------------------------------------------------------------------------------------------
// Local copies of the stock helpers the time-split knobs need to change (identical when no knob is set).
// ----------------------------------------------------------------------------------------------------------------

// blocked_matmul_and_pack (compute_streaming.hpp); skip_mm drops the matmul instructions only.
template <bool transpose, uint32_t in1_stride, uint32_t out_num_cols, bool skip_mm = false>
#if defined(ARCH_WORMHOLE)
__attribute__((noinline))
#endif
void cp_blocked_matmul_and_pack(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t row_subblock_idx,
    uint32_t out_col_offset,
    uint32_t subblock_w,
    uint32_t subblock_h,
    uint32_t inner_dim,
    uint32_t matmul_stride,
    bool skip_pack_configure = false) {
    tile_regs_acquire();
    if constexpr (!skip_mm) {
        uint32_t dst_index = 0;
        uint32_t in0_index = in0_index_start;
        uint32_t in1_index = in1_index_start;
        for (uint32_t inner = 0; inner < inner_dim; ++inner) {
            sdpa_matmul_block_no_mop(
                in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, matmul_stride);
            in0_index++;
            in1_index += in1_stride;
        }
    }
    tile_regs_commit();

    tile_regs_wait();
    if (!skip_pack_configure) {
        configure_row_pack_width(out_cb, subblock_w);
    }
    pack_contiguous_rows_nocfg(
        out_cb, row_subblock_idx * subblock_h, subblock_h, out_num_cols, out_col_offset, subblock_w);
    tile_regs_release();
}

// sub_exp_block_bcast_cols (compute_streaming.hpp), profiling zones removed.
template <uint32_t scale_fp32>
void cp_sub_exp_block_bcast_cols(
    uint32_t inout_cb,
    uint32_t max_cb,
    uint32_t reduce_cb,
    uint32_t cols_in_row,
    uint32_t q_subblock,
    uint32_t global_col_base,
    uint32_t sbh,
    uint32_t sbw,
    bool skip_pack_configure = false) {
    const uint32_t tiles_per_row = sbh;
    const uint32_t tiles_per_column = sbw;
    const uint32_t max_row_base = q_subblock * tiles_per_row;

    sub_bcast_cols_init_short_custom(inout_cb, max_cb, tiles_per_column);

    // inout_cb assumed ready (max_cb was already computed from it)
    CircularBuffer(max_cb).wait_front((q_subblock + 1) * tiles_per_row);

    tile_regs_acquire();
    {
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base;
#ifdef CP_SKIP_SUB
            (void)in0_tile_index;
#else
            sub_tiles_bcast_cols_custom(
                inout_cb, max_cb, in0_tile_index, max_row_base + i, dst_index, tiles_per_column);
#endif
            dst_index += tiles_per_column;
        }
    }
    tile_regs_commit();

    tile_regs_wait();
    PACK((llk_pack_relu_config(ReluConfig::zero())));
    {
#ifndef CP_SKIP_EXP
        uint32_t dst_index = 0;
        constexpr int iterations = 32;
        constexpr VectorMode vector_mode_exp = VectorMode::None;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index++, vector_mode_exp);
            }
        }
#endif
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }

    {
        if (skip_pack_configure) {
            pack_contiguous_rows_nocfg(
                inout_cb, max_row_base, tiles_per_row, cols_in_row, global_col_base, tiles_per_column);
        } else {
            pack_contiguous_rows(inout_cb, max_row_base, tiles_per_row, cols_in_row, global_col_base, tiles_per_column);
        }
        configure_single_tile_pack(reduce_cb);
#ifndef CP_SKIP_SUMPACK
        // Row sums: L1-accumulated packs of the exp tiles in the stock per-row order (column 0 overwrites, then
        // columns 1.. accumulate), grouped so that one sub_exp block needs a single L1-acc switch: each switch
        // stalls until the packer is idle, and the stock loop switches per row (3-5 per block). L1 acc is off on
        // entry (every L1-acc user turns it off again).
        {
            if (global_col_base == 0) {
#pragma GCC unroll 1
                for (uint32_t i = 0; i < tiles_per_row; i++) {
                    pack_tile<true>(i * tiles_per_column, reduce_cb, max_row_base + i);
                }
            }
            const uint32_t j0 = global_col_base == 0 ? 1 : 0;
            if (j0 < tiles_per_column) {
                PACK((llk_pack_reconfig_l1_acc(1)));
#pragma GCC unroll 1
                for (uint32_t i = 0; i < tiles_per_row; i++) {
#pragma GCC unroll 1
                    for (uint32_t j = j0; j < tiles_per_column; ++j) {
                        pack_tile<true>(i * tiles_per_column + j, reduce_cb, max_row_base + i);
                    }
                }
            }
        }
#endif
    }

    tile_regs_release();

    PACK((llk_pack_relu_config(ReluConfig::none())));
    PACK((llk_pack_reconfig_l1_acc(0)));
}

// reduce_c_row_group (compute_streaming.hpp) without the mirror CB.
template <uint32_t in0_cb, uint32_t scale_cb, uint32_t row_stride>
void cp_reduce_c_row_group(
    uint32_t out_cb,
    uint32_t prev_cb,
    uint32_t row_group_index,
    bool do_eltwise_max,
    uint32_t sbh,
    uint32_t reduce_cols,
    bool respect_trigger = false,
    bool overlap_first_half = false) {
    const uint32_t group_size = sbh;
    const uint32_t row_start = row_group_index * group_size;
    const uint32_t cumulative_input_tiles = (row_group_index + 1) * group_size * row_stride;
    const uint32_t cumulative_prev_tiles = (row_group_index + 1) * group_size;

    tile_regs_acquire();

    if (do_eltwise_max) {
        CircularBuffer(prev_cb).wait_front(cumulative_prev_tiles);
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < group_size; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
    }

    if (!respect_trigger) {
        CircularBuffer(in0_cb).wait_front(cumulative_input_tiles);
    }

    reduce_block_max_row_init_runtime(out_cb, reduce_cols, in0_cb, scale_cb, respect_trigger);
    for (uint32_t i = 0; i < group_size; i++) {
        const uint32_t input_tile_start = (row_start + i) * row_stride;
        reduce_block_max_row_runtime(in0_cb, scale_cb, input_tile_start, i, respect_trigger, overlap_first_half);
    }
    reduce_block_max_row_uninit_runtime(in0_cb, respect_trigger, overlap_first_half);

    tile_regs_commit();
    tile_regs_wait();

    for (uint32_t i = 0; i < group_size; i++) {
        pack_tile<false>(i, out_cb);
    }

    tile_regs_release();
}

#ifdef CP_SKIP_QK_MM
constexpr bool cp_skip_qk_mm = true;
#else
constexpr bool cp_skip_qk_mm = false;
#endif
#ifdef CP_SKIP_PV_MM
constexpr bool cp_skip_pv_mm = true;
#else
constexpr bool cp_skip_pv_mm = false;
#endif

/**
 * One K-chunk iteration (stock sdpa_inner_loop_step, causal non-ring streaming case).
 * Phase 1: Q@KT directly into cb_qkt_im with cb_push_back_hold_wr_ptr, in-place sub_exp.
 * Phase 2: Drain + QKT@V with SALAD corrections, streaming normalization on last K iter.
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_h,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_h,
    uint32_t qktv_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    uint32_t cb_mask_in>
static void cp_inner_loop_step(
    AccumulatorHalf& prev,
    AccumulatorHalf& cur,
    const bool is_last_iter,
    const bool is_first_iter,
    const uint32_t active_Sk,
    const bool reduce_trigger,
    const uint32_t actual_sbw,
    const uint32_t mask_q_start_tile,
    const uint32_t mask_k_start_tile,
    const uint32_t neginf_idx,
    const uint32_t primary_diag_idx) {
    CP_PROF_SCOPE(CP_ST_STEP);
    CP_PROF_ITER();
    constexpr uint32_t KT_stride = Sk_chunk_t;
    // Callers guarantee active_Sk is evenly divisible by actual_sbw (via largest_factor_le).
    const uint32_t kt_num_full_subblocks = active_Sk / actual_sbw;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t in0_block_w = DHt;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qkt_subblock_h;
    constexpr uint32_t q_subblock_num_tiles = qkt_subblock_h * in0_block_w;
    constexpr uint32_t row_tiles = qkt_subblock_h * KT_stride;

    uint32_t pushed_rows = 0;
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();

    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT_stride);
    CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);

    // ========== PHASE 1: Q@KT directly into cb_qkt_im ==========
    {
        CP_PROF_SCOPE(CP_ST_INIT1);
        CircularBuffer(cb_kt_in).wait_front(DHt * KT_stride);
    }

    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        {
            CP_PROF_SCOPE(CP_ST_INIT1);
            CircularBuffer(cb_q_in).wait_front(q_wait_tiles);
            kt_index_offset = 0;

            sdpa_maybe_pack_reconfig_data_format<cb_normalized_out, cb_qkt_im>();
            sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_identity_scale_in, cb_q_in>();
            sdpa_mm_init(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);
            configure_row_pack_width(cb_qkt_im, actual_sbw);
        }

        // Causal lightweight mask: stamped below for row groups that reach the diagonal.
        // run()#1 (the reduce's first half) may overlap the second-half pack only if no masked column lands in
        // [0, active_Sk/2) — see sdpa_first_half_unmasked.
        const bool overlap_first_half = reduce_trigger && sdpa_first_half_unmasked<true, false, 0, false, Sk_chunk_t>(
                                                              false,
                                                              true,
                                                              false,
                                                              false,
                                                              0,
                                                              0,
                                                              mask_q_start_tile,
                                                              mask_k_start_tile,
                                                              q_subblock * qkt_subblock_h,
                                                              active_Sk);
        const uint32_t first_half_last_sb = (active_Sk / 2 - 1) / actual_sbw;

        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_full_subblocks; ++kt_subblock) {
            if (q_subblock > 0) {
                CP_PROF_SCOPE(CP_ST_SUBEXP1);
                uint32_t prev_q_subblock = q_subblock - 1;
                sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
                cp_sub_exp_block_bcast_cols<scale_fp32>(
                    cb_qkt_im,
                    cur.max,
                    cur.sum,
                    KT_stride,
                    prev_q_subblock,
                    kt_subblock * actual_sbw,
                    qkt_subblock_h,
                    actual_sbw,
                    /*skip_pack_configure=*/true);
                sdpa_maybe_pack_reconfig_data_format<cb_recip_scratch, cb_qkt_im>();
                sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in>();
                sdpa_mm_reinit(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);
            }
            {
                CP_PROF_SCOPE(CP_ST_QK);
                cp_blocked_matmul_and_pack<true, KT_stride, KT_stride, cp_skip_qk_mm>(
                    cb_q_in,
                    cb_kt_in,
                    cb_qkt_im,
                    q_index_offset,
                    kt_index_offset,
                    q_subblock,
                    kt_subblock * actual_sbw,
                    actual_sbw,
                    qkt_subblock_h,
                    in0_block_w,
                    in0_block_w,
                    /*skip_pack_configure=*/q_subblock == 0);
                if (overlap_first_half && kt_subblock == first_half_last_sb) {
                    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::UNPACK_MATH_DONE)));
                }
                kt_index_offset += actual_sbw;
            }
        }
        // Restore float16b for mask/reduce after Q@KT.
        sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();

        {
            CP_PROF_SCOPE(CP_ST_MASK);
#ifndef CP_SKIP_MASK
            // Rows of this group whose diagonal lies at or past active_Sk stamp nothing; the only other writes are
            // neginf fills of the columns past active_Sk, which no consumer reads (reduce, sub_exp and P@V all
            // stop at active_Sk).
            if (mask_q_start_tile + q_subblock * qkt_subblock_h < mask_k_start_tile + active_Sk) {
                begin_mask_l1_accumulate<false>(cb_qkt_im, cb_mask_in);
                apply_lightweight_mask_streaming<KT_stride, true, false, 0, 0, 0, 0, false>(
                    cb_mask_in,
                    cb_qkt_im,
                    q_subblock,
                    Sk_chunk_t - active_Sk,
                    false,
                    0,
                    qkt_subblock_h,
                    true,
                    neginf_idx,
                    primary_diag_idx,
                    0,
                    0,
                    0,
                    mask_q_start_tile,
                    mask_k_start_tile,
                    active_Sk,
                    false,
                    0,
                    0,
                    {});
                end_mask_l1_accumulate();
            }
#endif
        }

        // Push row (visible for UNPACK reads) but keep wr_ptr stable
        cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles);

        if (reduce_trigger) {
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::FPU_SFPU)));
        }

        {
            CP_PROF_SCOPE(CP_ST_REDUCE);
            CircularBuffer(cur.max).reserve_back(qkt_subblock_h);
            configure_single_tile_pack(cur.max);
            cp_reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT_stride>(
                cur.max,
                prev.max,
                q_subblock,
                !is_first_iter /*do_eltwise_max*/,
                qkt_subblock_h,
                active_Sk,
                reduce_trigger,
                overlap_first_half);
            CircularBuffer(cur.max).push_back(qkt_subblock_h);
        }

        q_index_offset += qkt_subblock_h * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }

    CircularBuffer(cb_kt_in).pop_front(DHt * KT_stride);

    // Q is no longer needed after Phase 1. On the last K chunk, pop early so the
    // reader can start fetching the next Q chunk during Phase 2.
    if (is_last_iter) {
        sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
    }

    // ========== PHASE 2: Drain last row + QKT@V + SALAD ==========
    {
        constexpr uint32_t qktv_h =
            ttnn::transformer::sdpa::streaming_qktv_h(qktv_subblock_h, qktv_subblock_w, dst_size, Sq_chunk_t);
        constexpr uint32_t qktv_remainder_h = Sq_chunk_t % qktv_h;
        constexpr bool has_qktv_remainder = qktv_remainder_h != 0;
        static_assert(Sq_chunk_t >= qktv_h, "Sq_chunk_t must be at least qktv_h");
        static_assert(vDHt % qktv_subblock_w == 0, "vDHt must be evenly divisible by qktv_subblock_w");
        static_assert(qktv_h * qktv_subblock_w <= dst_size, "qktv subblock must fit in dest register file");
        constexpr uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_h;
        constexpr bool qktv_first_group_reads_inplace_row = (q_num_subblocks - 1) * qkt_subblock_h < qktv_h;
        constexpr uint32_t qktv_v_num_subblocks = vDHt / qktv_subblock_w;
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * vDHt;
        constexpr uint32_t qktv_in0_row_tiles = qktv_h * KT_stride;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_row_tiles;

        const uint32_t out_cb = cur.out;

        CircularBuffer(out_cb).reserve_back(qktv_output_num_tiles);

        // q_subblock 0: drain last row's sub_exp in-place + first QKT@V matmul (split drain)
        {
            CP_PROF_SCOPE(CP_ST_DRAIN);
            const uint32_t matmul_inner = actual_sbw;
            for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {
                cp_sub_exp_block_bcast_cols<scale_fp32>(
                    cb_qkt_im,
                    cur.max,
                    cur.sum,
                    KT_stride,
                    q_num_subblocks - 1,
                    kt_sub * actual_sbw,
                    qkt_subblock_h,
                    actual_sbw);
                if constexpr (qktv_first_group_reads_inplace_row) {
                    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                }
                if (kt_sub == 0) {
                    CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
                    CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * vDHt);
                }
                if (kt_sub > 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }

                {
                    uint32_t v_index_offset = 0;
                    sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                        out_cb, out_cb);
                    sdpa_pv_mm_setup(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h, KT_stride);
                    configure_row_pack_width(out_cb, qktv_subblock_w);
                    if constexpr (qktv_first_group_reads_inplace_row) {
                        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                    }
                    for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                        const uint32_t qktv_in1_index = kt_sub * matmul_inner * vDHt + v_index_offset;
                        cp_blocked_matmul_and_pack<false, vDHt, vDHt, cp_skip_pv_mm>(
                            cb_qkt_im,
                            cb_v_in,
                            out_cb,
                            qktv_in0_index_offset + kt_sub * matmul_inner,
                            qktv_in1_index,
                            0,
                            v_subblock * qktv_subblock_w,
                            qktv_subblock_w,
                            qktv_h,
                            matmul_inner,
                            KT_stride,
                            /*skip_pack_configure=*/true);
                        v_index_offset += qktv_subblock_w;
                    }
                    sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                }

                if (kt_sub > 0) {
                    PACK((llk_pack_reconfig_l1_acc(0)));
                }
            }
            qktv_in0_index_offset += qktv_h * KT_stride;
            qktv_in0_wait_tiles += qktv_in0_row_tiles;
        }

        // Pack→unpack barrier between Phase 2's q_sub=0 drain and the main V-matmul loop.
        // Late form: the drain wrote only the last row group in place, and with whole row groups only the last
        // main-loop P@V reads it, so the unpack side takes the token right before that matmul instead of
        // stalling the first one until the drain's packs land (stock form otherwise).
        constexpr uint32_t total_v_row_groups = qktv_q_num_subblocks + (has_qktv_remainder ? 1 : 0);
        constexpr bool late_barrier = !has_qktv_remainder && qktv_h == qkt_subblock_h && total_v_row_groups >= 2;
        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
        if constexpr (!late_barrier) {
            UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
            UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
        }

        auto normalize_row = [&](uint32_t& pushed, uint32_t sbh) {
            CP_PROF_SCOPE(CP_ST_NORM);
            CircularBuffer(cur.sum).push_back(sbh);
            CircularBuffer(out_cb).push_back(sbh * vDHt);
            normalize_row_streaming<
                false,
                vDHt,
                dst_size,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                scale_fp32,
                false,
                INVALID_CB>(cur.sum, out_cb, sbh, cur.max, 0);
            pushed++;
        };

        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, uint32_t sbh) {
            CP_PROF_SCOPE(CP_ST_SALAD);
            PACK((llk_pack_reconfig_l1_acc(1)));
#ifndef CP_SKIP_SALAD
            if constexpr (has_qktv_remainder) {
                if (sbh == qktv_remainder_h) {
                    salad_correct_fused<qktv_remainder_h, vDHt, dst_size>(
                        prev.out, prev.sum, cb_exp_max_diff, out_cb, cur.sum, 0, salad_row, w_salad);
                } else {
                    salad_correct_fused<qktv_h, vDHt, dst_size>(
                        prev.out, prev.sum, cb_exp_max_diff, out_cb, cur.sum, 0, salad_row, w_salad);
                }
            } else {
                salad_correct_fused<qktv_h, vDHt, dst_size>(
                    prev.out, prev.sum, cb_exp_max_diff, out_cb, cur.sum, 0, salad_row, w_salad);
            }
#else
            CircularBuffer(prev.out).wait_front(sbh * vDHt);
            CircularBuffer(prev.sum).wait_front((salad_row + 1) * sbh);
            CircularBuffer(cb_exp_max_diff).wait_front(sbh);
#endif
            CircularBuffer(cb_exp_max_diff).pop_front(sbh);
            CircularBuffer(prev.out).pop_front(sbh * vDHt);
            PACK((llk_pack_reconfig_l1_acc(0)));
        };

        auto salad_exp = [&](uint32_t row, uint32_t h) {
            CP_PROF_SCOPE(CP_ST_SALAD);
            CircularBuffer(cb_exp_max_diff).reserve_back(h);
            sub_exp_first_col_blocks<false, scale_fp32>(prev.max, cur.max, cb_exp_max_diff, row, h);
            CircularBuffer(cb_exp_max_diff).push_back(h);
        };

        exp_packthread_tile_init<EXP_APPROX_MODE>();
        for (uint32_t q_subblock = 1; q_subblock < total_v_row_groups; ++q_subblock) {
            const bool is_remainder_iter = has_qktv_remainder && (q_subblock == qktv_q_num_subblocks);
            const uint32_t cur_h = is_remainder_iter ? qktv_remainder_h : qktv_h;
            uint32_t salad_row = q_subblock - 1;
            uint32_t w_salad = salad_row - pushed_rows;
            uint32_t w_q =
                is_remainder_iter ? (qktv_q_num_subblocks - pushed_rows) * qktv_h : (q_subblock - pushed_rows);

            if (!is_first_iter) {
                salad_exp(salad_row, qktv_h);
            }

            {
                CP_PROF_SCOPE(CP_ST_PV);
                if (is_remainder_iter) {
                    CircularBuffer(cb_qkt_im).wait_front(Sq_chunk_t * KT_stride);
                } else {
                    CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
                }
                if constexpr (late_barrier) {
                    if (q_subblock == total_v_row_groups - 1) {
                        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                    }
                }
                uint32_t v_index_offset = 0;
                sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                    out_cb, out_cb);
                sdpa_pv_mm_setup(cb_qkt_im, cb_v_in, false, qktv_subblock_w, cur_h, KT_stride);
                configure_row_pack_width(out_cb, qktv_subblock_w);
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    cp_blocked_matmul_and_pack<false, vDHt, vDHt, cp_skip_pv_mm>(
                        cb_qkt_im,
                        cb_v_in,
                        out_cb,
                        qktv_in0_index_offset,
                        v_index_offset,
                        w_q,
                        v_subblock * qktv_subblock_w,
                        qktv_subblock_w,
                        cur_h,
                        active_Sk,
                        KT_stride,
                        /*skip_pack_configure=*/true);
                    v_index_offset += qktv_subblock_w;
                }
                sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
            }

            if (!is_first_iter) {
                if (q_subblock == total_v_row_groups - 1) {
                    constexpr uint32_t drain_h = has_qktv_remainder ? qktv_remainder_h : qktv_h;
                    const uint32_t drain_salad_row =
                        has_qktv_remainder ? (qktv_q_num_subblocks * qktv_h) : (qktv_q_num_subblocks - 1);

                    salad_exp(drain_salad_row, drain_h);

                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(qktv_h);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt);
                        pushed_rows++;
                    }

                    const uint32_t drain_w = has_qktv_remainder ? ((qktv_q_num_subblocks - pushed_rows) * qktv_h)
                                                                : (qktv_q_num_subblocks - 1 - pushed_rows);
                    salad_correct_row(drain_salad_row, drain_w, drain_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(drain_h);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt);
                        pushed_rows++;
                    }
                } else {
                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(qktv_h);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt);
                        pushed_rows++;
                    }
                }
            } else if (is_last_iter) {
                normalize_row(pushed_rows, qktv_h);
            } else {
                CircularBuffer(cur.sum).push_back(qktv_h);
                CircularBuffer(out_cb).push_back(qktv_h * vDHt);
                pushed_rows++;
            }

            qktv_in0_index_offset += cur_h * KT_stride;
            qktv_in0_wait_tiles += cur_h * KT_stride;
        }

        // Pipeline drain: SALAD for the last group
        {
            constexpr uint32_t drain_h = has_qktv_remainder ? qktv_remainder_h : qktv_h;
            if constexpr (total_v_row_groups == 1) {
                if (!is_first_iter) {
                    constexpr uint32_t drain_salad_row = 0;
                    salad_exp(drain_salad_row, drain_h);
                    salad_correct_row(drain_salad_row, 0, drain_h);
                }
                if (is_last_iter) {
                    normalize_row(pushed_rows, drain_h);
                } else {
                    CircularBuffer(cur.sum).push_back(drain_h);
                    CircularBuffer(out_cb).push_back(drain_h * vDHt);
                    pushed_rows++;
                }
            } else {
                if (is_first_iter) {
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(drain_h);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt);
                        pushed_rows++;
                    }
                }
            }
        }

        CircularBuffer(cb_v_in).pop_front(KT_stride * vDHt);
        CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT_stride);
    }
}
