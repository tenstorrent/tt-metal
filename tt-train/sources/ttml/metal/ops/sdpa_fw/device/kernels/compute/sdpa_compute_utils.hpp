// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/sdpa_compute_utils_common.hpp"
#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h"
#endif

constexpr uint32_t onetile = 1U;

// Round `a` up to the nearest multiple of `b`. Uses the bitwise fast path when
// `b` is a power of two and the standard divmul form otherwise.
inline constexpr uint32_t round_up(uint32_t a, uint32_t b) {
    if ((b & (b - 1U)) == 0U) {
        return (a + (b - 1U)) & -b;
    }
    return ((a + b - 1U) / b) * b;
}

// SFPU intrinsics for first-column-only operations.
// Column vectors (from row-reduce) only have meaningful data in column 0,
// so we process 4 SFPU iterations (half-face) instead of the standard 8,
// saving ~75% of SFPU cycles.
#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void recip_tile_first_column(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sdpa_fw_recip_first_column,
        (is_fp32_dest_acc_en),
        idst,
        VectorMode::C);
}
#endif  // TRISC_MATH

// Apply an attention mask to a Q@K^T score tile already sitting in DST register `register_idx`.
//
// The input `attn_mask` tile holds 1.0 at valid (keep) positions and 0.0 at masked (drop)
// positions. The function transforms it to (0.0, -1e9) in-place and adds it to the score
// tile so that masked positions become large negative values that vanish under softmax.
//
// Preconditions:
//   - The DST register buffer must be in acquired state via *acquire_dst*.
//   - `cb_attn_mask` must have at least `mask_tile_idx + 1` tiles fronted.
//
// `mask_tile_idx` selects which mask tile in `cb_attn_mask` to apply: the host-provided
// mask is staged row-major per step, so callers pass `row * Sk_chunk_t + n`.
//
// Scaling by the SDPA scaler factor is NOT done here; it is deferred to after the
// row-max subtraction for better numerical precision.
void apply_mask_on_reg(
    uint32_t register_idx,
    uint32_t cb_attn_mask,
    uint32_t minus_one_bits,
    uint32_t custom_inf_bits,
    uint32_t mask_tile_idx = 0U) {
    const uint32_t mask_register = register_idx + 1U;  // mask register should be next to data register
    cb_wait_front(cb_attn_mask, mask_tile_idx + 1U);
    copy_init(cb_attn_mask);
    copy_tile(
        cb_attn_mask,
        /* tile_idx */ mask_tile_idx,
        /* register idx */ mask_register);

    // Apply the attention mask to Q @ K^T scores:
    // masked positions receive 0.0, unmasked positions remain unchanged
    mask_tile_init();
    mask_tile(register_idx, mask_register);

    // Transform mask: 1/0 → 0/-1 → 0/-1e9F, then add to scores.
    // Scaling is NOT applied here — it is deferred to after max-subtraction
    // for better numerical precision (avoids scale * large_raw_score in BF16).
    binop_with_scalar_tile_init();
    add_unary_tile(mask_register, minus_one_bits);   // subtract 1.0 from mask, so it becomes 0.0 and -1.0
    mul_unary_tile(mask_register, custom_inf_bits);  // multiply by 1e9F to transform mask to 0.0 and -1e9F

    // Add mask to scaled matmul result:
    // masked positions receive large negative values (will be 0.0 after softmax),
    // unmasked positions remain unchanged
    add_binary_tile_init();
    add_binary_tile(register_idx, mask_register, register_idx);
}

// ---------------------------------------------------------------------------------------------
// Per-row softmax state helpers.
//
// The compute kernel handles Sq_chunk_t query tile-rows per Q chunk. The score CB holds the
// Sq_chunk_t x Sk_chunk_t tiles of the current K/V step row-major (row r, K tile n at
// r * Sk_chunk_t + n), and every running statistic CB (max, exp-sum, exp-max-diff) is
// Sq_chunk_t tiles deep with row r in slot r. The helpers below work on one row; the caller
// owns the reserve/push of the row-deep CBs so a whole chunk's rows share one push.
// ---------------------------------------------------------------------------------------------

// TODO: replace FPU reduce_tile(MAX) + sub_tiles_bcast_cols with SFPU equivalents once LLK
// provides SFPU row-reduce-max and sub_bcast_col. This will allow the full softmax (max, sub,
// exp, sum, reciprocal) to stay in DST at FP32 without pack/unpack round-trips through the CB.
//
// Reduces over the `Sk_chunk_t` score tiles of row `row`, accumulating the max into a single
// DST register, optionally folds in the running max of the row, and packs the result into
// slot `row` of cb_cur_max. Caller: cb_attention_weights fronted, cb_cur_max reserved
// (Sq_chunk_t tiles), cb_prev_max fronted when do_eltwise_max.
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t Sk_chunk_t = 1U>
void update_cur_row_max_value(
    uint32_t cb_attention_weights,
    uint32_t cb_identity_scaler,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    bool do_eltwise_max,
    uint32_t row) {
    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1U;
    reconfig_data_format(cb_attention_weights, cb_identity_scaler);
    reduce_init<pool_type, reduce_dim>(cb_attention_weights, cb_identity_scaler, cb_cur_max);
    tile_regs_acquire();
    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
        reduce_tile<pool_type, reduce_dim>(
            cb_attention_weights,
            cb_identity_scaler,
            /* in0 tile_idx */ row * Sk_chunk_t + n,
            /* in1 tile_idx */ 0,
            /* dst_reg_idx */ reduce_dst_idx);
    }
    reduce_uninit();

    if (do_eltwise_max) {
        reconfig_data_format_srca(cb_attention_weights, cb_prev_max);
        copy_init(cb_prev_max);
        copy_tile(cb_prev_max, /* tile_idx */ row, /* register idx */ prev_max_dst_idx);

        // find max value between current max and previous max
        binary_max_tile_init();
        binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx, VectorMode::C);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_max);
    pack_tile</* out_of_order */ true>(reduce_dst_idx, cb_cur_max, /* offset */ row);
    tile_regs_release();
}

// Apply exp(scale * (score - max)) in place on the Sk_chunk_t score tiles of row `row`, and
// build the row's partial sum (over the Sk_chunk_t tiles) in slot `row` of cb_cur_exp_sum.
//
// The partial sum is built directly in L1 via the packer's L1-accumulate path: the first exp
// tile overwrites the slot, subsequent tiles add to it. This avoids an extra DST cycle (and
// an extra row-reduce CB) just to fold the per-tile partial sums together.
//
// DST budget: `sm_block_size` registers are held live across sub_tiles_bcast_cols + exp_tile +
// (two) pack passes, so the row is processed in Sk_chunk_t / sm_block_size DST cycles.
//
// Caller: cb_attention_weights popped and re-reserved on the same L1 slots (single-buffered
// CB wraps back), cb_cur_max fronted, cb_cur_exp_sum reserved (Sq_chunk_t tiles).
template <uint32_t scaler_fp32, uint32_t scaler_bf16, uint32_t Sk_chunk_t, uint32_t sm_block_size>
void apply_exp_inplace_and_find_exp_sum(
    uint32_t cb_attention_weights, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum, uint32_t row) {
    static_assert(Sk_chunk_t % sm_block_size == 0U, "sm_block_size must divide Sk_chunk_t");

    reconfig_data_format(cb_attention_weights, cb_cur_max);
    sub_bcast_cols_init(cb_attention_weights, cb_cur_max);

    for (uint32_t n0 = 0; n0 < Sk_chunk_t; n0 += sm_block_size) {
        // Fused scale+exp: compute exp(scale * (score - max)) in a single SFPU pass.
        // sdpa_exp_tile_scaled dispatches to the arch-appropriate path (WH sfpi mul or BH LREG12 fold).
        tile_regs_acquire();
        for (uint32_t i = 0; i < sm_block_size; ++i) {
            sub_tiles_bcast_cols(
                cb_attention_weights,
                cb_cur_max,
                /* in0 tile_idx */ row * Sk_chunk_t + n0 + i,
                /* in1 tile_idx */ row,
                /* dst_reg_idx */ i);
            sdpa_exp_tile_scaled<scaler_fp32, scaler_bf16>(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        // Write the exp tiles back over their own score slots.
        pack_reconfig_data_format(cb_attention_weights);
        pack_reconfig_l1_acc(false);
        for (uint32_t i = 0; i < sm_block_size; ++i) {
            pack_tile</* out_of_order */ true>(i, cb_attention_weights, /* offset */ row * Sk_chunk_t + n0 + i);
        }

        // Row partial sum in cb_cur_exp_sum[row]: the very first tile of the row overwrites
        // the slot (L1-acc off); every other tile L1-accumulates onto it.
        pack_reconfig_data_format(cb_cur_exp_sum);
        uint32_t i = 0;
        if (n0 == 0U) {
            pack_tile</* out_of_order */ true>(/* dst */ 0, cb_cur_exp_sum, /* offset */ row);
            i = 1U;
        }
        if (i < sm_block_size) {
            pack_reconfig_l1_acc(true);
            for (; i < sm_block_size; ++i) {
                pack_tile</* out_of_order */ true>(/* dst */ i, cb_cur_exp_sum, /* offset */ row);
            }
            pack_reconfig_l1_acc(false);
        }
        tile_regs_release();
    }
}

// Computes (attention_weights @ V) for the whole Q chunk: attention_weights is Sq_chunk_t rows
// of Sk_chunk_t score tiles (row-major), V is the Sk_chunk_t × Wt block of this step. The
// K-dimension reduction (over Sk_chunk_t) is accumulated inside DST across the inner loop.
//
// Each (block_h × block_w) rectangle of output tiles is produced by one matmul_block call per
// K step (ct_dim=block_w, rt_dim=block_h, kt_dim=Sk_chunk_t). matmul_block walks in0 rows at
// stride kt_dim, which is exactly the row-major score layout, and V is row-major in cb_value
// (seq outer, feat inner) so the `block_w` V tiles consumed per K step at a fixed seq are
// contiguous — no transpose read needed (contrast with QK^T where K must be col-major).
//
// `block_h * block_w` must fit DST. matmul_block writes its output tiles directly into DST
// and needs no scratch register, so callers can use the full `dst_size` (4 for fp32_acc,
// 8 for bf16_acc) — distinct from the SFPU block_size used elsewhere that needs `+1` for
// `unary_bcast` scratch and is therefore capped at `dst_size - 1`.
template <uint32_t Sk_chunk_t, uint32_t Sq_chunk_t>
void matmul_qk_by_v(
    uint32_t Wt,
    uint32_t block_h,
    uint32_t block_w,
    uint32_t cb_attention_weights,
    uint32_t cb_value,
    uint32_t cb_cur_mm_out) {
    cb_wait_front(cb_attention_weights, Sq_chunk_t * Sk_chunk_t);
    cb_wait_front(cb_value, Sk_chunk_t * Wt);
    cb_reserve_back(cb_cur_mm_out, Sq_chunk_t * Wt);

    // matmul maps: in0(attention_weights)→SrcB, in1(value)→SrcA
    reconfig_data_format(cb_value, cb_attention_weights);
    pack_reconfig_data_format(cb_cur_mm_out);
    matmul_block_init(
        cb_attention_weights,
        cb_value,
        /* transpose */ 0,
        /* ct_dim */ block_w,
        /* rt_dim */ block_h,
        /* kt_dim */ Sk_chunk_t);

    for (uint32_t qb = 0; qb < Sq_chunk_t; qb += block_h) {
        for (uint32_t vb = 0; vb < Wt; vb += block_w) {
            tile_regs_acquire();
            // K-reduction: at each k step, matmul_block writes block_h x block_w output tiles
            // into DST[0..block_h*block_w-1], accumulating across k steps.
            for (uint32_t k_in_chunk = 0; k_in_chunk < Sk_chunk_t; ++k_in_chunk) {
                matmul_block(
                    cb_attention_weights,
                    cb_value,
                    /* in0 (A) tile_idx */ qb * Sk_chunk_t + k_in_chunk,
                    /* in1 (B) tile_idx */ k_in_chunk * Wt + vb,
                    /* dst_idx */ 0,
                    /* transpose */ 0,
                    /* ct_dim */ block_w,
                    /* rt_dim */ block_h,
                    /* kt_dim */ Sk_chunk_t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < block_h; ++i) {
                for (uint32_t j = 0; j < block_w; ++j) {
                    pack_tile</* out_of_order */ true>(i * block_w + j, cb_cur_mm_out, /* offset */ (qb + i) * Wt + vb + j);
                }
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_cur_mm_out, Sq_chunk_t * Wt);
}

// exp(scale * (prev_max - cur_max)) for row `row`, packed into slot `row` of cb_exp_max_diff.
// Caller: both max CBs fronted, cb_exp_max_diff reserved (Sq_chunk_t tiles).
template <uint32_t scaler_fp32, uint32_t scaler_bf16>
void update_exp_max_diff(uint32_t cb_prev_max_value, uint32_t cb_cur_max_value, uint32_t cb_exp_max_diff, uint32_t row) {
    const uint32_t exp_max_diff_dst_idx = 0;
    reconfig_data_format(cb_prev_max_value, cb_cur_max_value);
    tile_regs_acquire();
    sub_init(cb_prev_max_value, cb_cur_max_value);
    sub_tiles(
        cb_prev_max_value,
        cb_cur_max_value,
        /* tile_idx */ row,
        /* tile_idx */ row,
        /* dst_reg_idx */ exp_max_diff_dst_idx);

    // First-column scaled exp: exp(scale * (prev_max - cur_max)).
    // Both max values are column vectors, so the result is a column vector —
    // only column 0 has data. Process 4× fewer SFPU iterations than full-tile exp.
    sdpa_exp_tile_first_column_scaled<scaler_fp32, scaler_bf16>(exp_max_diff_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_exp_max_diff);
    pack_tile</* out_of_order */ true>(exp_max_diff_dst_idx, cb_exp_max_diff, /* offset */ row);
    tile_regs_release();
}

// cur_sum[row] = prev_sum[row] * exp_max_diff[row] + cur_sum[row], written back over slot
// `row`. Caller: prev_sum and exp_max_diff fronted; cb_cur_sum_exp popped and re-reserved on
// the same L1 slots (the CB is exactly Sq_chunk_t deep, so pop+reserve wraps back).
void update_cur_exp_sum_inplace(
    uint32_t cb_prev_sum_exp, uint32_t cb_cur_sum_exp, uint32_t cb_exp_max_diff, uint32_t row) {
    const uint32_t exp_sum_dst_idx = 0;
    reconfig_data_format(cb_prev_sum_exp, cb_exp_max_diff);  // reconfig data format to precise
    mul_bcast_cols_init(cb_prev_sum_exp, cb_exp_max_diff);
    tile_regs_acquire();
    // multiply previous exp sum with exp_max_diff
    mul_tiles_bcast_cols(cb_prev_sum_exp, cb_exp_max_diff, row, row, exp_sum_dst_idx);

    // copy current sum exp to next register
    copy_init(cb_cur_sum_exp);
    copy_tile(cb_cur_sum_exp, /* tile_idx */ row, /* register idx */ exp_sum_dst_idx + 1U);

    // add to updated previous exp sum with current exp sum
    add_binary_tile_init();
    add_binary_tile(exp_sum_dst_idx, exp_sum_dst_idx + 1U, exp_sum_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_sum_exp);
    pack_tile</* out_of_order */ true>(exp_sum_dst_idx, cb_cur_sum_exp, /* offset */ row);
    tile_regs_release();
}

// Lightweight init for unary_bcast<COL> with B2D datacopy path.
// Reads column 0 from a CB tile and replicates it to all 32 columns in a DST register.
// Only reprograms UNPACK and MATH MOPs — safe inside a tile_regs_acquire block.
// Uses reconfig_data_format_srcb so SrcA format register is preserved for the caller.
void init_unary_bcast_col(uint32_t cb_col_vec) {
    reconfig_data_format_srcb(cb_col_vec);
    UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, cb_col_vec)));
    MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, DST_ACCUM_MODE, BroadcastType::COL>(
        cb_col_vec)));
}

// Scale every row's prev_mm_out tiles by that row's column-broadcast factor and L1-accumulate
// onto cur_mm_out (Sq_chunk_t * Wt tiles, row r at r * Wt). Uses the SFPU path: prev_mm_out
// is loaded via UnpackToDestFp32 (full FP32 in DST), the factor via unary_bcast<COL> (col 0
// broadcast to all cols), and mul_binary_tile does the FP32 multiply.
template <uint32_t Sq_chunk_t>
void update_cur_mm_out(
    uint32_t Wt, uint32_t block_size, uint32_t cb_prev_mm_out, uint32_t cb_cur_mm_out, uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_prev_mm_out, Sq_chunk_t * Wt);
    cb_wait_front(cb_cur_mm_out, Sq_chunk_t * Wt);
    cb_wait_front(cb_exp_max_diff, Sq_chunk_t);

    pack_reconfig_data_format(cb_cur_mm_out);
    pack_reconfig_l1_acc(true);

    for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
        for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
            tile_regs_acquire();

            // Load prev_mm_out tiles to DST via UnpackToDestFp32 (full FP32 precision)
            reconfig_data_format(cb_prev_mm_out, cb_prev_mm_out);
            copy_init(cb_prev_mm_out);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                copy_tile(cb_prev_mm_out, row * Wt + tile_idx + block_idx, block_idx);
            }

            // Load this row's exp_max_diff with column broadcast to DST[block_size]
            init_unary_bcast_col(cb_exp_max_diff);
            unary_bcast<BroadcastType::COL>(cb_exp_max_diff, row, block_size);

            // SFPU element-wise multiply: DST[i] = DST[i] * DST[block_size]
            mul_binary_tile_init();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_binary_tile(block_idx, block_size, block_idx);
            }

            tile_regs_commit();
            tile_regs_wait();
            // cur_mm_out is full and exactly Sq_chunk_t * Wt deep, so its write pointer has
            // wrapped onto the fronted tiles: the L1-acc pack adds onto the live slots.
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                pack_tile</* out_of_order */ true>(block_idx, cb_cur_mm_out, /* offset */ row * Wt + tile_idx + block_idx);
            }
            tile_regs_release();
        }
    }
    pack_reconfig_l1_acc(false);

    // Re-front the accumulated tiles (bookkeeping only; the data stayed in place).
    cb_pop_front(cb_cur_mm_out, Sq_chunk_t * Wt);
    cb_reserve_back(cb_cur_mm_out, Sq_chunk_t * Wt);
    cb_push_back(cb_cur_mm_out, Sq_chunk_t * Wt);
}

// Row-reduce tile `row` of cb_in_idx in place via matmul with a column-of-ones tile.
// Result is a column vector (value in column 0 of each row).
// Caller: cb_in_idx popped and re-reserved on the same slots, cb_matmul_reduce fronted.
template <uint32_t cb_identity_scaler, uint32_t cb_matmul_reduce>
void row_reduce_tile_inplace(uint32_t cb_in_idx, uint32_t row) {
    const uint32_t reduce_dst_idx = 0;

    reconfig_data_format(cb_matmul_reduce, cb_in_idx);
    tile_regs_acquire();

    matmul_init(cb_in_idx, cb_matmul_reduce, 0);
    matmul_tiles(cb_in_idx, cb_matmul_reduce, /* tile_idx */ row, /* tile_idx */ 0, reduce_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_in_idx);
    pack_tile</* out_of_order */ true>(reduce_dst_idx, cb_in_idx, /* offset */ row);
    tile_regs_release();
}

// Apply reciprocal to tile `row` of cb_in_idx in place.
// Caller: cb_in_idx popped and re-reserved on the same slots.
void recip_tile_inplace(uint32_t cb_in_idx, uint32_t row) {
    const uint32_t dst_idx = 0;
    tile_regs_acquire();
    reconfig_data_format(cb_in_idx, cb_in_idx);
    copy_init(cb_in_idx);
    copy_tile(cb_in_idx, /* tile_idx */ row, dst_idx);
    recip_tile_init();
    MATH((recip_tile_first_column(dst_idx)));
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_in_idx);
    pack_tile</* out_of_order */ true>(dst_idx, cb_in_idx, /* offset */ row);
    tile_regs_release();
}

// Compute logsumexp = scale*max + log(sum_exp) for row `row` and pack it into slot `row` of
// cb_out_lse with column-0 masking. Max is unscaled (raw QK^T max), so we multiply by scale
// to get the correct LSE. Reads cb_sum_exp (row-reduced) and cb_max WITHOUT consuming them.
// cb_mask_tile should contain 1.0 in column 0 and 0.0 elsewhere (cb_matmul_reduce).
// The lse tile is computed entirely in FP32 DST registers for maximum precision.
// Caller: cb_out_lse reserved (Sq_chunk_t tiles).
void compute_and_pack_lse(
    uint32_t cb_sum_exp,
    uint32_t cb_max,
    uint32_t cb_out_lse,
    uint32_t cb_mask_tile,
    uint32_t scaler_bits,
    uint32_t row) {
    // mask_tile HW constraint: mask register must be idst_data + 1
    const uint32_t lse_reg = 0;
    const uint32_t mask_reg = 1U;
    const uint32_t max_reg = 2U;

    tile_regs_acquire();

    reconfig_data_format(cb_sum_exp, cb_sum_exp);
    copy_init(cb_sum_exp);
    copy_tile(cb_sum_exp, /* tile_idx */ row, lse_reg);

    log_tile_init</* fast_and_approx */ false>();
    log_tile</* fast_and_approx */ false>(lse_reg);

    reconfig_data_format_srca(/* old_cb_idx */ cb_sum_exp, /* new_cb_idx */ cb_max);
    copy_init(/* new_cb_idx */ cb_max);
    copy_tile(cb_max, /* tile_idx */ row, max_reg);

    // lse = scale * max + log(sum_exp)
    binop_with_scalar_tile_init();
    mul_unary_tile(max_reg, scaler_bits);

    add_binary_tile_init();
    add_binary_tile(max_reg, lse_reg, lse_reg);

    reconfig_data_format_srca(/* old_cb_idx */ cb_max, /* new_cb_idx */ cb_mask_tile);
    copy_init(/* new_cb_idx */ cb_mask_tile);
    copy_tile(cb_mask_tile, /* tile_idx */ 0, mask_reg);

    mask_tile_init();
    mask_tile(lse_reg, mask_reg);

    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_out_lse);
    pack_tile</* out_of_order */ true>(lse_reg, cb_out_lse, /* offset */ row);
    tile_regs_release();
}

// Pack intermediate result with masking to ensure only column 0 has value, rest are zeros.
// cb_mask_tile should contain 1.0 in column 0 and 0.0 elsewhere (use cb_matmul_reduce).
// Input tile has reduced value in column 0 after row reduction.
// Output tile will have value only in column 0, all other columns zeroed out.
void pack_intermediate_result(
    uint32_t cb_in_idx, uint32_t cb_out_idx, uint32_t cb_mask_tile, uint32_t tiles_count = 1U) {
    cb_wait_front(cb_in_idx, tiles_count);
    cb_reserve_back(cb_out_idx, tiles_count);

    const uint32_t dst_idx = 0;

    for (uint32_t tile_idx = 0; tile_idx < tiles_count; ++tile_idx) {
        tile_regs_acquire();
        reconfig_data_format_srca(/* old_cb_idx */ cb_mask_tile, /* new_cb_idx */ cb_in_idx);
        copy_init(/* new_cb_idx */ cb_in_idx);
        copy_tile(cb_in_idx, /* tile_idx */ tile_idx, /* register idx */ dst_idx);

        reconfig_data_format_srca(/* old_cb_idx */ cb_in_idx, /* new_cb_idx */ cb_mask_tile);
        copy_init(/* new_cb_idx */ cb_mask_tile);
        copy_tile(cb_mask_tile, /* tile_idx */ 0, /* register idx */ dst_idx + 1U);

        mask_tile_init();
        mask_tile(dst_idx, dst_idx + 1U);

        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_out_idx);
        pack_tile(dst_idx, cb_out_idx);
        tile_regs_release();
    }

    cb_push_back(cb_out_idx, tiles_count);
}
