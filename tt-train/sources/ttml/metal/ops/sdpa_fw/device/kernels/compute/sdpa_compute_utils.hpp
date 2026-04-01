// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t onetile = 1U;

// now we have to multiply result by scaler factor and then apply mask
// we need to transform the attention mask for use in softmax:
// The input `attn_mask` contains 1.0 for valid (keep) positions and 0.0 for masked (drop) positions.
// To convert this into a format compatible with softmax masking:
//   - Subtract 1.0 from the mask, so values become 0.0 (keep) and -1.0 (mask).
//   - Multiply by a large negative value (e.g., 1e9F), resulting in 0.0 for valid entries and -inf for
//   masked ones.
// This way, after applying softmax, masked positions will effectively become zero,
// and only the unmasked positions will retain meaningful attention weights
void apply_mask_on_reg(
    uint32_t register_idx, uint32_t cb_attn_mask, uint32_t minus_one_bits, uint32_t custom_inf_bits) {
    /* The DST register buffer must be in acquired state via *acquire_dst* call.*/

    const uint32_t mask_register = register_idx + 1U;  // mask register should be next to data register
    cb_wait_front(cb_attn_mask, onetile);
    copy_tile_init(cb_attn_mask);
    copy_tile(
        cb_attn_mask,
        /* tile_idx */ 0,
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

    add_binary_tile_init();
    add_binary_tile(register_idx, mask_register, register_idx);
}

template <PoolType pool_type, ReduceDim reduce_dim>
void update_cur_row_max_value(
    uint32_t cb_attention_weights,
    uint32_t cb_identity_scaler,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    bool do_eltwise_max = false) {
    cb_wait_front(cb_attention_weights, onetile);

    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1U;
    reconfig_data_format(cb_attention_weights, cb_identity_scaler);
    reduce_init<pool_type, reduce_dim>(cb_attention_weights, cb_identity_scaler, cb_cur_max);
    tile_regs_acquire();
    reduce_tile<pool_type, reduce_dim>(
        cb_attention_weights, cb_identity_scaler, /* tile_idx */ 0, /* tile_idx */ 0, /* dst_reg_idx */ reduce_dst_idx);
    reduce_uninit();

    if (do_eltwise_max) {
        cb_wait_front(cb_prev_max, onetile);
        copy_tile_to_dst_init_short_with_dt(cb_attention_weights, cb_prev_max);
        copy_tile(cb_prev_max, /* tile_idx */ 0, /* register idx */ prev_max_dst_idx);

        // find max value between current max and previous max
        binary_max_tile_init();
        binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx, static_cast<int>(VectorMode::C));
    }
    tile_regs_commit();

    cb_reserve_back(cb_cur_max, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_max);
    pack_tile(reduce_dst_idx, cb_cur_max);
    tile_regs_release();
    cb_push_back(cb_cur_max, onetile);
}

/* We process data by one tile, because we read only one row of K
 * Maybe we can read two rows of K and V and then process data by subblocks*/
void apply_exp_inplace_and_find_exp_sum(
    uint32_t cb_attention_weights, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum, uint32_t scaler_bits) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_cur_max, onetile);

    const uint32_t exp_dst_idx = 0;
    reconfig_data_format(cb_attention_weights, cb_cur_max);
    sub_bcast_cols_init_short(cb_attention_weights, cb_cur_max);
    tile_regs_acquire();
    sub_tiles_bcast_cols(
        cb_attention_weights, cb_cur_max, /* tile_idx */ 0, /* tile_idx */ 0, /* dst_reg_idx */ exp_dst_idx);

    // Apply scale after max-subtraction: exp(scale * (score - max)).
    // This gives better precision than scaling raw scores before finding max.
    binop_with_scalar_tile_init();
    mul_unary_tile(exp_dst_idx, scaler_bits);

    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(exp_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    // update current qk matmul result with exp values
    cb_pop_front(cb_attention_weights, onetile);
    cb_reserve_back(cb_attention_weights, onetile);
    pack_reconfig_data_format(cb_attention_weights);
    pack_tile(exp_dst_idx, cb_attention_weights);

    /* update current exp sum with exp values
     * at the moment we pack one tile here
     * but we can use L1 accumlator to pack more tiles
     * in case we will be able to read more then one row of K and V
     */
    cb_reserve_back(cb_cur_exp_sum, onetile);
    pack_reconfig_data_format(cb_cur_exp_sum);
    pack_tile(exp_dst_idx, cb_cur_exp_sum);
    tile_regs_release();

    cb_push_back(cb_attention_weights, onetile);
    cb_push_back(cb_cur_exp_sum, onetile);
}

void matmul_qk_by_v(
    uint32_t Wt, uint32_t block_size, uint32_t cb_attention_weights, uint32_t cb_value, uint32_t cb_cur_mm_out) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_value, Wt);
    cb_reserve_back(cb_cur_mm_out, Wt);

    mm_init_short(cb_attention_weights, cb_value, /* transpose */ 0);
    pack_reconfig_data_format(cb_cur_mm_out);
    // matmul maps: in0(attention_weights)→SrcB, in1(value)→SrcA
    reconfig_data_format(cb_value, cb_attention_weights);
    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_attention_weights,
                cb_value,
                /* tile_idx */ 0,
                /* tile_idx */ tile_idx + block_idx,
                block_idx);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_cur_mm_out);
        }
        tile_regs_release();
    }
    cb_push_back(cb_cur_mm_out, Wt);
}

void update_exp_max_diff(
    uint32_t cb_prev_max_value, uint32_t cb_cur_max_value, uint32_t cb_exp_max_diff, uint32_t scaler_bits) {
    cb_wait_front(cb_prev_max_value, onetile);
    cb_wait_front(cb_cur_max_value, onetile);

    cb_reserve_back(cb_exp_max_diff, onetile);

    const uint32_t exp_max_diff_dst_idx = 0;
    reconfig_data_format(cb_prev_max_value, cb_cur_max_value);
    tile_regs_acquire();
    sub_tiles_init(cb_prev_max_value, cb_cur_max_value);
    sub_tiles(
        cb_prev_max_value,
        cb_cur_max_value,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* dst_reg_idx */ exp_max_diff_dst_idx);

    // Max values are unscaled, so correction factor is exp(scale * (prev_max - cur_max))
    binop_with_scalar_tile_init();
    mul_unary_tile(exp_max_diff_dst_idx, scaler_bits);

    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(exp_max_diff_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_exp_max_diff);
    pack_tile(exp_max_diff_dst_idx, cb_exp_max_diff);
    tile_regs_release();
    cb_push_back(cb_exp_max_diff, onetile);
}

void update_cur_exp_sum_inplace(uint32_t cb_prev_sum_exp, uint32_t cb_cur_sum_exp, uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_prev_sum_exp, onetile);
    cb_wait_front(cb_cur_sum_exp, onetile);
    cb_wait_front(cb_exp_max_diff, onetile);

    const uint32_t exp_sum_dst_idx = 0;
    mul_bcast_cols_init_short(cb_prev_sum_exp, cb_exp_max_diff);
    tile_regs_acquire();
    // multiply previous exp sum with exp_max_diff
    reconfig_data_format(cb_prev_sum_exp, cb_exp_max_diff);  // reconfig data format to precise
    mul_tiles_bcast_cols(cb_prev_sum_exp, cb_exp_max_diff, 0, 0, exp_sum_dst_idx);

    // copy current sum exp to next register
    copy_tile_init(cb_cur_sum_exp);
    copy_tile(cb_cur_sum_exp, /* tile_idx */ 0, /* register idx */ exp_sum_dst_idx + 1U);

    // add to updated previous exp sum with current exp sum
    add_binary_tile_init();
    add_binary_tile(exp_sum_dst_idx, exp_sum_dst_idx + 1U, exp_sum_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_cur_sum_exp, onetile);
    cb_reserve_back(cb_cur_sum_exp, onetile);
    pack_reconfig_data_format(cb_cur_sum_exp);
    pack_tile(exp_sum_dst_idx, cb_cur_sum_exp);
    tile_regs_release();
    cb_push_back(cb_cur_sum_exp, onetile);
}

// Lightweight init for unary_bcast<COL> with B2D datacopy path.
// Reads column 0 from a CB tile and replicates it to all 32 columns in a DST register.
// Only reprograms UNPACK and MATH MOPs — safe inside a tile_regs_acquire block.
void init_unary_bcast_col(uint32_t cb_prev, uint32_t cb_col_vec) {
    reconfig_data_format(cb_prev, cb_col_vec);
    UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, cb_col_vec)));
    MATH((llk_math_eltwise_unary_datacopy_init<B2D, DST_ACCUM_MODE, BroadcastType::COL>(cb_col_vec)));
}

// Scale prev_mm_out tiles by a column-broadcast factor and L1-accumulate onto cur_mm_out.
// Uses SFPU path: prev_mm_out is loaded via UnpackToDestFp32 (full FP32 in DST),
// the scale factor is loaded via unary_bcast<COL> (col 0 broadcast to all cols),
// and mul_binary_tile does FP32 SFPU multiply.
void update_cur_mm_out(
    uint32_t Wt, uint32_t block_size, uint32_t cb_prev_mm_out, uint32_t cb_cur_mm_out, uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_prev_mm_out, Wt);
    cb_wait_front(cb_cur_mm_out, Wt);
    cb_wait_front(cb_exp_max_diff, onetile);

    pack_reconfig_data_format(cb_cur_mm_out);
    pack_reconfig_l1_acc(true);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();

        // Load prev_mm_out tiles to DST via UnpackToDestFp32 (full FP32 precision)
        reconfig_data_format(cb_prev_mm_out, cb_prev_mm_out);
        copy_tile_init(cb_prev_mm_out);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_prev_mm_out, tile_idx + block_idx, block_idx);
        }

        // Load exp_max_diff with column broadcast to DST[block_size]
        init_unary_bcast_col(cb_prev_mm_out, cb_exp_max_diff);
        unary_bcast<BroadcastType::COL>(cb_exp_max_diff, 0, block_size);

        // SFPU element-wise multiply: DST[i] = DST[i] * DST[block_size]
        mul_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_binary_tile(block_idx, block_size, block_idx);
        }

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_cur_mm_out);
        }
        tile_regs_release();
    }
    pack_reconfig_l1_acc(false);

    cb_pop_front(cb_cur_mm_out, Wt);
    cb_reserve_back(cb_cur_mm_out, Wt);
    cb_push_back(cb_cur_mm_out, Wt);
}

// Row-reduce a tile in place via matmul with a column-of-ones tile.
// Result is a column vector (value in column 0 of each row).
template <uint32_t cb_identity_scaler, uint32_t cb_matmul_reduce>
void row_reduce_tile_inplace(uint32_t cb_in_idx) {
    cb_wait_front(cb_in_idx, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);

    const uint32_t reduce_dst_idx = 0;

    reconfig_data_format(cb_in_idx, cb_matmul_reduce);
    tile_regs_acquire();

    mm_init(cb_in_idx, cb_matmul_reduce, cb_identity_scaler, 0);
    matmul_tiles(cb_in_idx, cb_matmul_reduce, /* tile_idx */ 0, /* tile_idx */ 0, reduce_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_in_idx, onetile);
    cb_reserve_back(cb_in_idx, onetile);
    pack_reconfig_data_format(cb_in_idx);
    pack_tile(reduce_dst_idx, cb_in_idx);
    tile_regs_release();
    cb_push_back(cb_in_idx, onetile);
}

// Apply reciprocal to a tile in place.
void recip_tile_inplace(uint32_t cb_in_idx) {
    cb_wait_front(cb_in_idx, onetile);

    const uint32_t dst_idx = 0;
    tile_regs_acquire();
    reconfig_data_format(cb_in_idx, cb_in_idx);
    copy_tile_init(cb_in_idx);
    copy_tile(cb_in_idx, /* tile_idx */ 0, dst_idx);
    recip_tile_init();
    recip_tile(dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_in_idx, onetile);
    cb_reserve_back(cb_in_idx, onetile);
    pack_reconfig_data_format(cb_in_idx);
    pack_tile(dst_idx, cb_in_idx);
    tile_regs_release();
    cb_push_back(cb_in_idx, onetile);
}

// Compute logsumexp = scale*max + log(sum_exp) and pack to output CB with column-0 masking.
// Max is unscaled (raw QK^T max), so we multiply by scale to get the correct LSE.
// Reads from cb_sum_exp (row-reduced) and cb_max WITHOUT consuming them.
// cb_mask_tile should contain 1.0 in column 0 and 0.0 elsewhere (cb_matmul_reduce).
// The lse tile is computed entirely in FP32 DST registers for maximum precision.
void compute_and_pack_lse(
    uint32_t cb_sum_exp, uint32_t cb_max, uint32_t cb_out_lse, uint32_t cb_mask_tile, uint32_t scaler_bits) {
    // mask_tile HW constraint: mask register must be idst_data + 1
    const uint32_t lse_reg = 0;
    const uint32_t mask_reg = 1U;
    const uint32_t max_reg = 2U;

    tile_regs_acquire();

    reconfig_data_format(cb_sum_exp, cb_sum_exp);
    copy_tile_init(cb_sum_exp);
    copy_tile(cb_sum_exp, /* tile_idx */ 0, lse_reg);

    log_tile_init</* fast_and_approx */ false>();
    log_tile</* fast_and_approx */ false>(lse_reg);

    copy_tile_to_dst_init_short_with_dt(/* old_cb_idx */ cb_sum_exp, /* new_cb_idx */ cb_max);
    copy_tile(cb_max, /* tile_idx */ 0, max_reg);

    // lse = scale * max + log(sum_exp)
    binop_with_scalar_tile_init();
    mul_unary_tile(max_reg, scaler_bits);

    add_binary_tile_init();
    add_binary_tile(max_reg, lse_reg, lse_reg);

    copy_tile_to_dst_init_short_with_dt(/* old_cb_idx */ cb_max, /* new_cb_idx */ cb_mask_tile);
    copy_tile(cb_mask_tile, /* tile_idx */ 0, mask_reg);

    mask_tile_init();
    mask_tile(lse_reg, mask_reg);

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out_lse, onetile);
    pack_reconfig_data_format(cb_out_lse);
    pack_tile(lse_reg, cb_out_lse);
    tile_regs_release();
    cb_push_back(cb_out_lse, onetile);
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
        copy_tile_to_dst_init_short_with_dt(/* old_cb_idx */ cb_mask_tile, /* new_cb_idx */ cb_in_idx);
        copy_tile(cb_in_idx, /* tile_idx */ tile_idx, /* register idx */ dst_idx);

        copy_tile_to_dst_init_short_with_dt(/* old_cb_idx */ cb_in_idx, /* new_cb_idx */ cb_mask_tile);
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
