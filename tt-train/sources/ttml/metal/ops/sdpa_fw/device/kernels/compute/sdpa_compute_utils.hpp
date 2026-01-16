// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/softplus.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

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
    uint32_t register_idx,
    uint32_t cb_attn_mask,
    uint32_t scaler_bits,
    uint32_t minus_one_bits,
    uint32_t custom_inf_bits) {
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

    binop_with_scalar_tile_init();
    mul_unary_tile(register_idx, scaler_bits);       // multiply by scaler factor
    add_unary_tile(mask_register, minus_one_bits);   // subtract 1.0 from mask, so it becomes 0.0 and -1.0
    mul_unary_tile(mask_register, custom_inf_bits);  // multiply by 1e9F to transform mask to 0.0 and -1e9F

    // Add mask to scaled matmul result:
    // masked positions receive large negative values (will be 0.0 after softmax),
    // unmasked positions remain unchanged
    add_binary_tile_init();
    add_binary_tile(register_idx, mask_register, register_idx);
}

template <PoolType pool_type, ReduceDim reduce_dim>
void update_cur_row_max_value(
    uint32_t cb_qk_result,
    uint32_t cb_identity_scaler,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    bool do_eltwise_max = false) {
    cb_wait_front(cb_qk_result, onetile);

    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1U;
    reduce_init<pool_type, reduce_dim>(cb_qk_result, cb_identity_scaler, cb_cur_max);
    tile_regs_acquire();
    reduce_tile<pool_type, reduce_dim>(
        cb_qk_result, cb_identity_scaler, /* tile_idx */ 0, /* tile_idx */ 0, /* dst_reg_idx */ reduce_dst_idx);
    reduce_uninit();

    if (do_eltwise_max) {
        cb_wait_front(cb_prev_max, onetile);
        copy_tile_init(cb_prev_max);
        copy_tile(cb_prev_max, /* tile_idx */ 0, /* register idx */ prev_max_dst_idx);

        // find max value between current max and previous max
        max_tile_init();
        max_tile(reduce_dst_idx, prev_max_dst_idx, static_cast<int>(VectorMode::C));
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
void apply_exp_inplace_and_find_exp_sum(uint32_t cb_qk_result, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum) {
    cb_wait_front(cb_qk_result, onetile);
    cb_wait_front(cb_cur_max, onetile);

    const uint32_t exp_dst_idx = 0;
    sub_bcast_cols_init_short(cb_qk_result, cb_cur_max);
    tile_regs_acquire();
    sub_tiles_bcast_cols(cb_qk_result, cb_cur_max, /* tile_idx */ 0, /* tile_idx */ 0, /* dst_reg_idx */ exp_dst_idx);

    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(exp_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    // update current qk matmul result with exp values
    cb_pop_front(cb_qk_result, onetile);
    cb_reserve_back(cb_qk_result, onetile);
    pack_reconfig_data_format(cb_qk_result);
    pack_tile(exp_dst_idx, cb_qk_result);

    /* update current exp sum with exp values
     * at the moment we pack one tile here
     * but we can use L1 accumlator to pack more tiles
     * in case we will be able to read more then one row of K and V
     */
    cb_reserve_back(cb_cur_exp_sum, onetile);
    pack_reconfig_data_format(cb_cur_exp_sum);
    pack_tile(exp_dst_idx, cb_cur_exp_sum);
    tile_regs_release();

    cb_push_back(cb_qk_result, onetile);
    cb_push_back(cb_cur_exp_sum, onetile);
}

void matmul_qk_by_v(
    uint32_t Wt, uint32_t block_size, uint32_t cb_qk_result, uint32_t cb_value, uint32_t cb_cur_mm_out) {
    cb_wait_front(cb_qk_result, onetile);
    cb_wait_front(cb_value, Wt);
    cb_reserve_back(cb_cur_mm_out, Wt);

    mm_init_short(cb_qk_result, cb_value, /* transpose */ 0);
    pack_reconfig_data_format(cb_cur_mm_out);
    reconfig_data_format(cb_qk_result, cb_value);
    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_qk_result,
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

void update_exp_max_diff(uint32_t cb_prev_max_value, uint32_t cb_cur_max_value, uint32_t cb_exp_max_diff) {
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

/*This uses L1 accumulation to accumulate onto cb_cur_mm_out*/
void update_cur_mm_out(
    uint32_t Wt, uint32_t block_size, uint32_t cb_prev_mm_out, uint32_t cb_cur_mm_out, uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_prev_mm_out, Wt);
    cb_wait_front(cb_cur_mm_out, Wt);
    cb_wait_front(cb_exp_max_diff, onetile);

    pack_reconfig_data_format(cb_cur_mm_out);
    // This function would ideally be called after other initialization functions that initialize the packer for a
    // specific operation.
    pack_reconfig_l1_acc(true);

    reconfig_data_format(cb_prev_mm_out, cb_exp_max_diff);
    mul_bcast_cols_init_short(cb_prev_mm_out, cb_exp_max_diff);
    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_tiles_bcast_cols(cb_prev_mm_out, cb_exp_max_diff, tile_idx + block_idx, 0, block_idx);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(/*dst_reg_idx*/ block_idx, cb_cur_mm_out);
        }
        tile_regs_release();
    }
    pack_reconfig_l1_acc(false);
    // update cb_cur_mm_out pointer
    cb_pop_front(cb_cur_mm_out, Wt);
    cb_reserve_back(cb_cur_mm_out, Wt);
    cb_push_back(cb_cur_mm_out, Wt);
}

// reduce and recip in place
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t cb_identity_scaler, uint32_t cb_matmul_reduce>
void reduce_and_recip_tile_inplace(uint32_t cb_in_idx) {
    cb_wait_front(cb_in_idx, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);  // generate tile for matmul row reduce)

    const uint32_t reduce_dst_idx = 0;

    reconfig_data_format(cb_in_idx, cb_matmul_reduce);  // reconfig data format to precise
    tile_regs_acquire();

    mm_init(cb_in_idx, cb_matmul_reduce, cb_identity_scaler, 0);
    matmul_tiles(cb_in_idx, cb_matmul_reduce, /* tile_idx */ 0, /* tile_idx */ 0, reduce_dst_idx);

    recip_tile_init();
    recip_tile(reduce_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_in_idx, onetile);
    cb_reserve_back(cb_in_idx, onetile);
    pack_reconfig_data_format(cb_in_idx);
    pack_tile(reduce_dst_idx, cb_in_idx);
    tile_regs_release();
    cb_push_back(cb_in_idx, onetile);
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
