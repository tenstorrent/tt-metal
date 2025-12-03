// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/reg_api.h>
#include <debug/dprint.h>

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
#include "compute_kernel_api/transpose_wh_dest.h"

// test
#include "compute_kernel_api/eltwise_unary/fill.h"

constexpr uint32_t onetile = 1U;

#ifdef FP32_DEST_ACC_EN
constexpr uint32_t dst_reg_number = 4U;
#else
constexpr uint32_t dst_reg_number = 8U;
#endif

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

    cb_pop_front(cb_attn_mask, onetile);
}

/* In-place mask: keep column 0 of `intermediates`, zero everything else.
 * Uses the matmul-reduction tile (1s in col 0, 0s elsewhere) as the mask.
 */
void mask_intermediates(
    uint32_t cb_intermediates, uint32_t cb_mat_mul_reduction, uint32_t cb_masked_interm, uint32_t num_of_interm_tiles) {
    /*
     * copy interm value to dst_reg with indexes 0 and 2, copy mask to dst_reg with indexes 1 and 3
     * then apply mask on dst_reg with indexes 0 and 2
     */
    cb_wait_front(cb_intermediates, num_of_interm_tiles);
    tile_regs_acquire();
    for (uint32_t tile_idx = 0; tile_idx < num_of_interm_tiles; ++tile_idx) {
        // UNPACK({ DPRINT << "Copying intermediates tile " << tile_idx << " to " << 2 * tile_idx << ENDL(); });
        reconfig_data_format(cb_intermediates, cb_intermediates);
        copy_tile_init(cb_intermediates);
        copy_tile(cb_intermediates, /* tile_idx */ tile_idx, /* register idx */ 2 * tile_idx);

        // UNPACK({
        //     DPRINT << "Copying matmul reduction for tile " << 2 * tile_idx << " to " << 2 * tile_idx + 1 << ENDL();
        // });
        reconfig_data_format(cb_mat_mul_reduction, cb_mat_mul_reduction);
        copy_tile_init(cb_mat_mul_reduction);
        copy_tile(cb_mat_mul_reduction, /* tile_idx */ 0, /* register idx */ 2 * tile_idx + 1);

        mask_tile_init();
        mask_tile(2 * tile_idx, 2 * tile_idx + 1);  // mask should be next to tile register
    }
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_masked_interm, num_of_interm_tiles);
    pack_reconfig_data_format(cb_masked_interm);
    for (uint32_t tile_idx = 0; tile_idx < num_of_interm_tiles; ++tile_idx) {
        pack_tile(2 * tile_idx, cb_masked_interm);
    }
    tile_regs_release();
    cb_push_back(cb_masked_interm, num_of_interm_tiles);

    cb_pop_front(cb_intermediates, num_of_interm_tiles);
}

void apply_statistics_inplace(uint32_t cb_attention_weights, uint32_t cb_masked_interm, uint32_t num_of_interm_tiles) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_masked_interm, num_of_interm_tiles);

    const uint32_t working_reg = 0;
    const uint32_t intermediates_reg = 1U;

    init_bcast<ELWSUB, BroadcastType::COL>(cb_attention_weights, cb_masked_interm, cb_attention_weights);

    reconfig_data_format(cb_attention_weights, cb_masked_interm);
    tile_regs_acquire();
    // apply statistics: subtract per row max value stored in intermediates[0]
    sub_bcast_cols_init_short(cb_attention_weights, cb_masked_interm);
    sub_tiles_bcast_cols(cb_attention_weights, cb_masked_interm, /* tile_idx */ 0, /* tile_idx */ 0, working_reg);

    // exp(x - max(x))
    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(working_reg);

    reconfig_data_format(cb_masked_interm, cb_masked_interm);
    // bcast 1/sum(exp(x - max(x))) stored in intermediates[1]
    unary_bcast_init<BroadcastType::COL>(cb_masked_interm, cb_masked_interm);
    unary_bcast<BroadcastType::COL>(cb_masked_interm, /* tile idx */ 1U, /* reg tile idx */ intermediates_reg);

    mul_binary_tile_init();
    mul_binary_tile(working_reg, intermediates_reg, working_reg);  // multiply by 1/sum(exp(x - max(x)))
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_attention_weights, onetile);
    cb_reserve_back(cb_attention_weights, onetile);
    pack_reconfig_data_format(cb_attention_weights);
    pack_tile(working_reg, cb_attention_weights);
    tile_regs_release();
    cb_push_back(cb_attention_weights, onetile);

    // [DPRINT]: For debug, should be removed later
    // need to pop intermediates outside this function because for grad Q kernel I need one interm per q row
    // cb_pop_front(cb_masked_interm, num_of_interm_tiles);
}

inline void transpose_attn_weights(uint32_t cb_attention_weights, uint32_t cb_transpose_wh) {
    cb_wait_front(cb_attention_weights, onetile);
    // transpose attention weights
    reconfig_data_format(cb_attention_weights, cb_attention_weights);
    tile_regs_acquire();
    transpose_wh_init(cb_attention_weights, cb_transpose_wh);
    transpose_wh_tile(cb_attention_weights, /* tile idx */ 0, /* reg idx */ 0);
    tile_regs_commit();

    cb_reserve_back(cb_transpose_wh, onetile);
    pack_reconfig_data_format(cb_transpose_wh);
    tile_regs_wait();
    pack_tile(0, cb_transpose_wh);
    tile_regs_release();
    cb_push_back(cb_transpose_wh, onetile);
}

void compute_u_scalar_row(
    uint32_t cb_grad_output,
    uint32_t cb_attn_output,
    /*output result*/ uint32_t cb_u_scalar_row,
    /*mutmul reduction*/ uint32_t cb_mat_mul_reduction,
    uint32_t tiles_per_row,
    uint32_t scaler_bits) {
    const uint32_t accum_register = 0;
    // TODO[check]: probably I need reconfig data format for cb_grad_output and cb_attn_output here
    // using general init function instead of specific mul_tiles_init() because specific one doesn't support
    // accumulation to dest regs
    reconfig_data_format(cb_grad_output, cb_attn_output);
    binary_tiles_init<true, ELWMUL>(cb_grad_output, cb_attn_output, /*acc_to_dest*/ true);
    tile_regs_acquire();
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
        mul_tiles(
            cb_grad_output,
            cb_attn_output,
            /* tile_idx */ tile_idx,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ accum_register);
    }

    binop_with_scalar_tile_init();
    mul_unary_tile(/* dst_reg_idx*/ accum_register, scaler_bits);  // multiply by scaler factor
    tile_regs_commit();

    pack_reconfig_data_format(cb_u_scalar_row);
    cb_reserve_back(cb_u_scalar_row, onetile);
    tile_regs_wait();
    pack_tile(accum_register, cb_u_scalar_row);
    tile_regs_release();
    cb_push_back(cb_u_scalar_row, onetile);

    cb_wait_front(cb_u_scalar_row, onetile);
    tile_regs_acquire();
    reconfig_data_format(cb_u_scalar_row, cb_mat_mul_reduction);
    mm_init_short(cb_u_scalar_row, cb_mat_mul_reduction, /* transpose */ 0);
    // mm_init(cb_u_scalar_row, cb_mat_mul_reduction, cb_u_scalar_row, /* transpose */ 0);
    matmul_tiles(
        cb_u_scalar_row,
        cb_mat_mul_reduction,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* dst_reg_idx*/ accum_register,
        /* transpose */ 0);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_u_scalar_row, onetile);
    cb_reserve_back(cb_u_scalar_row, onetile);
    pack_reconfig_data_format(cb_u_scalar_row);
    pack_tile(accum_register, cb_u_scalar_row);
    tile_regs_release();
    cb_push_back(cb_u_scalar_row, onetile);
}

// Compute gradient w.r.t. attention weights
void compute_grad_attn_weights(
    uint32_t cb_grad_output,
    uint32_t cb_value,
    uint32_t tiles_per_row,
    uint32_t cb_grad_attn_weights,
    uint32_t scaler_bits) {
    reconfig_data_format(cb_grad_output, cb_value);
    mm_init_short(cb_grad_output, cb_value, /* transpose */ 1);
    // mm_init(cb_grad_output, cb_value, cb_grad_attn_weights, /* transpose */ 1);
    tile_regs_acquire();
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
        matmul_tiles(
            cb_grad_output,
            cb_value,
            /* tile_idx */ tile_idx,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0,
            /* transpose */ 1);
    }

    binop_with_scalar_tile_init();
    mul_unary_tile(/* dst_reg_idx*/ 0, scaler_bits);  // multiply by scaler factor

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_grad_attn_weights, onetile);
    pack_reconfig_data_format(cb_grad_attn_weights);
    pack_tile(0, cb_grad_attn_weights);
    tile_regs_release();

    cb_push_back(cb_grad_attn_weights, onetile);
}

// Compute gradient w.r.t. scores(before softmax) dL/dZ = dL/d(Q@K^T) = (dP - u_scalar_row) * P
// TODO(vmelnykov): In general we need to use fp32 for cb_grad_scores but right now we can't do matmul beween fp16
// and fp32 CBs(need to compute grad Q and grad K with better accuracy)
void compute_grad_scores(
    uint32_t cb_grad_attn_weights,
    uint32_t cb_attention_weights,
    uint32_t cb_u_scalar_row,
    uint32_t scaler_bits,
    /* output */ uint32_t cb_grad_scores,
    bool transpose = false) {
    cb_wait_front(cb_grad_attn_weights, onetile);
    cb_wait_front(cb_u_scalar_row, onetile);

    const uint32_t grad_reg = 0;
    const uint32_t attn_weights_reg = 1U;
    const uint32_t u_scalar_reg = 2U;

    // compute: grad_scores = (grad_attn_weights - u_scalar_row) * attention_weights
    tile_regs_acquire();
    reconfig_data_format(cb_grad_attn_weights, cb_u_scalar_row);
    sub_bcast_cols_init_short(cb_grad_attn_weights, cb_u_scalar_row);
    sub_tiles_bcast_cols(
        cb_grad_attn_weights,
        cb_u_scalar_row,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        grad_reg);  // result in grad_reg

    // copy attention_weights to reg 1
    reconfig_data_format(cb_attention_weights, cb_attention_weights);
    copy_tile_init(cb_attention_weights);
    copy_tile(cb_attention_weights, /* tile_idx */ 0, /* register idx */ attn_weights_reg);

    mul_binary_tile_init();
    mul_binary_tile(grad_reg, attn_weights_reg, grad_reg);  // result in grad_reg

    // binop_with_scalar_tile_init();
    // mul_unary_tile(/* dst_reg_idx*/ grad_reg, scaler_bits);  // multiply by scaler factor

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_grad_scores, onetile);
    pack_reconfig_data_format(cb_grad_scores);
    pack_tile(grad_reg, cb_grad_scores);
    tile_regs_release();
    cb_push_back(cb_grad_scores, onetile);
}

void update_grad_value(
    uint32_t cb_attention_weights,
    uint32_t cb_transpose_wh,
    uint32_t cb_grad_output,
    uint32_t cb_prev_grad_value,
    uint32_t cb_cur_grad_value,
    uint32_t tiles_per_row,
    uint32_t block_size,
    bool do_accumulate = false) {
    transpose_attn_weights(cb_attention_weights, cb_transpose_wh);

    // grad_V = Attention^T @ grad_output
    cb_wait_front(cb_transpose_wh, onetile);

    // mm_init(cb_transpose_wh, cb_grad_output, cb_cur_grad_value, 0);
    cb_reserve_back(cb_cur_grad_value, tiles_per_row);
    pack_reconfig_data_format(cb_cur_grad_value);
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx++) {
        tile_regs_acquire();
        // reconfig_data_format(cb_transpose_wh, cb_grad_output);
        // mm_init_short(cb_transpose_wh, cb_grad_output, /* transpose */ 0);
        mm_init_short_with_dt(cb_transpose_wh, cb_grad_output, cb_prev_grad_value, /*transpose*/ 0);
        matmul_tiles(
            cb_transpose_wh,
            cb_grad_output,
            /* tile_idx */ 0,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0,
            /* transpose */ 0);

        if (do_accumulate) {
            copy_tile_to_dst_init_short_with_dt(cb_transpose_wh, cb_prev_grad_value);
            copy_tile_init(cb_prev_grad_value);
            copy_tile(cb_prev_grad_value, /* tile_idx */ tile_idx, /* register idx */ 1U);

            add_binary_tile_init();
            add_binary_tile(0, 1U, 0);  // accumulate in register 0
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_cur_grad_value);
        tile_regs_release();
    }
    cb_push_back(cb_cur_grad_value, tiles_per_row);

    // pop temporary cbs
    cb_pop_front(cb_transpose_wh, onetile);
    if (do_accumulate) {
        cb_pop_front(cb_prev_grad_value, tiles_per_row);
    }
}

void fill_corr_cb_with_zeros(uint32_t cb_corr, uint32_t tiles_per_head) {
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(/* dst_reg_idx*/ 0, 0.0F);  // fill with zeros
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_corr, tiles_per_head);
    pack_reconfig_data_format(cb_corr);
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_head; tile_idx++) {
        pack_tile(/*dst_reg_idx*/ 0, cb_corr);
    }
    cb_push_back(cb_corr, tiles_per_head);
}

// current matmul res is already in cur_sum_dest_reg_idx
// function assumes that dest_reg is in acquired state via *acquire_dst* call
void kahan_summation(uint32_t cb_prev_sum, uint32_t cb_corr, uint32_t tile_idx, uint32_t cur_mm_dest_reg_idx = 0) {
    const uint32_t corr_reg = cur_mm_dest_reg_idx + 1U;
    const uint32_t prev_sum_reg = cur_mm_dest_reg_idx + 2U;
    const uint32_t result_reg = cur_mm_dest_reg_idx + 3U;

    // y = cur_sum - corr
    reconfig_data_format(cb_corr, cb_corr);
    copy_tile_init(cb_corr);
    copy_tile(cb_corr, tile_idx, corr_reg);

    sub_binary_tile_init();
    sub_binary_tile(cur_mm_dest_reg_idx, corr_reg, cur_mm_dest_reg_idx);  // y = cur_mm_res - corr

    reconfig_data_format(cb_prev_sum, cb_prev_sum);
    copy_tile_init(cb_prev_sum);
    copy_tile(cb_prev_sum, tile_idx, prev_sum_reg);

    add_binary_tile_init();
    add_binary_tile(cur_mm_dest_reg_idx, prev_sum_reg, result_reg);  // t = prev_sum + y

    // corr = (t - prev_sum) - y
    sub_binary_tile_init();
    sub_binary_tile(result_reg, prev_sum_reg, corr_reg);  // corr = t - prev_sum
    sub_binary_tile_init();
    sub_binary_tile(corr_reg, cur_mm_dest_reg_idx, corr_reg);  // corr = (t - prev_sum) - y

    // prev_sum = t
    // result already in result_reg
}

// void update_grad_key(
//     uint32_t cb_grad_scores,
//     uint32_t cb_query,
//     uint32_t scaler_bits,
//     uint32_t cb_transpose_wh,
//     uint32_t cb_prev_grad_key,
//     uint32_t cb_cur_grad_key,
//     uint32_t cb_prev_corr,
//     uint32_t cb_cur_corr,
//     uint32_t tiles_per_row,
//     bool do_accumulate = false) {
//     // TODO: rename trasnpose function
//     transpose_attn_weights(cb_grad_scores, cb_transpose_wh);
//     cb_wait_front(cb_transpose_wh, onetile);

//     // mm_init(cb_transpose_wh, cb_query, cb_cur_grad_key, 0);
//     cb_reserve_back(cb_cur_grad_key, tiles_per_row);
//     pack_reconfig_data_format(cb_cur_grad_key);

//     if (do_accumulate) {
//         cb_wait_front(cb_prev_corr, tiles_per_row);
//         cb_reserve_back(cb_cur_corr, tiles_per_row);
//     }

//     // reconfig_data_format(cb_transpose_wh, cb_query);
//     for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx++) {
//         tile_regs_acquire();
//         // reconfig_data_format(cb_transpose_wh, cb_query);
//         // mm_init_short(cb_transpose_wh, cb_query, /* transpose */ 0);
//         mm_init_short_with_dt(cb_transpose_wh, cb_query, cb_prev_grad_key, /*transpose*/ 0);
//         matmul_tiles(
//             cb_transpose_wh,
//             cb_query,
//             /* tile_idx */ 0,
//             /* tile_idx */ tile_idx,
//             /* dst_reg_idx*/ 0,
//             /* transpose */ 0);

//         // apply scaler factor to the result before matmul accumulation
//         // maybe will achive better accuracy
//         // binop_with_scalar_tile_init();
//         // mul_unary_tile(/* dst_reg_idx*/ 0, scaler_bits);  // multiply by scaler factor

//         if (do_accumulate) {
//             // reconfig_data_format(cb_prev_grad_key, cb_prev_grad_key);
//             // copy_tile_init(cb_prev_grad_key);
//             // copy_tile(cb_prev_grad_key, /* tile_idx */ tile_idx, /* register idx */ 1U);

//             // add_binary_tile_init();
//             // add_binary_tile(0, 1U, 0);  // accumulate in register 0
//             kahan_summation(cb_prev_grad_key, cb_prev_corr, tile_idx, /* cur_mm_dest_reg_idx */ 0);
//         }
//         tile_regs_commit();

//         tile_regs_wait();
//         pack_reconfig_data_format(cb_cur_grad_key);
//         const uint32_t result_reg = do_accumulate ? 3U : 0U;
//         pack_tile(result_reg, cb_cur_grad_key);
//         if (do_accumulate) {
//             pack_reconfig_data_format(cb_cur_corr);
//             pack_tile(1U, cb_cur_corr);
//         }
//         tile_regs_release();
//     }
//     cb_push_back(cb_cur_grad_key, tiles_per_row);

//     cb_pop_front(cb_transpose_wh, onetile);
//     if (do_accumulate) {
//         cb_pop_front(cb_prev_grad_key, tiles_per_row);
//         cb_pop_front(cb_prev_corr, tiles_per_row);

//         cb_push_back(cb_cur_corr, tiles_per_row);
//     }
// }

void update_grad_key(
    uint32_t cb_grad_scores,
    uint32_t cb_query,
    uint32_t scaler_bits,
    uint32_t cb_transpose_wh,
    uint32_t cb_prev_grad_key,
    uint32_t cb_cur_grad_key,
    uint32_t tiles_per_row,
    bool do_accumulate = false) {
    // TODO: rename trasnpose function
    transpose_attn_weights(cb_grad_scores, cb_transpose_wh);
    cb_wait_front(cb_transpose_wh, onetile);

    // mm_init(cb_transpose_wh, cb_query, cb_cur_grad_key, 0);
    cb_reserve_back(cb_cur_grad_key, tiles_per_row);
    pack_reconfig_data_format(cb_cur_grad_key);
    // reconfig_data_format(cb_transpose_wh, cb_query);
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx++) {
        tile_regs_acquire();
        // reconfig_data_format(cb_transpose_wh, cb_query);
        // mm_init_short(cb_transpose_wh, cb_query, /* transpose */ 0);
        mm_init_short_with_dt(cb_transpose_wh, cb_query, cb_prev_grad_key, /*transpose*/ 0);
        matmul_tiles(
            cb_transpose_wh,
            cb_query,
            /* tile_idx */ 0,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0,
            /* transpose */ 0);

        // apply scaler factor to the result before matmul accumulation
        // maybe will achive better accuracy
        // binop_with_scalar_tile_init();
        // mul_unary_tile(/* dst_reg_idx*/ 0, scaler_bits);  // multiply by scaler factor

        if (do_accumulate) {
            // reconfig_data_format_srca(cb_prev_grad_key);
            // copy_tile_init(cb_prev_grad_key);
            copy_tile_to_dst_init_short_with_dt(cb_transpose_wh, cb_prev_grad_key);
            copy_tile(cb_prev_grad_key, /* tile_idx */ tile_idx, /* register idx */ 1U);

            add_binary_tile_init();
            add_binary_tile(0, 1U, 0);  // accumulate in register 0
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_cur_grad_key);
        tile_regs_release();
    }
    cb_push_back(cb_cur_grad_key, tiles_per_row);

    cb_pop_front(cb_transpose_wh, onetile);
    if (do_accumulate) {
        cb_pop_front(cb_prev_grad_key, tiles_per_row);
    }
}

void update_grad_query(
    uint32_t cb_grad_scores,
    uint32_t cb_key,
    uint32_t scaler_bits,
    uint32_t cb_prev_grad_query,
    uint32_t cb_cur_grad_query,
    uint32_t tiles_per_row,
    bool do_accumulate = false) {
    cb_wait_front(cb_grad_scores, onetile);
    cb_reserve_back(cb_cur_grad_query, tiles_per_row);
    pack_reconfig_data_format(cb_cur_grad_query);
    // TODO(vmelnykov): In general we need to use fp32 for cb_grad_scores but right now we can't do matmul beween fp16
    // and fp32 CBs(need to compute grad Q and grad K with better accuracy)
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx++) {
        tile_regs_acquire();
        // reconfig_data_format(cb_grad_scores, cb_key);
        // mm_init_short(cb_grad_scores, cb_key, /* transpose */ 0);
        mm_init_short_with_dt(cb_grad_scores, cb_key, cb_prev_grad_query, /*transpose*/ 0);
        matmul_tiles(
            cb_grad_scores,
            cb_key,
            /* tile_idx */ 0,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0,
            /* transpose */ 0);

        // apply scaler factor to the result before matmul accumulation
        // maybe will achive better accuracy
        // binop_with_scalar_tile_init();
        // mul_unary_tile(/* dst_reg_idx*/ 0, scaler_bits);  // multiply by scaler factor

        if (do_accumulate) {
            copy_tile_to_dst_init_short_with_dt(cb_grad_scores, cb_prev_grad_query);
            copy_tile_init(cb_prev_grad_query);
            copy_tile(cb_prev_grad_query, /* tile_idx */ tile_idx, /* register idx */ 1U);

            add_binary_tile_init();
            add_binary_tile(0, 1U, 0);  // accumulate in register 0
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_cur_grad_query);
        tile_regs_release();
    }
    cb_push_back(cb_cur_grad_query, tiles_per_row);

    cb_pop_front(cb_grad_scores, onetile);
    if (do_accumulate) {
        cb_pop_front(cb_prev_grad_query, tiles_per_row);
    }
}

void pack_result(uint32_t cb_source, uint32_t cb_output, uint32_t num_tiles) {
    cb_wait_front(cb_source, num_tiles);
    cb_reserve_back(cb_output, num_tiles);

    pack_reconfig_data_format(cb_output);
    reconfig_data_format(cb_source, cb_source);

    copy_tile_init(cb_source);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tile_regs_acquire();
        copy_tile(
            cb_source,
            /* tile_idx */ tile_idx,
            /* register idx */ 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/* register idx */ 0, cb_output);
        tile_regs_release();
    }
    cb_push_back(cb_output, num_tiles);
    cb_pop_front(cb_source, num_tiles);
}

void sync_with_writer(uint32_t cb_sync_output_writer) {
    cb_reserve_back(cb_sync_output_writer, onetile);
    cb_push_back(cb_sync_output_writer, onetile);
}
