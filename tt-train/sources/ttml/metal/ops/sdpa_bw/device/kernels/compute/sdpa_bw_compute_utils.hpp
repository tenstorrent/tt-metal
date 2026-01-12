// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/debug/dprint.h>
#include <compute_kernel_api/reg_api.h>

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
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

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
//   - Multiply by infinity, resulting in 0.0 for valid entries and -inf for
//   masked ones.
// This way, after applying softmax, masked positions will effectively become zero,
// and only the unmasked positions will retain meaningful attention weights
void apply_mask_on_reg(
    const uint32_t register_idx,
    const uint32_t cb_attn_mask,
    const uint32_t scaler_bits,
    const uint32_t minus_one_bits,
    const uint32_t custom_inf_bits) {
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
    mul_unary_tile(mask_register, custom_inf_bits);  // multiply by inf to transform mask to 0.0 and -inf

    // Add mask to scaled matmul result:
    // masked positions receive large negative values (will be 0.0 after softmax),
    // unmasked positions remain unchanged
    add_binary_tile_init();
    add_binary_tile(register_idx, mask_register, register_idx);

    cb_pop_front(cb_attn_mask, onetile);
}

// Recomputes attention weights from pre-softmax scores using stored statistics.
// Given raw attention scores and intermediates (max_val at [0], recip_sum_exp at [1]),
// computes: softmax(x) = exp(x - max) * recip_sum_exp
// This is used in backward pass to reconstruct P from stored forward pass statistics.
void apply_statistics_inplace(
    const uint32_t cb_attention_weights, const uint32_t cb_intermediates, const uint32_t num_of_interm_tiles) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_intermediates, num_of_interm_tiles);

    const uint32_t working_reg = 0;
    const uint32_t intermediates_reg = 1U;

    init_bcast<ELWSUB, BroadcastType::COL>(cb_attention_weights, cb_intermediates, cb_attention_weights);

    reconfig_data_format(cb_attention_weights, cb_intermediates);
    tile_regs_acquire();
    // apply statistics: subtract per row max value stored in intermediates[0]
    sub_bcast_cols_init_short(cb_attention_weights, cb_intermediates);
    sub_tiles_bcast_cols(cb_attention_weights, cb_intermediates, /* tile_idx */ 0, /* tile_idx */ 0, working_reg);

    // exp(x - max(x))
    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(working_reg);

    reconfig_data_format(cb_intermediates, cb_intermediates);
    // bcast 1/sum(exp(x - max(x))) stored in intermediates[1]
    unary_bcast_init<BroadcastType::COL>(cb_intermediates, cb_intermediates);
    unary_bcast<BroadcastType::COL>(cb_intermediates, /* tile idx */ 1U, /* reg tile idx */ intermediates_reg);

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
}

// Transposes a single tile by swapping width and height dimensions.
// Used for computing A^T @ B matmuls in backward pass (e.g., dV = P^T @ dO).
inline void transpose_tile(const uint32_t cb_input, /*output cb*/ const uint32_t cb_transpose_wh) {
    cb_wait_front(cb_input, onetile);
    // transpose attention weights
    reconfig_data_format(cb_input, cb_input);
    tile_regs_acquire();
    transpose_wh_init(cb_input, cb_transpose_wh);
    transpose_wh_tile(cb_input, /* tile idx */ 0, /* reg idx */ 0);
    tile_regs_commit();

    cb_reserve_back(cb_transpose_wh, onetile);
    pack_reconfig_data_format(cb_transpose_wh);
    tile_regs_wait();
    pack_tile(0, cb_transpose_wh);
    tile_regs_release();
    cb_push_back(cb_transpose_wh, onetile);
}

// Computes the per-row scalar u = sum(dO * O) needed for softmax backward.
// This is part of the softmax gradient: dS = P * (dP - u), where u = sum(P * dP) per row.
// Since O = P @ V, we have dP = dO @ V^T, and u = sum(dO * O) row-wise.
// The reduction is done via matmul with a column of ones (cb_mat_mul_reduction).
void compute_u_scalar_row(
    const uint32_t cb_grad_output,
    const uint32_t cb_attn_output,
    /*output result*/ const uint32_t cb_u_scalar_row,
    /*mutmul reduction*/ const uint32_t cb_mat_mul_reduction,
    const uint32_t tiles_per_row,
    const uint32_t scaler_bits) {
    const uint32_t accum_register = 0;
    // using binary_tiles_init function instead of specific mul_tiles_init() because specific one doesn't support
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

    // This call is required to set up the matmul correctly
    mm_init_short(cb_u_scalar_row, cb_mat_mul_reduction, /* transpose */ 0);
    matmul_tiles(
        cb_u_scalar_row,
        cb_mat_mul_reduction,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* dst_reg_idx*/ accum_register);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_u_scalar_row, onetile);
    cb_reserve_back(cb_u_scalar_row, onetile);
    pack_reconfig_data_format(cb_u_scalar_row);
    pack_tile(accum_register, cb_u_scalar_row);
    tile_regs_release();
    cb_push_back(cb_u_scalar_row, onetile);
}

// Computes gradient w.r.t. attention weights: dP = dO @ V^T
// This is the first step in the backward chain from output gradient to score gradient.
// Input: dO (grad_output) and V (value), Output: dP (grad_attn_weights)
void compute_grad_attn_weights(
    const uint32_t cb_grad_output,
    const uint32_t cb_value,
    const uint32_t tiles_per_row,
    const uint32_t cb_grad_attn_weights,
    const uint32_t scaler_bits) {
    reconfig_data_format(cb_grad_output, cb_value);
    // This call is required to set up the matmul correctly
    mm_init_short(cb_grad_output, cb_value, /* transpose */ 1);
    tile_regs_acquire();
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
        matmul_tiles(
            cb_grad_output,
            cb_value,
            /* tile_idx */ tile_idx,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0);
    }
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_grad_attn_weights, onetile);
    pack_reconfig_data_format(cb_grad_attn_weights);
    pack_tile(0, cb_grad_attn_weights);
    tile_regs_release();

    cb_push_back(cb_grad_attn_weights, onetile);
}

// Computes gradient w.r.t. pre-softmax scores (softmax backward pass).
// Formula: dS = P * (dP - u) * scale, where:
//   - P = attention_weights (softmax output)
//   - dP = grad_attn_weights (gradient from dO @ V^T)
//   - u = u_scalar_row (per-row sum: sum(dO * O))
//   - scale = 1/sqrt(d_k) applied here for numerical stability in subsequent matmuls
void compute_grad_scores(
    const uint32_t cb_grad_attn_weights,
    const uint32_t cb_attention_weights,
    const uint32_t cb_u_scalar_row,
    const uint32_t scaler_bits,
    /* output */ const uint32_t cb_grad_scores) {
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

    // We apply scaling here to improve numerical stability of upcoming matmuls for grad Q and grad K
    binop_with_scalar_tile_init();
    mul_unary_tile(/* dst_reg_idx*/ grad_reg, scaler_bits);  // multiply by scaler factor

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_grad_scores, onetile);
    pack_reconfig_data_format(cb_grad_scores);
    pack_tile(grad_reg, cb_grad_scores);
    tile_regs_release();
    cb_push_back(cb_grad_scores, onetile);
}

// Computes gradient w.r.t. Value tensor: dV = P^T @ dO
// For grouped query attention, gradients from multiple query heads are accumulated
// into their shared KV head using do_accumulate flag.
// Uses cb_transpose_wh as scratch space for transposed attention weights.
void update_grad_value(
    const uint32_t cb_attention_weights,
    const uint32_t cb_transpose_wh,
    const uint32_t cb_grad_output,
    const uint32_t cb_prev_grad_value,
    const uint32_t cb_cur_grad_value,
    const uint32_t tiles_per_row,
    const uint32_t block_size,
    const bool do_accumulate = false) {
    transpose_tile(cb_attention_weights, cb_transpose_wh);

    // grad_V = Attention^T @ grad_output
    cb_wait_front(cb_transpose_wh, onetile);

    cb_reserve_back(cb_cur_grad_value, tiles_per_row);
    pack_reconfig_data_format(cb_cur_grad_value);
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
        tile_regs_acquire();
        // This call is required to set up the matmul correctly
        mm_init_short_with_dt(cb_transpose_wh, cb_grad_output, cb_prev_grad_value, /*transpose*/ 0);
        matmul_tiles(
            cb_transpose_wh,
            cb_grad_output,
            /* tile_idx */ 0,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0);

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

// Computes gradient w.r.t. Key tensor: dK = dS^T @ Q
// where dS = scaled gradient w.r.t. scores (already includes 1/sqrt(d_k) scaling).
// For grouped query attention, gradients from multiple query heads are accumulated
// into their shared KV head using do_accumulate flag.
void update_grad_key(
    const uint32_t cb_grad_scores,
    const uint32_t cb_query,
    const uint32_t scaler_bits,
    const uint32_t cb_transpose_wh,
    const uint32_t cb_prev_grad_key,
    const uint32_t cb_cur_grad_key,
    const uint32_t tiles_per_row,
    const bool do_accumulate = false) {
    transpose_tile(cb_grad_scores, cb_transpose_wh);
    cb_wait_front(cb_transpose_wh, onetile);

    cb_reserve_back(cb_cur_grad_key, tiles_per_row);
    pack_reconfig_data_format(cb_cur_grad_key);
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
        tile_regs_acquire();
        // This call is required to set up the matmul correctly
        mm_init_short_with_dt(cb_transpose_wh, cb_query, cb_prev_grad_key, /*transpose*/ 0);
        matmul_tiles(
            cb_transpose_wh,
            cb_query,
            /* tile_idx */ 0,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0);

        if (do_accumulate) {
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

// Computes gradient w.r.t. Query tensor: dQ = dS @ K
// where dS = scaled gradient w.r.t. scores (already includes 1/sqrt(d_k) scaling).
// Accumulates across sequence blocks when processing in tiles (do_accumulate=true).
void update_grad_query(
    const uint32_t cb_grad_scores,
    const uint32_t cb_key,
    const uint32_t scaler_bits,
    const uint32_t cb_prev_grad_query,
    const uint32_t cb_cur_grad_query,
    const uint32_t tiles_per_row,
    const bool do_accumulate = false) {
    cb_wait_front(cb_grad_scores, onetile);
    cb_reserve_back(cb_cur_grad_query, tiles_per_row);
    pack_reconfig_data_format(cb_cur_grad_query);
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
        tile_regs_acquire();
        // This call is required to set up the matmul correctly
        mm_init_short_with_dt(cb_grad_scores, cb_key, cb_prev_grad_query, /*transpose*/ 0);
        matmul_tiles(
            cb_grad_scores,
            cb_key,
            /* tile_idx */ 0,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0);

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
