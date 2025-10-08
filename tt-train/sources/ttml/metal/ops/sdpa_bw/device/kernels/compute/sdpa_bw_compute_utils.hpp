// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
}

void apply_statistics_inplace(uint32_t cb_attention_weights, uint32_t cb_intermediates) {
    cb_wait_front(cb_attention_weights, onetile);

    const uint32_t working_reg = 0;
    const uint32_t intermediates_reg = 1U;

    reconfig_data_format(cb_attention_weights, cb_attention_weights);
    tile_regs_acquire();
    // apply statistics: subtract per row max value stored in intermediates[0]
    sub_bcast_cols_init_short(cb_attention_weights, cb_intermediates);
    sub_tiles_bcast_cols(cb_attention_weights, cb_intermediates, /* tile_idx */ 0, /* tile_idx */ 0, working_reg);

    // exp(x - max(x))
    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(working_reg);

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

// Compute gradient w.r.t. V: grad_V = Attention^T @ grad_output
void update_grad_value(
    uint32_t cb_attention_weights,
    uint32_t cb_transpose_wh,
    uint32_t cb_grad_output,
    uint32_t cb_mm_result_holder,
    uint32_t cb_grad_value,
    uint32_t tiles_per_row,
    uint32_t block_size,
    bool do_accumulate = false) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_grad_output, tiles_per_row);

    // transpose attention weights
    cb_reserve_back(cb_transpose_wh, onetile);
    pack_reconfig_data_format(cb_transpose_wh);
    tile_regs_acquire();
    transpose_wh_init(cb_attention_weights, cb_transpose_wh);
    transpose_wh_tile(cb_attention_weights, /* tile idx */ 0, /* reg idx */ 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_transpose_wh);
    tile_regs_release();
    cb_push_back(cb_transpose_wh, onetile);

    // grad_V = Attention^T @ grad_output
    cb_wait_front(cb_transpose_wh, onetile);
    reconfig_data_format(cb_transpose_wh, cb_grad_output);
    mm_init_short(cb_transpose_wh, cb_grad_output, /* transpose */ 0);
    pack_reconfig_data_format(cb_grad_value);
    // TODO[optimize](vmelnykov): we can optimize this by processing tile by blocks
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx++) {
        tile_regs_acquire();
        matmul_tiles(
            cb_transpose_wh,
            cb_grad_output,
            /* tile_idx */ 0,
            /* tile_idx */ tile_idx,
            /* dst_reg_idx*/ 0,
            /* transpose */ 0);
        if (do_accumulate) {
            copy_tile_init(cb_grad_value);
            copy_tile(cb_grad_value, /* tile_idx */ 0, /* register idx */ 1U);

            add_binary_tile_init();
            add_binary_tile(0, 1U, 0);  // accumulate in register 0
        }
        tile_regs_commit();

        cb_pop_front(cb_grad_value, onetile);
        tile_regs_wait();
        cb_reserve_back(cb_grad_value, onetile);
        pack_tile(0, cb_grad_value);
        tile_regs_release();
    }
    cb_push_back(cb_grad_value, tiles_per_row);
    cb_pop_front(cb_transpose_wh, onetile);

    // for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
    //     tile_regs_acquire();
    //     for (uint32_t block_idx = 0; block_idx < block_size; block_idx++) {
    //         matmul_tiles(
    //             cb_attention_weights,
    //             cb_grad_output,
    //             /* tile_idx of A*/ 0,
    //             /* tile_idx of B*/ tile_idx + block_idx,
    //             /* dest_reg_idx */ block_idx);
    //     }
    //     tile_regs_commit();
    //     tile_regs_wait();
    //     if (do_accumulate) {
    //         cb_reserve_back(cb_mm_result_holder, block_size);
    //         pack_reconfig_data_format(cb_mm_result_holder);
    //         for (uint32_t block_idx = 0; block_idx < block_size; block_idx++) {
    //             pack_tile(block_idx, cb_mm_result_holder);
    //         }
    //         tile_regs_release();
    //         cb_push_back(cb_mm_result_holder, block_size);
    //         cb_wait_front(cb_mm_result_holder, block_size);

    //         tile_regs_acquire();
    //         add_tiles_init();
    //         for (uint32_t block_idx = 0; block_idx < block_size; block_idx++) {
    //             add_tiles(
    //                 cb_mm_result_holder,
    //                 cb_grad_value,
    //                 /* tile_idx of A*/ block_idx,
    //                 /* tile_idx of B*/ tile_idx + block_idx,
    //                 /* dest_reg_idx */ block_idx);
    //         }
    //         tile_regs_commit();
    //         tile_regs_wait();
    //         cb_pop_front(cb_mm_result_holder, block_size);
    //         cb_pop_front(cb_grad_value, block_size);
    //         cb_resereve_back(cb_grad_value, block_size);
    //         pack_reconfig_data_format(cb_grad_value);
    //         for (uint32_t block_idx = 0; block_idx < block_size; block_idx++) {
    //             pack_tile(block_idx, cb_grad_value);
    //         }
    //         tile_regs_release();
    //         cb_push_back(cb_grad_value, block_size);
    //     } else {
    //         cb_reserve_back(cb_grad_value, block_size);
    //         pack_reconfig_data_format(cb_grad_value);
    //         for (uint32_t block_idx = 0; block_idx < block_size; block_idx++) {
    //             pack_tile(block_idx, cb_grad_value);
    //         }
    //         tile_regs_release();
    //         cb_push_back(cb_grad_value, block_size);
    //     }
    // }
}
