// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <api/compute/reg_api.h>
#include <api/debug/dprint.h>
#include <api/debug/dprint_tensix.h>

#include <cstdint>

#include "api/compute/bcast.h"
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
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh_dest.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

// now we have to multiply result by scaler factor and then apply mask
// we need to transform the attention mask for use in softmax:
// The input `attn_mask` contains 1.0 for valid (keep) positions and 0.0 for masked (drop) positions.
// To convert this into a format compatible with softmax masking:
//   - Subtract 1.0 from the mask, so values become 0.0 (keep) and -1.0 (mask).
//   - Multiply by infinity, resulting in 0.0 for valid entries and -inf for
//   masked ones.
// This way, after applying softmax, masked positions will effectively become zero,
// and only the unmasked positions will retain meaningful attention weights
//
// Note: Does NOT pop the mask tile - caller must pop explicitly when done with the tile.
// This allows reusing the same mask tile for causal masks.
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
}

// Recomputes attention weights from pre-softmax scores using stored logsumexp.
// Given raw attention scores S and intermediates[0] = lse (logsumexp),
// computes: P = exp(S - lse)
void apply_statistics_inplace(const uint32_t cb_attention_weights, const uint32_t cb_intermediates) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_intermediates, onetile);

    const uint32_t working_reg = 0;

    init_bcast<ELWSUB, BroadcastType::COL>(cb_attention_weights, cb_intermediates, cb_attention_weights);

    reconfig_data_format(cb_attention_weights, cb_intermediates);
    tile_regs_acquire();
    sub_bcast_cols_init_short(cb_attention_weights, cb_intermediates);
    sub_tiles_bcast_cols(cb_attention_weights, cb_intermediates, /* tile_idx */ 0, /* tile_idx */ 0, working_reg);

    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(working_reg);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_attention_weights, onetile);
    cb_reserve_back(cb_attention_weights, onetile);
    pack_reconfig_data_format(cb_attention_weights);
    pack_tile(working_reg, cb_attention_weights);
    tile_regs_release();
    cb_push_back(cb_attention_weights, onetile);
}

// Applies softmax statistics (exp(S - lse)) directly on DST registers.
// Scores must already be in DST[scores_reg] from the matmul + mask/scale.
// Loads lse from cb_intermediates via copy_tile, then uses sfpu_sub_bcast_col
// to subtract column 0 (lse values) broadcast across all columns — entirely in DST.
// Scores stay in DST at full FP32 — no CB roundtrip, no TF32 truncation.
// Must be called inside a tile_regs_acquire/commit block, after matmul + mask.
//
// Context: this function is called inside a tile_regs_acquire block AFTER:
//   1. matmul_tiles() accumulated Q @ K^T into DST[scores_reg] (FPU matmul)
//   2. mul_unary_tile(scores_reg, scaler_bits) scaled the result (SFPU scalar mul)
//   (optionally: apply_mask_on_reg() for causal masking)
//
// At this point DST[scores_reg] contains the scaled scores in FP32.
// We want to compute: DST[scores_reg] = exp(DST[scores_reg] - bcast_col(lse))
//
// sfpu_sub_bcast_col should do exactly this: subtract column 0 of DST[lse_reg]
// broadcast across all 32 columns from DST[scores_reg], entirely within DST.
// However, when tested it appears to be a no-op (DST[scores_reg] unchanged after call).
// cb_intermediates is Float32, fp32_dest_acc_en = true.
void apply_softmax_statistics_on_dst(const uint32_t scores_reg, const uint32_t cb_intermediates) {
    const uint32_t lse_reg = scores_reg + 1U;

    // --- sfpu_sub_bcast_col attempt (BROKEN — no-op observed) ---
    // Step 1: Load lse tile into DST[lse_reg] via copy_tile.
    //   cb_intermediates contains logsumexp values in column 0 (from forward pass row-reduce).
    //   copy_tile places the full tile into DST[lse_reg] without any broadcast.
    reconfig_data_format(cb_intermediates, cb_intermediates);
    copy_tile_init(cb_intermediates);
    copy_tile(cb_intermediates, /* tile_idx */ 0, lse_reg);

    // Debug: dump DST state before and after sfpu_sub_bcast_col
    DPRINT << "=== apply_softmax_statistics_on_dst ===" << ENDL();
    DPRINT << "DST[scores_reg=" << scores_reg << "] BEFORE sfpu_sub_bcast_col:" << ENDL();
    dprint_tensix_dest_reg(scores_reg);
    DPRINT << "DST[lse_reg=" << lse_reg << "] (loaded via copy_tile from cb_intermediates):" << ENDL();
    dprint_tensix_dest_reg(lse_reg);

    // Step 2: SFPU column-broadcast subtract.
    //   Expected: DST[scores_reg][r][c] -= DST[lse_reg][r][0] for all c in [0..31]
    //   Observed: DST[scores_reg] is unchanged after this call.
    sfpu_sub_bcast_col_init();
    sfpu_sub_bcast_col(scores_reg, lse_reg);

    DPRINT << "DST[scores_reg=" << scores_reg << "] AFTER sfpu_sub_bcast_col:" << ENDL();
    dprint_tensix_dest_reg(scores_reg);

    // Step 3: exp in-place
    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(scores_reg);
}

// Transposes a single tile using the FPU transpose_wh path (reads via SrcA).
inline void transpose_tile_fpu(const uint32_t cb_input, /*output cb*/ const uint32_t cb_transpose_wh) {
    cb_wait_front(cb_input, onetile);

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
// Uses sfpu_reduce<SUM, REDUCE_ROW> to reduce rows directly in DST at full FP32,
// eliminating the previous matmul-with-ones CB roundtrip.
// Output: row sums stored in column 0 of the tile (compatible with sfpu_sub_bcast_col).
void compute_u_scalar_row(
    const uint32_t cb_grad_output,
    const uint32_t cb_attn_output,
    /*output result*/ const uint32_t cb_u_scalar_row,
    /*unused, kept for API compat*/ const uint32_t cb_mat_mul_reduction,
    const uint32_t tiles_per_row,
    const uint32_t scaler_bits,
    const uint32_t cb_u_scaler_output) {
    const uint32_t accum_register = 0;

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

    // Row reduction via SFPU: reduces all 32 columns to column 0 at full FP32.
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(accum_register);

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_u_scalar_row, onetile);
    pack_reconfig_data_format(cb_u_scalar_row);
    pack_tile(accum_register, cb_u_scalar_row);

    cb_reserve_back(cb_u_scaler_output, onetile);
    pack_reconfig_data_format(cb_u_scaler_output);
    pack_tile(accum_register, cb_u_scaler_output);
    cb_push_back(cb_u_scaler_output, onetile);

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

    tile_regs_acquire();
    reconfig_data_format(cb_grad_attn_weights, cb_u_scalar_row);
    sub_bcast_cols_init_short(cb_grad_attn_weights, cb_u_scalar_row);
    sub_tiles_bcast_cols(
        cb_grad_attn_weights,
        cb_u_scalar_row,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        grad_reg);

    copy_tile_to_dst_init_short_with_dt(cb_grad_attn_weights, cb_attention_weights, /*transpose=*/0);
    copy_tile(cb_attention_weights, /* tile_idx */ 0, /* register idx */ attn_weights_reg);

    mul_binary_tile_init();
    mul_binary_tile(grad_reg, attn_weights_reg, grad_reg);

    binop_with_scalar_tile_init();
    mul_unary_tile(grad_reg, scaler_bits);

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_grad_scores, onetile);
    pack_reconfig_data_format(cb_grad_scores);
    pack_tile(grad_reg, cb_grad_scores);
    tile_regs_release();
    cb_push_back(cb_grad_scores, onetile);
}

// Fused: computes dP = dO @ V^T and then dS = P * (dP - u) * scale.
// Eliminates the separate compute_grad_attn_weights call. Uses cb_grad_attn_weights
// as an internal temporary (single L1 roundtrip within the compute kernel, no DRAM).
// The subtraction uses the proven FPU sub_tiles_bcast_cols path.
void compute_grad_scores_fused(
    const uint32_t cb_grad_output,
    const uint32_t cb_value,
    const uint32_t tiles_per_row,
    const uint32_t cb_attention_weights,
    const uint32_t cb_u_scalar_row,
    const uint32_t scaler_bits,
    /* output */ const uint32_t cb_grad_scores) {
    // Phase 1: matmul dO @ V^T → dP, pack to internal temp
    reconfig_data_format(cb_grad_output, cb_value);
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
    cb_reserve_back(cb_grad_scores, onetile);
    pack_reconfig_data_format(cb_grad_scores);
    pack_tile(0, cb_grad_scores);
    tile_regs_release();
    cb_push_back(cb_grad_scores, onetile);

    // Phase 2: dS = P * (dP - u) * scale
    cb_wait_front(cb_grad_scores, onetile);
    cb_wait_front(cb_u_scalar_row, onetile);
    cb_wait_front(cb_attention_weights, onetile);

    const uint32_t grad_reg = 0;
    const uint32_t attn_weights_reg = 1U;

    tile_regs_acquire();
    reconfig_data_format(cb_grad_scores, cb_u_scalar_row);
    sub_bcast_cols_init_short(cb_grad_scores, cb_u_scalar_row);
    sub_tiles_bcast_cols(
        cb_grad_scores,
        cb_u_scalar_row,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        grad_reg);

    copy_tile_to_dst_init_short_with_dt(cb_grad_scores, cb_attention_weights, /*transpose=*/0);
    copy_tile(cb_attention_weights, /* tile_idx */ 0, /* register idx */ attn_weights_reg);

    mul_binary_tile_init();
    mul_binary_tile(grad_reg, attn_weights_reg, grad_reg);

    binop_with_scalar_tile_init();
    mul_unary_tile(grad_reg, scaler_bits);

    tile_regs_commit();

    cb_pop_front(cb_grad_scores, onetile);

    tile_regs_wait();
    cb_reserve_back(cb_grad_scores, onetile);
    pack_reconfig_data_format(cb_grad_scores);
    pack_tile(grad_reg, cb_grad_scores);
    tile_regs_release();
    cb_push_back(cb_grad_scores, onetile);
}

// Computes gradient w.r.t. Query tensor: dQ = dS @ K
// where dS = scaled gradient w.r.t. scores (already includes 1/sqrt(d_k) scaling).
// Uses L1 accumulation to accumulate across sequence blocks (do_accumulate=true).
void update_grad_query(
    const uint32_t cb_grad_scores,
    const uint32_t cb_key,
    const uint32_t cb_grad_query_accum,
    const uint32_t tiles_per_row,
    const uint32_t block_size,
    const bool do_accumulate = false) {
    cb_wait_front(cb_grad_scores, onetile);

    pack_reconfig_data_format(cb_grad_query_accum);
    // First iteration: reserve space for result
    // Subsequent iterations: enable L1 accumulation to add to existing values
    if (!do_accumulate) {
        cb_reserve_back(cb_grad_query_accum, tiles_per_row);
    } else {
        // This function would ideally be called after other initialization functions that initialize the packer for a
        // specific operation.
        pack_reconfig_l1_acc(true);
    }

    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
        tile_regs_acquire();
        mm_init_short_with_dt(cb_grad_scores, cb_key, cb_grad_query_accum, /*transpose*/ 0);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_grad_scores,
                cb_key,
                /* tile_idx */ 0,
                /* tile_idx */ tile_idx + block_idx,
                /* dst_reg_idx*/ block_idx);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(/*dst_reg_idx*/ block_idx, cb_grad_query_accum);
        }
        tile_regs_release();
    }

    if (do_accumulate) {
        pack_reconfig_l1_acc(false);
        cb_pop_front(cb_grad_query_accum, tiles_per_row);
        cb_reserve_back(cb_grad_query_accum, tiles_per_row);
    }

    cb_push_back(cb_grad_query_accum, tiles_per_row);
    cb_pop_front(cb_grad_scores, onetile);
}

// Computes gradient w.r.t. Value tensor: dV = P^T @ dO
// Uses L1 accumulation to accumulate across sequence blocks and query heads (do_accumulate=true).
// Uses cb_transpose_wh as scratch space for transposed attention weights.
void update_grad_value(
    const uint32_t cb_attention_weights,
    const uint32_t cb_transpose_wh,
    const uint32_t cb_grad_output,
    const uint32_t cb_grad_value_accum,
    const uint32_t tiles_per_row,
    const uint32_t block_size,
    const bool do_accumulate = false) {
    transpose_tile_fpu(cb_attention_weights, cb_transpose_wh);

    // grad_V = Attention^T @ grad_output
    cb_wait_front(cb_transpose_wh, onetile);

    pack_reconfig_data_format(cb_grad_value_accum);
    // First iteration: reserve space for result
    // Subsequent iterations: enable L1 accumulation to add to existing values
    if (!do_accumulate) {
        cb_reserve_back(cb_grad_value_accum, tiles_per_row);
    } else {
        pack_reconfig_l1_acc(true);
    }

    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
        tile_regs_acquire();
        mm_init_short_with_dt(cb_transpose_wh, cb_grad_output, cb_grad_value_accum, /*transpose*/ 0);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_transpose_wh,
                cb_grad_output,
                /* tile_idx */ 0,
                /* tile_idx */ tile_idx + block_idx,
                /* dst_reg_idx*/ block_idx);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(/*dst_reg_idx*/ block_idx, cb_grad_value_accum);
        }
        tile_regs_release();
    }

    if (do_accumulate) {
        pack_reconfig_l1_acc(false);
        cb_pop_front(cb_grad_value_accum, tiles_per_row);
        cb_reserve_back(cb_grad_value_accum, tiles_per_row);
    }

    cb_push_back(cb_grad_value_accum, tiles_per_row);
    cb_pop_front(cb_transpose_wh, onetile);
}

// Computes gradient w.r.t. Key tensor: dK = dS^T @ Q
// Uses L1 accumulation to accumulate across sequence blocks and query heads (do_accumulate=true).
// Uses cb_transpose_wh as scratch space for transposed grad scores.
void update_grad_key(
    const uint32_t cb_grad_scores,
    const uint32_t cb_query,
    const uint32_t cb_transpose_wh,
    const uint32_t cb_grad_key_accum,
    const uint32_t tiles_per_row,
    const uint32_t block_size,
    const bool do_accumulate = false) {
    transpose_tile_fpu(cb_grad_scores, cb_transpose_wh);
    cb_wait_front(cb_transpose_wh, onetile);

    pack_reconfig_data_format(cb_grad_key_accum);
    if (!do_accumulate) {
        cb_reserve_back(cb_grad_key_accum, tiles_per_row);
    } else {
        pack_reconfig_l1_acc(true);
    }

    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
        tile_regs_acquire();
        mm_init_short_with_dt(cb_transpose_wh, cb_query, cb_grad_key_accum, /*transpose*/ 0);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_transpose_wh,
                cb_query,
                /* tile_idx */ 0,
                /* tile_idx */ tile_idx + block_idx,
                /* dst_reg_idx*/ block_idx);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(/*dst_reg_idx*/ block_idx, cb_grad_key_accum);
        }
        tile_regs_release();
    }

    if (do_accumulate) {
        pack_reconfig_l1_acc(false);
        cb_pop_front(cb_grad_key_accum, tiles_per_row);
        cb_reserve_back(cb_grad_key_accum, tiles_per_row);
    }

    cb_push_back(cb_grad_key_accum, tiles_per_row);
    cb_pop_front(cb_transpose_wh, onetile);
}
