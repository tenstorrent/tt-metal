// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <api/compute/reg_api.h>
#include <api/debug/dprint.h>

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/common/sdpa_compute_utils_common.hpp"

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
    copy_init(cb_attn_mask);
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

    compute_kernel_hw_startup(cb_attention_weights, cb_intermediates, cb_attention_weights);
    bcast_init<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(cb_attention_weights, cb_intermediates);

    reconfig_data_format(cb_attention_weights, cb_intermediates);
    tile_regs_acquire();
    sub_bcast_cols_init(cb_attention_weights, cb_intermediates);
    sub_tiles_bcast_cols(cb_attention_weights, cb_intermediates, /* tile_idx */ 0, /* tile_idx */ 0, working_reg);

    sdpa_exp_tile_init();
    sdpa_exp_tile(working_reg);
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
// Loads lse from cb_intermediates via column-broadcast and computes in-place using SFPU.
// Scores stay in DST at full FP32 — no CB roundtrip, no TF32 truncation.
// Must be called inside a tile_regs_acquire/commit block, after matmul + mask.
void apply_softmax_statistics_on_dst(const uint32_t scores_reg, const uint32_t cb_intermediates) {
    const uint32_t lse_reg = scores_reg + 1U;

    // Reconfigure only SrcB for the B2D broadcast (Float32 intermediates).
    // SrcA is left as-is from the preceding matmul to preserve its format.
    reconfig_data_format_srcb(cb_intermediates);

    // Lightweight MOP reinit for unary_bcast<COL> with B2D path.
    // Only reprograms UNPACK and MATH MOPs — does NOT reset MATH-PACK sync or PACK dest,
    // so it is safe inside an acquire block (unlike the full unary_bcast_init).
    UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, cb_intermediates)));
    MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, DST_ACCUM_MODE, BroadcastType::COL>(
        cb_intermediates)));

    unary_bcast<BroadcastType::COL>(cb_intermediates, /* tile_idx */ 0, lse_reg);

    sub_binary_tile_init();
    sub_binary_tile(scores_reg, lse_reg, scores_reg);

    sdpa_exp_tile_init();
    sdpa_exp_tile(scores_reg);
}

// Computes the per-row scalar u = sum(dO * O) needed for softmax backward.
// This is part of the softmax gradient: dS = P * (dP - u), where u = sum(P * dP) per row.
// Since O = P @ V, we have dP = dO @ V^T, and u = sum(dO * O) row-wise.
// The reduction is done via matmul with a column of ones (cb_mat_mul_reduction).
// Computes u_scalar = rowsum(dO * O) and packs the result to cb_u_scalar_row.
// When cb_u_scaler_output != 0, also packs a second copy to that CB (for DRAM flush
// to the KV kernel). Both packs happen directly from DST at full FP32 precision,
// avoiding a separate copy-pack cycle.
void compute_u_scalar_row(
    const uint32_t cb_grad_output,
    const uint32_t cb_attn_output,
    /*output result*/ const uint32_t cb_u_scalar_row,
    /*mutmul reduction*/ const uint32_t cb_mat_mul_reduction,
    const uint32_t tiles_per_row,
    const uint32_t scaler_bits,
    const uint32_t cb_u_scaler_output) {
    const uint32_t accum_register = 0;
    // using binary_tiles_init function instead of specific mul_init() because specific one doesn't support
    // accumulation to dest regs
    reconfig_data_format(cb_grad_output, cb_attn_output);
    binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(cb_grad_output, cb_attn_output, /*acc_to_dest*/ true);
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
    matmul_init(cb_u_scalar_row, cb_mat_mul_reduction, /* transpose */ 0);
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

    cb_reserve_back(cb_u_scaler_output, onetile);
    pack_reconfig_data_format(cb_u_scalar_row, cb_u_scaler_output);
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
    const uint32_t cb_prev_pack,
    const uint32_t scaler_bits) {
    reconfig_data_format(cb_grad_output, cb_value);
    // This call is required to set up the matmul correctly
    matmul_init(cb_grad_output, cb_value, /* transpose */ 1);
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
    // 2-arg pack_reconfig: skip reprogram if new PACK format matches the previously-configured one.
    pack_reconfig_data_format(cb_prev_pack, cb_grad_attn_weights);
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
    sub_bcast_cols_init(cb_grad_attn_weights, cb_u_scalar_row);
    sub_tiles_bcast_cols(
        cb_grad_attn_weights,
        cb_u_scalar_row,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        grad_reg);

    reconfig_data_format_srca(cb_grad_attn_weights, cb_attention_weights);
    copy_init(cb_attention_weights, /*transpose=*/0);
    copy_tile(cb_attention_weights, /* tile_idx */ 0, /* register idx */ attn_weights_reg);

    mul_binary_tile_init();
    mul_binary_tile(grad_reg, attn_weights_reg, grad_reg);

    binop_with_scalar_tile_init();
    mul_unary_tile(grad_reg, scaler_bits);

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_grad_scores, onetile);
    // 2-arg pack_reconfig: skip reprogram if new PACK format matches the previously-configured one.
    pack_reconfig_data_format(cb_grad_attn_weights, cb_grad_scores);
    pack_tile(grad_reg, cb_grad_scores);
    tile_regs_release();
    cb_push_back(cb_grad_scores, onetile);
}

// KV-kernel variant of the softmax backward: computes dS and, from the same DST acquire
// region, produces both transposed matmul operands:
//   dS   = P * (dP - u) * scale         (same op order as compute_grad_scores — bit-identical)
//   dS^T -> cb_grad_scores_transposed   (for dK = dS^T @ Q)
//   P^T  -> cb_attn_weights_transposed  (for dV = P^T @ dO)
// P is already resident in DST for the dS multiply (full-FP32 copy_tile from
// cb_attention_weights, which mul_binary_tile leaves intact), and dS is produced in DST —
// so both are transposed in place with the 32-bit transpose_dest and packed directly.
// This removes the former pack->CB->unpack roundtrip per transpose (transpose_tile_fpu)
// from the inner loop. The transposes are lossless permutations of FP32 DST values, so
// results are bit-identical to the UnpackToDestFp32 roundtrip path this replaces — that
// path was already exact; the precision gain over the original TF32-SrcA transposes came
// from the earlier UnpackToDestFp32 change, not from this restructuring.
//
// transpose_dest_init records math replay-buffer slots [16,32) and reprograms the math
// MOP/addr-mods. That is safe here: every MOP-driven op of this region (bcast sub,
// copy_tile) has already executed by then, and downstream consumers re-init their own
// state per call (matmul_init per phase, sub_bcast_cols_init per iteration).
void compute_grad_scores_and_transposes(
    const uint32_t cb_grad_attn_weights,
    const uint32_t cb_attention_weights,
    const uint32_t cb_u_scalar_row,
    const uint32_t scaler_bits,
    /* output dS^T */ const uint32_t cb_grad_scores_transposed,
    /* output P^T */ const uint32_t cb_attn_weights_transposed) {
    cb_wait_front(cb_grad_attn_weights, onetile);
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_u_scalar_row, onetile);

    const uint32_t grad_reg = 0;
    const uint32_t attn_weights_reg = 1U;

    tile_regs_acquire();
    reconfig_data_format(cb_grad_attn_weights, cb_u_scalar_row);
    sub_bcast_cols_init(cb_grad_attn_weights, cb_u_scalar_row);
    sub_tiles_bcast_cols(
        cb_grad_attn_weights,
        cb_u_scalar_row,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        grad_reg);

    reconfig_data_format_srca(cb_grad_attn_weights, cb_attention_weights);
    copy_init(cb_attention_weights);
    copy_tile(cb_attention_weights, /* tile_idx */ 0, /* register idx */ attn_weights_reg);

    mul_binary_tile_init();
    mul_binary_tile(grad_reg, attn_weights_reg, grad_reg);

    binop_with_scalar_tile_init();
    mul_unary_tile(grad_reg, scaler_bits);

    // In-place 32-bit tile transposes (DST holds FP32; one init serves both calls).
    transpose_dest_init</* is_32bit */ true>(cb_attention_weights);
    transpose_dest</* is_32bit */ true>(grad_reg);          // dS -> dS^T
    transpose_dest</* is_32bit */ true>(attn_weights_reg);  // P  -> P^T

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_grad_scores_transposed, onetile);
    cb_reserve_back(cb_attn_weights_transposed, onetile);
    // 2-arg pack_reconfig: skip reprogram if new PACK format matches the previously-configured one.
    pack_reconfig_data_format(cb_grad_attn_weights, cb_grad_scores_transposed);
    pack_tile(grad_reg, cb_grad_scores_transposed);
    pack_reconfig_data_format(cb_grad_scores_transposed, cb_attn_weights_transposed);
    pack_tile(attn_weights_reg, cb_attn_weights_transposed);
    tile_regs_release();
    cb_push_back(cb_grad_scores_transposed, onetile);
    cb_push_back(cb_attn_weights_transposed, onetile);
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

    // 2-arg pack_reconfig: skip reprogram if new PACK format matches the previously-configured one.
    pack_reconfig_data_format(cb_grad_scores, cb_grad_query_accum);
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
        reconfig_data_format_srca(cb_grad_query_accum, cb_key);
        matmul_init(cb_grad_scores, cb_key, /*transpose*/ 0);
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
// P^T is produced in DST and packed to cb_attn_weights_transposed by
// compute_grad_scores_and_transposes — no transpose roundtrip here.
// Uses L1 accumulation to accumulate across sequence blocks and query heads (do_accumulate=true).
void update_grad_value(
    const uint32_t cb_attn_weights_transposed,
    const uint32_t cb_grad_output,
    const uint32_t cb_grad_value_accum,
    const uint32_t tiles_per_row,
    const uint32_t block_size,
    const bool do_accumulate = false) {
    // grad_V = Attention^T @ grad_output
    cb_wait_front(cb_attn_weights_transposed, onetile);

    // 2-arg pack_reconfig: skip reprogram if new PACK format matches the previously-configured one.
    // The previous PACK target is cb_attn_weights_transposed (packed last by
    // compute_grad_scores_and_transposes).
    pack_reconfig_data_format(cb_attn_weights_transposed, cb_grad_value_accum);
    // First iteration: reserve space for result
    // Subsequent iterations: enable L1 accumulation to add to existing values
    if (!do_accumulate) {
        cb_reserve_back(cb_grad_value_accum, tiles_per_row);
    } else {
        pack_reconfig_l1_acc(true);
    }

    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_grad_value_accum, cb_grad_output);
        matmul_init(cb_attn_weights_transposed, cb_grad_output, /*transpose*/ 0);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_attn_weights_transposed,
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
    cb_pop_front(cb_attn_weights_transposed, onetile);
}

// Computes gradient w.r.t. Key tensor: dK = dS^T @ Q
// dS^T is produced in DST and packed to cb_grad_scores_transposed by
// compute_grad_scores_and_transposes — no transpose roundtrip here.
// cb_prev_pack / cb_prev_srca are the actual previous PACK target and UNPACK-A operand
// (cb_grad_value_accum / cb_grad_output when called after update_grad_value), so the
// conditional reconfigs skip when formats match.
// Uses L1 accumulation to accumulate across sequence blocks and query heads (do_accumulate=true).
void update_grad_key(
    const uint32_t cb_grad_scores_transposed,
    const uint32_t cb_query,
    const uint32_t cb_grad_key_accum,
    const uint32_t tiles_per_row,
    const uint32_t block_size,
    const uint32_t cb_prev_pack,
    const uint32_t cb_prev_srca,
    const bool do_accumulate = false) {
    cb_wait_front(cb_grad_scores_transposed, onetile);

    // 2-arg pack_reconfig: skip reprogram if new PACK format matches the previously-configured one.
    pack_reconfig_data_format(cb_prev_pack, cb_grad_key_accum);
    if (!do_accumulate) {
        cb_reserve_back(cb_grad_key_accum, tiles_per_row);
    } else {
        pack_reconfig_l1_acc(true);
    }

    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_prev_srca, cb_query);
        matmul_init(cb_grad_scores_transposed, cb_query, /*transpose*/ 0);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_grad_scores_transposed,
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
    cb_pop_front(cb_grad_scores_transposed, onetile);
}
