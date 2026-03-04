// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/compute/cb_api.h>
#include <api/compute/pack.h>
#include <api/compute/reconfig_data_format.h>
#include <api/compute/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "sdpa_bw_compute_utils.hpp"

// ----------------------------------------------------------------------
// SDPA Backward Compute Kernel for Query Gradient (dQ)
// ----------------------------------------------------------------------
//
// Forward pass (for reference):
//   P = softmax(Q @ K^T / sqrt(d) + mask)    // attention weights [B, H, S, S]
//   O = P @ V                                 // output [B, H, S, D]
//
// Backward pass computes dQ given dO (upstream gradient):
//   dP = dO @ V^T                             // gradient w.r.t. attention weights
//   u  = rowsum(dO ⊙ O)                       // per-row scalar for softmax backward
//   dS = P ⊙ (dP - u)                         // softmax backward (element-wise)
//   dQ = (1/sqrt(d)) * dS @ K                 // gradient w.r.t. query
//
// Note: We apply the scale factor inside dS computation for numerical stability.
//
// Processing order:
//   for each query row q:
//     compute u_scalar = rowsum(dO[q] ⊙ O[q])
//     for each K/V row k:
//       P[q,k] = softmax(Q[q] @ K[k]^T / sqrt(d) + mask[q,k])  // recomputed
//       dP[q,k] = dO[q] @ V[k]^T
//       dS[q,k] = P[q,k] * (dP[q,k] - u_scalar) * scale
//       dQ[q] += dS[q,k] @ K[k]                                 // accumulate
// ----------------------------------------------------------------------

// For standard mode: num_rows_per_core = rows to process
// For balanced mode: num_rows_per_core = num_pairs (each pair = 2 rows)
constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t qWt = get_compile_time_arg_val(1);              // num tile in inner dim (qWt == kWt == vWt)
constexpr uint32_t Ht = get_compile_time_arg_val(2);               // num_seq_len / TILE_H
constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(4);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(5);  // used to transform mask from 0/-1 to 0/-inf
constexpr uint32_t block_size = get_compile_time_arg_val(6);       // block size
constexpr uint32_t pairs_per_seq = Ht / 2;

constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;         // Gradient w.r.t. output
constexpr uint32_t cb_attn_output = tt::CBIndex::c_1;         // Attention output from forward pass
constexpr uint32_t cb_query = tt::CBIndex::c_2;               // Original query
constexpr uint32_t cb_key = tt::CBIndex::c_3;                 // Original key
constexpr uint32_t cb_value = tt::CBIndex::c_4;               // Original value
#if defined(CAUSAL_MASK) || defined(USE_ATTN_MASK)
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;  // Attention mask (causal or arbitrary)
#endif
constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;       // Forward pass intermediates
constexpr uint32_t cb_mat_mul_reduction = tt::CBIndex::c_7;   // Temporary computations
constexpr uint32_t cb_grad_query_accum = tt::CBIndex::c_8;    // L1 accumulator for grad_query
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_9;   // Recomputed attention weights = softmax(QK^T / sqrt(Et))
constexpr uint32_t cb_grad_attn_weights = tt::CBIndex::c_10;  // Gradient w.r.t. attention: dL/dP
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_11;        // Gradient w.r.t. QK scores
constexpr uint32_t cb_u_scalar_row = tt::CBIndex::c_12;       // u_scalar per row
constexpr uint32_t cb_grad_query = tt::CBIndex::c_13;         // Output: grad_Q

const uint32_t tiles_per_row = qWt;       // number of tiles per row (qWt == kWt == vWt)
const uint32_t num_of_interm_tiles = 2U;  // number of tiles in intermediates buffer per head

/**
 * Process a single row of the SDPA backward Q computation.
 * This function handles the full backward attention computation for one query row:
 * - Recompute P = softmax(Q @ K^T / sqrt(d) + mask) using stored intermediates
 * - Compute dP = dO @ V^T
 * - Compute dS = P * (dP - u) * scale (softmax backward)
 * - Accumulate dQ += dS @ K
 *
 * @param global_row_idx The global row index (across all batches/heads/sequences)
 */
FORCE_INLINE void process_single_row(uint32_t global_row_idx) {
    cb_wait_front(cb_attn_output, tiles_per_row);
    cb_wait_front(cb_grad_output, tiles_per_row);
    cb_wait_front(cb_query, tiles_per_row);

    compute_u_scalar_row(
        cb_grad_output, cb_attn_output, cb_u_scalar_row, cb_mat_mul_reduction, tiles_per_row, scaler_bits);

    const uint32_t q_row_tile = global_row_idx % Ht;

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    const uint32_t num_kv_tiles_to_process = q_row_tile + 1;
#else
    const uint32_t num_kv_tiles_to_process = Ht;
#endif

    const uint32_t matmul_accum_reg = 0;
    for (uint32_t h = 0; h < num_kv_tiles_to_process; ++h) {
        cb_wait_front(cb_key, tiles_per_row);
        cb_wait_front(cb_value, tiles_per_row);

        reconfig_data_format(cb_query, cb_key);
        mm_init_short(cb_query, cb_key, /*transpose*/ 1);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
            matmul_tiles(
                cb_query,
                cb_key,
                /* tile_idx */ tile_idx,
                /* tile_idx */ tile_idx,
                /* dst_reg_idx*/ matmul_accum_reg);
        }

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
        if (h == q_row_tile) {
            apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
        } else {
            binop_with_scalar_tile_init();
            mul_unary_tile(matmul_accum_reg, scaler_bits);
        }
#elif defined(USE_ATTN_MASK)
        apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
        cb_pop_front(cb_attn_mask, onetile);
#else
        binop_with_scalar_tile_init();
        mul_unary_tile(matmul_accum_reg, scaler_bits);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_attention_weights);
        pack_tile(matmul_accum_reg, cb_attention_weights);
        tile_regs_release();
        cb_push_back(cb_attention_weights, onetile);

        apply_statistics_inplace(cb_attention_weights, cb_intermediates, num_of_interm_tiles);

        compute_grad_attn_weights(cb_grad_output, cb_value, tiles_per_row, cb_grad_attn_weights, scaler_bits);

        compute_grad_scores(cb_grad_attn_weights, cb_attention_weights, cb_u_scalar_row, scaler_bits, cb_grad_scores);

        update_grad_query(
            cb_grad_scores,
            cb_key,
            cb_grad_query_accum,
            tiles_per_row,
            block_size,
            /* do_accumulate */ (h > 0));
        cb_wait_front(cb_grad_query_accum, tiles_per_row);

        cb_pop_front(cb_key, tiles_per_row);
        cb_pop_front(cb_value, tiles_per_row);
        cb_pop_front(cb_attention_weights, onetile);
        cb_pop_front(cb_grad_attn_weights, onetile);
    }

    pack_tiles_to_output(cb_grad_query_accum, cb_grad_query, tiles_per_row);

    cb_pop_front(cb_u_scalar_row, onetile);
    cb_pop_front(cb_intermediates, num_of_interm_tiles);
    cb_pop_front(cb_query, tiles_per_row);
    cb_pop_front(cb_attn_output, tiles_per_row);
    cb_pop_front(cb_grad_output, tiles_per_row);
}

void kernel_main() {
    // Runtime args
    // For standard mode: arg0 = start_row
    // For balanced mode: arg0 = start_pair_idx, arg1 = num_pairs
    const uint32_t start_idx = get_arg_val<uint32_t>(0);

#ifdef BALANCED_PARALLELISM
    const uint32_t num_pairs = get_arg_val<uint32_t>(1);
#endif

    init_sfpu(cb_query, cb_key);
    binary_op_init_common(cb_grad_output, cb_query, cb_key);

    cb_wait_front(cb_mat_mul_reduction, onetile);
    mm_init(cb_query, cb_key, cb_attention_weights);

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Wait for causal mask tile ONCE - it's generated by writer and will be reused for every diagonal
    cb_wait_front(cb_attn_mask, onetile);
#endif

#ifdef BALANCED_PARALLELISM
    for (uint32_t p = 0; p < num_pairs; ++p) {
        const uint32_t global_pair_idx = start_idx + p;

        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

        process_single_row(light_global_row);
        process_single_row(heavy_global_row);
    }
#else
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        const uint32_t global_row_idx = start_idx + row;
        process_single_row(global_row_idx);
    }
#endif

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Pop the causal mask tile after all rows are processed (was reused for every diagonal)
    cb_pop_front(cb_attn_mask, onetile);
#endif
}
