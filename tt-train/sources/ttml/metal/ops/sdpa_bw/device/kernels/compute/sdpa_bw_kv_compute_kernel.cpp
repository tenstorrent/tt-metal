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

namespace NAMESPACE {

// ----------------------------------------------------------------------
// SDPA Backward Compute Kernel for Key and Value Gradients (dK, dV)
// ----------------------------------------------------------------------
//
// Forward pass (for reference):
//   P = softmax(Q @ K^T / sqrt(d) + mask)    // attention weights [B, H, S, S]
//   O = P @ V                                 // output [B, H, S, D]
//
// Backward pass computes dK and dV given dO (upstream gradient):
//   dP = dO @ V^T                             // gradient w.r.t. attention weights
//   u  = rowsum(dO ⊙ O)                       // per-row scalar for softmax backward
//   dS = P ⊙ (dP - u)                         // softmax backward (element-wise)
//   dV = P^T @ dO                             // gradient w.r.t. value
//   dK = (1/sqrt(d)) * dS^T @ Q               // gradient w.r.t. key
//
// Note: We apply the scale factor inside dS computation for numerical stability.
//
// Grouped Query Attention (GQA):
//   Multiple query heads share the same K/V head. Gradients from all query heads
//   in a group are accumulated into their shared K/V head's gradient.
//
// Processing order:
//   for each K/V row k:
//     for each query head h in group:
//       for each query row q:
//         P[q,k] = softmax(Q[q] @ K[k]^T / sqrt(d) + mask[q,k])  // recomputed
//         dV[k] += P[q,k]^T @ dO[q]                               // accumulate
//         u_scalar = rowsum(dO[q] ⊙ O[q])
//         dP[q,k] = dO[q] @ V[k]^T
//         dS[q,k] = P[q,k] * (dP[q,k] - u_scalar) * scale
//         dK[k] += dS[q,k]^T @ Q[q]                               // accumulate
// ----------------------------------------------------------------------

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);       // size of block (used in update_grad_value)
constexpr uint32_t qWt = get_compile_time_arg_val(2);              // num tile in inner dim (qWt == kWt == vWt)
constexpr uint32_t Ht = get_compile_time_arg_val(3);               // num_seq_len / TILE_H
constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // number of heads per group
constexpr uint32_t scaler_bits = get_compile_time_arg_val(5);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(6);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(7);  // used to transform mask from 0/-1 to 0/-inf

constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;         // Gradient w.r.t. output
constexpr uint32_t cb_attn_output = tt::CBIndex::c_1;         // Attention output from forward pass
constexpr uint32_t cb_query = tt::CBIndex::c_2;               // Original query
constexpr uint32_t cb_key = tt::CBIndex::c_3;                 // Original key
constexpr uint32_t cb_value = tt::CBIndex::c_4;               // Original value
#if defined(CAUSAL_MASK) || defined(USE_ATTN_MASK)
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;           // Original mask
#endif
constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;       // Forward pass intermediates
constexpr uint32_t cb_mat_mul_reduction = tt::CBIndex::c_7;   // Temporary computations
constexpr uint32_t cb_grad_value_accum = tt::CBIndex::c_8;    // L1 accumulator for grad_value
constexpr uint32_t cb_grad_key_accum = tt::CBIndex::c_9;      // L1 accumulator for grad_key
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_10;  // Recomputed attention weights = softmax(QK^T / sqrt(Et))
constexpr uint32_t cb_grad_attn_weights = tt::CBIndex::c_11;  // Gradient w.r.t. attention: dL/dP
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_12;        // Gradient w.r.t. QK scores
constexpr uint32_t cb_transpose_wh = tt::CBIndex::c_13;       // Transpose of attention weights
constexpr uint32_t cb_u_scalar_row = tt::CBIndex::c_14;       // u_scalar per row
constexpr uint32_t cb_grad_key = tt::CBIndex::c_15;           // Output: grad_K
constexpr uint32_t cb_grad_value = tt::CBIndex::c_16;         // Output: grad_V

const uint32_t onetile = 1U;

// in future optimization we can process data by chunks(for example 2 at once)
const uint32_t tiles_per_row = qWt;       // assuming qWt == kWt == vWt
const uint32_t num_of_interm_tiles = 2U;  // number of tiles in intermediates buffer per head

void MAIN {
    // Runtime args - needed for causal mask to know global position within sequence
    const uint32_t start_row = get_arg_val<uint32_t>(0);

    init_sfpu(cb_query, cb_key);
    binary_op_init_common(cb_grad_output, cb_query, cb_key);

    cb_wait_front(cb_mat_mul_reduction, onetile);

    mm_init(cb_query, cb_key, cb_attention_weights);

#ifdef CAUSAL_MASK
    // Wait for causal mask tile ONCE - it's generated by writer and will be reused for every diagonal
    cb_wait_front(cb_attn_mask, onetile);
#endif

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_key, tiles_per_row);
        cb_wait_front(cb_value, tiles_per_row);

#ifdef CAUSAL_MASK
        // Calculate global position for this K/V row
        const uint32_t global_row_idx = start_row + row;
        const uint32_t k_row_tile = global_row_idx % Ht;  // position within sequence (0 to Ht-1)

        // For causal mask: only process Q rows from k_row_tile to Ht-1
        // Q rows 0 to k_row_tile-1 have zero attention weights (can't attend to future keys)
        const uint32_t q_start_tile = k_row_tile;
        const uint32_t num_q_tiles_to_process = Ht - k_row_tile;
#else
        const uint32_t q_start_tile = 0;
        const uint32_t num_q_tiles_to_process = Ht;
#endif

        for (uint32_t head_idx = 0; head_idx < heads_per_group; ++head_idx) {
            const uint32_t matmul_accum_reg = 0;

            for (uint32_t q_idx = 0; q_idx < num_q_tiles_to_process; ++q_idx) {
                const uint32_t h = q_start_tile + q_idx;  // actual Q row tile index

                // Wait for Q, dO, O, mask and intermediates for this K/V row
                cb_wait_front(cb_query, tiles_per_row);
                cb_wait_front(cb_grad_output, tiles_per_row);
                cb_wait_front(cb_attn_output, tiles_per_row);

                // Step 1: Recompute attention scores Z = QK^T/sqrt(Et) + mask
                // (we will produce column of attention weights matrix while
                // streaming through Q, dO, O)
                reconfig_data_format(cb_query, cb_key);
                // This call is required to set up the matmul correctly
                mm_init_short(cb_query, cb_key, /* transpose */ 1);
                tile_regs_acquire();
                for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
                    matmul_tiles(
                        cb_query,
                        cb_key,
                        /* tile_idx */ tile_idx,
                        /* tile_idx */ tile_idx,
                        /* dst_reg_idx*/ matmul_accum_reg);  // accumulate in dest_reg 0
                }

#ifdef CAUSAL_MASK
                // For causal mask: apply triangular mask on diagonal tile (h == k_row_tile)
                // Writer generates causal mask tile once, reused for every diagonal
                if (h == k_row_tile) {
                    apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
                    // Don't pop - causal mask tile is reused for all diagonal positions
                } else {
                    // Off-diagonal (h > k_row_tile): just scale, no mask needed
                    binop_with_scalar_tile_init();
                    mul_unary_tile(matmul_accum_reg, scaler_bits);
                }
#elif defined(USE_ATTN_MASK)
                // Apply attention mask from DRAM
                // Transforms mask from 1/0 to 0/-inf and applies it on dest_reg
                apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
                cb_pop_front(cb_attn_mask, onetile);  // Pop each unique mask tile after use
#else
                // No mask: just scale
                binop_with_scalar_tile_init();
                mul_unary_tile(matmul_accum_reg, scaler_bits);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_attention_weights);
                pack_tile(matmul_accum_reg, cb_attention_weights);
                tile_regs_release();
                cb_push_back(cb_attention_weights, onetile);

                // Step 2: apply statistics inplace: P = softmax(Z) = softmax(QK^T / sqrt(Et))
                apply_statistics_inplace(cb_attention_weights, cb_intermediates, num_of_interm_tiles);

                // Step 3: Accumulate grad_V = Attention^T @ grad_output
                // For causal mask: first iteration is q_idx=0 (h=k_row_tile), head_idx=0
                update_grad_value(
                    cb_attention_weights,
                    cb_transpose_wh,
                    cb_grad_output,
                    cb_grad_value_accum,
                    tiles_per_row,
                    block_size,
                    /* do_accumulate */ q_idx > 0 || head_idx > 0);
                cb_wait_front(cb_grad_value_accum, tiles_per_row);

                // Step 4: calculate u_scalar_row = sum(dO * O) per row
                // TODO[optimization](vmelnykov): Calculate u_scalar_row once per query row and share with KV kernel
                // as additional intermediate (or keep as sharded L1 tensor)
                compute_u_scalar_row(
                    cb_grad_output, cb_attn_output, cb_u_scalar_row, cb_mat_mul_reduction, tiles_per_row, scaler_bits);

                // Step 5: compute grad w.r.t attention weights
                // dP = dO @ V^T (where dO is upstream grad_output)
                compute_grad_attn_weights(cb_grad_output, cb_value, tiles_per_row, cb_grad_attn_weights, scaler_bits);

                // Step 6: softmax backward block
                // dZ = (dP - u_scalar_row) * P (where P is attention weights, * - element-wise multiplication)
                compute_grad_scores(
                    cb_grad_attn_weights, cb_attention_weights, cb_u_scalar_row, scaler_bits, cb_grad_scores);

                // Step 7: compute grad w.r.t. key
                // dK = dS^T @ Q (scaling already applied in compute_grad_scores)
                update_grad_key(
                    cb_grad_scores,
                    cb_query,
                    cb_transpose_wh,
                    cb_grad_key_accum,
                    tiles_per_row,
                    block_size,
                    /* do_accumulate */ q_idx > 0 || head_idx > 0);
                cb_wait_front(cb_grad_key_accum, tiles_per_row);

                // Pop intermediate results used for computing dK and dV
                cb_pop_front(cb_u_scalar_row, onetile);
                cb_pop_front(cb_grad_attn_weights, onetile);
                cb_pop_front(cb_grad_scores, onetile);
                cb_pop_front(cb_attention_weights, onetile);
                cb_pop_front(cb_intermediates, num_of_interm_tiles);

                cb_pop_front(cb_query, tiles_per_row);
                cb_pop_front(cb_grad_output, tiles_per_row);
                cb_pop_front(cb_attn_output, tiles_per_row);
            }
        }

        // Push final grad_value and grad_key to output CBs
        pack_tiles_to_output(cb_grad_value_accum, cb_grad_value, tiles_per_row);
        pack_tiles_to_output(cb_grad_key_accum, cb_grad_key, tiles_per_row);

        cb_pop_front(cb_key, tiles_per_row);
        cb_pop_front(cb_value, tiles_per_row);
    }

#ifdef CAUSAL_MASK
    // Pop the causal mask tile after all rows are processed (was reused for every diagonal)
    cb_pop_front(cb_attn_mask, onetile);
#endif
}

}  // namespace NAMESPACE
