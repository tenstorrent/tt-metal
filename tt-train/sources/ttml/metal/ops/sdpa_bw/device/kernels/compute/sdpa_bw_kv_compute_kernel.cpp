// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/cb_api.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reconfig_data_format.h>
#include <compute_kernel_api/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh.h"
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
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;           // Original mask
constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;       // Forward pass intermediates
constexpr uint32_t cb_mat_mul_reduction = tt::CBIndex::c_7;   // Temporary computations
constexpr uint32_t cb_prev_grad_value = tt::CBIndex::c_8;     // used for holding previous grad value
constexpr uint32_t cb_cur_grad_value = tt::CBIndex::c_9;      // used for holding current grad value
constexpr uint32_t cb_prev_grad_key = tt::CBIndex::c_10;      // used for holding previous grad key
constexpr uint32_t cb_cur_grad_key = tt::CBIndex::c_11;       // used for holding current grad key
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_12;  // Recomputed attention weights = softmax(QK^T / sqrt(Et))
constexpr uint32_t cb_grad_attn_weights = tt::CBIndex::c_13;  // Gradient w.r.t. attention: dL/dP
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_14;        // Gradient w.r.t. QK scores
constexpr uint32_t cb_transpose_wh = tt::CBIndex::c_15;       // Transpose of attention weights
constexpr uint32_t cb_u_scalar_row = tt::CBIndex::c_16;       // u_scalar per row
constexpr uint32_t cb_grad_key = tt::CBIndex::c_17;           // Output: grad_K
constexpr uint32_t cb_grad_value = tt::CBIndex::c_18;         // Output: grad_V

const uint32_t onetile = 1U;

// in future optimization we can process data by chunks(for example 2 at once)
const uint32_t tiles_per_row = qWt;       // assuming qWt == kWt == vWt
const uint32_t num_of_interm_tiles = 2U;  // number of tiles in intermediates buffer per head

void MAIN {
    init_sfpu(cb_query, cb_key);
    binary_op_init_common(cb_grad_output, cb_query, cb_key);

    cb_wait_front(cb_mat_mul_reduction, onetile);

    mm_init(cb_query, cb_key, cb_attention_weights);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_key, tiles_per_row);
        cb_wait_front(cb_value, tiles_per_row);

        uint32_t alias_cb_prev_grad_value = cb_prev_grad_value;
        uint32_t alias_cb_cur_grad_value = cb_cur_grad_value;
        uint32_t alias_cb_prev_grad_key = cb_prev_grad_key;
        uint32_t alias_cb_cur_grad_key = cb_cur_grad_key;

        for (uint32_t head_idx = 0; head_idx < heads_per_group; ++head_idx) {
            const uint32_t matmul_accum_reg = 0;

            for (uint32_t h = 0; h < Ht; ++h) {
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

                /*
                 * apply attention mask on dest_reg.
                 * function assumes that dest_reg is in acquired state via *acquire_dst* call
                 * function transforms mask from 1/0 to 0/-inf and applies it on dest_reg
                 */
                apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_attention_weights);
                pack_tile(matmul_accum_reg, cb_attention_weights);
                tile_regs_release();
                cb_push_back(cb_attention_weights, onetile);

                // Step 2: apply statistics inplace: P = softmax(Z) = softmax(QK^T / sqrt(Et))
                apply_statistics_inplace(cb_attention_weights, cb_intermediates, num_of_interm_tiles);

                // Step 3: Accumulate grad_V = Attention^T @ grad_output
                update_grad_value(
                    cb_attention_weights,
                    cb_transpose_wh,
                    cb_grad_output,
                    alias_cb_prev_grad_value,
                    alias_cb_cur_grad_value,
                    tiles_per_row,
                    block_size,
                    /* do_accumulate */ h > 0 || head_idx > 0);

                cb_wait_front(alias_cb_cur_grad_value, tiles_per_row);

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
                // dK = scaler * (dZ^T @ Q)
                // we apply scaler inside compute_grad_scores function to improve numerical stablility of upcoming
                // matmul for grad K(and grad Q in q kernel)
                update_grad_key(
                    cb_grad_scores,
                    cb_query,
                    scaler_bits,
                    cb_transpose_wh,
                    alias_cb_prev_grad_key,
                    alias_cb_cur_grad_key,
                    tiles_per_row,
                    /* do_accumulate */ h > 0 || head_idx > 0);

                cb_wait_front(alias_cb_cur_grad_key, tiles_per_row);

                // Pop intermediate results used for computing dK and dV
                cb_pop_front(cb_u_scalar_row, onetile);
                cb_pop_front(cb_grad_attn_weights, onetile);
                cb_pop_front(cb_grad_scores, onetile);
                cb_pop_front(cb_attention_weights, onetile);
                cb_pop_front(cb_intermediates, num_of_interm_tiles);

                cb_pop_front(cb_query, tiles_per_row);
                cb_pop_front(cb_grad_output, tiles_per_row);
                cb_pop_front(cb_attn_output, tiles_per_row);

                std::swap(alias_cb_prev_grad_value, alias_cb_cur_grad_value);
                std::swap(alias_cb_prev_grad_key, alias_cb_cur_grad_key);
            }
        }

        pack_tiles_to_output(alias_cb_prev_grad_value, cb_grad_value, tiles_per_row);

        pack_tiles_to_output(alias_cb_prev_grad_key, cb_grad_key, tiles_per_row);

        cb_pop_front(cb_key, tiles_per_row);
        cb_pop_front(cb_value, tiles_per_row);
    }
}

}  // namespace NAMESPACE
