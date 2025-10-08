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
#include "debug/dprint.h"
#include "sdpa_bw_compute_utils.hpp"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);       // size of block
constexpr uint32_t qWt = get_compile_time_arg_val(2);              // num tile in inner dim in query(d/TILE_W)
constexpr uint32_t kWt = get_compile_time_arg_val(3);              // num tile in inner dim in key and value (d/TILE_W)
constexpr uint32_t Ht = get_compile_time_arg_val(4);               // num_seq_len / TILE_H
constexpr uint32_t q_heads = get_compile_time_arg_val(5);          // number of heads in query
constexpr uint32_t heads_per_group = get_compile_time_arg_val(6);  // number of heads per group
constexpr uint32_t scaler_bits = get_compile_time_arg_val(7);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(8);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(9);  // used to transform mask from 0/-1 to 0/-1e9F

// Circular buffer indices
constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;    // Gradient w.r.t. output
constexpr uint32_t cb_attn_output = tt::CBIndex::c_1;    // Attention output from forward pass
constexpr uint32_t cb_query = tt::CBIndex::c_2;          // Original query
constexpr uint32_t cb_key = tt::CBIndex::c_3;            // Original key
constexpr uint32_t cb_value = tt::CBIndex::c_4;          // Original value
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;      // Original mask
constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;  // Forward pass intermediates
// Temporary/utility buffers
constexpr uint32_t cb_mat_mul_reduction = tt::CBIndex::c_7;  // Temporary computations
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_8;   // Reduction scaler
constexpr uint32_t cb_mm_result_holder = tt::CBIndex::c_9;   // used for holding current matmul output

// Intermediate computation buffers
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_10;  // Recomputed attention weights = softmax(QK^T / sqrt(Et))
constexpr uint32_t cb_grad_attention = tt::CBIndex::c_11;     // Gradient w.r.t. attention
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_12;        // Gradient w.r.t. QK scores
constexpr uint32_t cb_transpose_wh = tt::CBIndex::c_13;       // Transpose of attention weights

// Output buffers
constexpr uint32_t cb_grad_query = tt::CBIndex::c_14;          // Output: grad_Q
constexpr uint32_t cb_grad_key = tt::CBIndex::c_15;            // Output: grad_K
constexpr uint32_t cb_grad_value = tt::CBIndex::c_16;          // Output: grad_V
constexpr uint32_t cb_sync_output_writer = tt::CBIndex::c_17;  // Used to sync with output writer kernel

const uint32_t onetile = 1U;

// in future optimization we can process data by chunks(for example 2 or 3 rows at once)
const uint32_t tiles_per_row = qWt;       // assuming qWt == kWt == vWt
const uint32_t num_of_interm_tiles = 2U;  // number of tiles in intermediates buffer per head

void MAIN {
    init_sfpu(cb_query, cb_key);
    binary_op_init_common(cb_grad_output, cb_query, cb_key);

    cb_wait_front(cb_reduction_scaler, onetile);
    cb_wait_front(cb_mat_mul_reduction, onetile);

    mm_init(cb_query, cb_key, cb_attention_weights);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_key, tiles_per_row);
        cb_wait_front(cb_value, tiles_per_row);

        for (uint32_t head_idx = 0; head_idx < heads_per_group; ++head_idx) {
            const uint32_t matmul_accum_reg = 0;
            const uint32_t mask_register = 1U;  // mask register should be next to data register

            for (uint32_t h = 0; h < Ht; ++h) {
                // Wait for Q, dO, O, mask and intermediates for this K/V row
                cb_wait_front(cb_query, tiles_per_row);
                cb_wait_front(cb_grad_output, tiles_per_row);
                cb_wait_front(cb_attn_output, tiles_per_row);

                // Step 1: Recompute attention weights(we will produce column of attention weights matrix while
                // streaming through Q, dO, O)
                reconfig_data_format(cb_query, cb_key);
                mm_init_short(cb_query, cb_key, /* transpose */ 1);
                tile_regs_acquire();
                for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
                    matmul_tiles(
                        cb_query,
                        cb_key,
                        /* tile_idx */ tile_idx,
                        /* tile_idx */ tile_idx,
                        /* dst_reg_idx*/ matmul_accum_reg,
                        /* transpose */ 1);  // accumulate in dest_reg 0
                }

                /*
                 * apply attention mask on dest_reg.
                 * function assumes that dest_reg is in acquired state via *acquire_dst* call
                 * function transforms mask from 1/0 to 0/-1e9F and applies it on dest_reg
                 */
                apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_attention_weights);
                pack_tile(matmul_accum_reg, cb_attention_weights);
                tile_regs_release();
                cb_push_back(cb_attention_weights, onetile);

                // apply statistics inplace: softmax(QK^T / sqrt(Et))
                cb_wait_front(cb_intermediates, num_of_interm_tiles);
                apply_statistics_inplace(cb_attention_weights, cb_intermediates);
                cb_wait_front(cb_attention_weights, onetile);

                // Step 2: Accumulate grad_V = Attention^T @ grad_output
                update_grad_value(
                    cb_attention_weights,
                    cb_transpose_wh,
                    cb_grad_output,
                    cb_mm_result_holder,
                    cb_grad_value,
                    tiles_per_row,
                    block_size,
                    h > 0);

                cb_pop_front(cb_query, tiles_per_row);
                cb_pop_front(cb_grad_output, tiles_per_row);
                cb_pop_front(cb_attn_output, tiles_per_row);
                cb_pop_front(cb_intermediates, num_of_interm_tiles);
                cb_pop_front(cb_attention_weights, onetile);
            }
        }

        // update pointer ins cb_sync_output_writer to signal output writer that one row is done
        cb_reserve_back(cb_sync_output_writer, onetile);
        cb_push_back(cb_sync_output_writer, onetile);  // signal output writer that one row is done

        cb_pop_front(cb_key, tiles_per_row);
        cb_pop_front(cb_value, tiles_per_row);
    }
}

}  // namespace NAMESPACE
