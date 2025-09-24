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
constexpr uint32_t block_size = get_compile_time_arg_val(1);         // size of block
constexpr uint32_t qWt = get_compile_time_arg_val(2);                // num tile in inner dim in query(d/TILE_W)
constexpr uint32_t kWt = get_compile_time_arg_val(3);                // num tile in inner dim in key and value (d/TILE_W)
constexpr uint32_t Ht = get_compile_time_arg_val(4);                 // num_seq_len / TILE_H
constexpr uint32_t q_heads = get_compile_time_arg_val(5);            // number of heads in query
constexpr uint32_t heads_per_group = get_compile_time_arg_val(6);    // number of heads per group
constexpr uint32_t scaler_bits = get_compile_time_arg_val(7);        // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(8);     // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(9);    // used to transform mask from 0/-1 to 0/-1e9F

// Circular buffer indices
constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;       // Gradient w.r.t. output
constexpr uint32_t cb_query = tt::CBIndex::c_1;            // Original query
constexpr uint32_t cb_key = tt::CBIndex::c_2;              // Original key
constexpr uint32_t cb_value = tt::CBIndex::c_3;            // Original value
constexpr uint32_t cb_mask = tt::CBIndex::c_4;             // Original mask
constexpr uint32_t cb_intermediates = tt::CBIndex::c_5;     // Forward pass intermediates

// Intermediate computation buffers
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_6;  // Recomputed attention weights
constexpr uint32_t cb_grad_attention = tt::CBIndex::c_7;     // Gradient w.r.t. attention
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_8;        // Gradient w.r.t. QK scores

// Output buffers  
constexpr uint32_t cb_grad_query = tt::CBIndex::c_9;        // Output: grad_Q
constexpr uint32_t cb_grad_key = tt::CBIndex::c_10;         // Output: grad_K
constexpr uint32_t cb_grad_value = tt::CBIndex::c_11;       // Output: grad_V

// Temporary/utility buffers
constexpr uint32_t cb_temp = tt::CBIndex::c_12;             // Temporary computations
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_13; // Reduction scaler

const uint32_t onetile = 1U;

void MAIN {
    init_sfpu(cb_grad_output, cb_grad_query);
    binary_op_init_common(cb_grad_output, cb_query, cb_key);

    cb_wait_front(cb_reduction_scaler, onetile);
    
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Read inputs for this row
        cb_wait_front(cb_grad_output, qWt);
        cb_wait_front(cb_query, qWt);
        cb_wait_front(cb_key, kWt);  
        cb_wait_front(cb_value, kWt);
        cb_wait_front(cb_intermediates, onetile);

        for (uint32_t h = 0; h < Ht; ++h) {
            // TODO: Implement the full backward pass computation
            // This is a skeleton - needs proper implementation of:
            
            // Step 1: Recompute attention weights from forward pass
            // attention_weights = softmax(scale * Q @ K^T + mask)
            
            // Step 2: Compute grad_V = Attention^T @ grad_output
            compute_grad_value<qWt, block_size>(
                cb_attention_weights, 
                cb_grad_output, 
                cb_grad_value);

            // Step 3: Compute grad_attention = grad_output @ V^T
            compute_grad_attention<qWt, block_size>(
                cb_grad_output,
                cb_value,
                cb_grad_attention);

            // Step 4: Compute gradient through softmax
            compute_grad_softmax(
                cb_attention_weights,
                cb_grad_attention, 
                cb_intermediates,
                cb_grad_scores);

            // Step 5: Compute grad_Q = grad_scores @ K
            compute_grad_query<kWt, block_size>(
                cb_grad_scores,
                cb_key,
                cb_grad_query);

            // Step 6: Compute grad_K = grad_scores^T @ Q
            compute_grad_key<qWt, block_size>(
                cb_grad_scores,
                cb_query,
                cb_grad_key);
        }

        // Pop front buffers for next iteration
        cb_pop_front(cb_grad_output, qWt);
        cb_pop_front(cb_query, qWt);
        cb_pop_front(cb_key, kWt);
        cb_pop_front(cb_value, kWt);
        cb_pop_front(cb_intermediates, onetile);
    }
}

}  // namespace NAMESPACE

