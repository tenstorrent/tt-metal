// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

// Compute gradient w.r.t. V: grad_V = Attention^T @ grad_output
template <uint32_t Wt, uint32_t block_size>
void compute_grad_value(
    uint32_t cb_attention_weights, 
    uint32_t cb_grad_output, 
    uint32_t cb_grad_value) {
    
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_grad_output, Wt);
    cb_reserve_back(cb_grad_value, Wt);

    // grad_V = Attention^T @ grad_output
    mm_init_short(cb_attention_weights, cb_grad_output, /* transpose */ 1);
    pack_reconfig_data_format(cb_grad_value);
    reconfig_data_format(cb_attention_weights, cb_grad_output);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_attention_weights,
                cb_grad_output,
                /* tile_idx */ 0,
                /* tile_idx */ tile_idx + block_idx,
                block_idx,
                0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_grad_value);
        }
        tile_regs_release();
    }
    cb_push_back(cb_grad_value, Wt);
}

// Compute gradient w.r.t. attention weights: grad_attention = grad_output @ V^T  
template <uint32_t Wt, uint32_t block_size>
void compute_grad_attention(
    uint32_t cb_grad_output,
    uint32_t cb_value,
    uint32_t cb_grad_attention) {
    
    cb_wait_front(cb_grad_output, Wt);
    cb_wait_front(cb_value, Wt);
    cb_reserve_back(cb_grad_attention, onetile);

    // grad_attention = grad_output @ V^T
    mm_init_short(cb_grad_output, cb_value, /* transpose */ 1);
    pack_reconfig_data_format(cb_grad_attention);
    reconfig_data_format(cb_grad_output, cb_value);

    tile_regs_acquire();
    for (uint32_t tile_idx = 0; tile_idx < Wt; ++tile_idx) {
        matmul_tiles(
            cb_grad_output,
            cb_value,
            /* tile_idx */ tile_idx,
            /* tile_idx */ tile_idx,
            0,
            tile_idx == 0 ? 0 : 1);  // accumulate
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_grad_attention);
    tile_regs_release();

    cb_push_back(cb_grad_attention, onetile);
}

// Compute gradient through softmax: grad_scores = attention .* (grad_attention - sum_over_cols(attention .* grad_attention))
template <uint32_t register_idx = 0>
void compute_grad_softmax(
    uint32_t cb_attention_weights,
    uint32_t cb_grad_attention,
    uint32_t cb_intermediates,  // 1/sum_exp from forward pass
    uint32_t cb_grad_scores) {
    
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_grad_attention, onetile);
    cb_wait_front(cb_intermediates, onetile);
    cb_reserve_back(cb_grad_scores, onetile);

    const uint32_t attention_reg = register_idx;
    const uint32_t grad_attention_reg = register_idx + 1U;
    const uint32_t intermediates_reg = register_idx + 2U;

    tile_regs_acquire();
    
    // Load tensors
    copy_tile_init(cb_attention_weights);
    copy_tile(cb_attention_weights, 0, attention_reg);
    
    copy_tile_init(cb_grad_attention);
    copy_tile(cb_grad_attention, 0, grad_attention_reg);
    
    copy_tile_init(cb_intermediates);
    copy_tile(cb_intermediates, 0, intermediates_reg);

    // Compute attention .* grad_attention
    mul_binary_tile_init();
    mul_binary_tile(attention_reg, grad_attention_reg, grad_attention_reg);

    // TODO: Implement full softmax backward computation
    // This is a simplified version - needs proper sum reduction and subtraction

    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_grad_scores);
    pack_tile(grad_attention_reg, cb_grad_scores);
    tile_regs_release();

    cb_push_back(cb_grad_scores, onetile);
}

// Compute gradient w.r.t. Q: grad_Q = grad_scores @ K
template <uint32_t Wt, uint32_t block_size>
void compute_grad_query(
    uint32_t cb_grad_scores,
    uint32_t cb_key,
    uint32_t cb_grad_query) {
    
    cb_wait_front(cb_grad_scores, onetile);
    cb_wait_front(cb_key, Wt);
    cb_reserve_back(cb_grad_query, Wt);

    // grad_Q = grad_scores @ K
    mm_init_short(cb_grad_scores, cb_key, /* transpose */ 0);
    pack_reconfig_data_format(cb_grad_query);
    reconfig_data_format(cb_grad_scores, cb_key);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_grad_scores,
                cb_key,
                /* tile_idx */ 0,
                /* tile_idx */ tile_idx + block_idx,
                block_idx,
                0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_grad_query);
        }
        tile_regs_release();
    }
    cb_push_back(cb_grad_query, Wt);
}

// Compute gradient w.r.t. K: grad_K = grad_scores^T @ Q  
template <uint32_t Wt, uint32_t block_size>
void compute_grad_key(
    uint32_t cb_grad_scores,
    uint32_t cb_query,
    uint32_t cb_grad_key) {
    
    cb_wait_front(cb_grad_scores, onetile);
    cb_wait_front(cb_query, Wt);
    cb_reserve_back(cb_grad_key, Wt);

    // grad_K = grad_scores^T @ Q
    mm_init_short(cb_grad_scores, cb_query, /* transpose */ 1);
    pack_reconfig_data_format(cb_grad_key);
    reconfig_data_format(cb_grad_scores, cb_query);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_grad_scores,
                cb_query,
                /* tile_idx */ 0,
                /* tile_idx */ tile_idx + block_idx,
                block_idx,
                0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_grad_key);
        }
        tile_regs_release();
    }
    cb_push_back(cb_grad_key, Wt);
}

