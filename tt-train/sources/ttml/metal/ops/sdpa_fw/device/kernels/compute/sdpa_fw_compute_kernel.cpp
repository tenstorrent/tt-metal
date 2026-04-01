// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <api/compute/cb_api.h>
#include <api/compute/pack.h>
#include <api/compute/reconfig_data_format.h>
#include <api/compute/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
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
#include "sdpa_compute_utils.hpp"

// For standard mode: num_rows_per_core = rows to process
// For balanced mode: num_rows_per_core = num_pairs (each pair = 2 rows)
constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);       // size of block
constexpr uint32_t qWt = get_compile_time_arg_val(2);              // num tile in inner dim in query(d/TILE_W)
constexpr uint32_t Ht = get_compile_time_arg_val(3);               // num_seq_len / TILE_H
constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(5);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(6);  // used to transform mask from 0/-1 to 0/-1e9F
constexpr uint32_t pairs_per_seq = Ht / 2;

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;  // isn't used right now, for debugging only
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_7;

constexpr uint32_t cb_prev_max = tt::CBIndex::c_8;       // used to store previous max value
constexpr uint32_t cb_cur_max = tt::CBIndex::c_9;        // used to store current max value
constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_10;  // used for holding exp max diff during reduce
constexpr uint32_t cb_prev_sum_exp = tt::CBIndex::c_11;  // used for holding exp sum during reduce
constexpr uint32_t cb_cur_sum_exp = tt::CBIndex::c_12;   // used for holding exp sum during reduce
constexpr uint32_t cb_prev_mm_out = tt::CBIndex::c_13;   // used for holding previous matmul output
constexpr uint32_t cb_cur_mm_out = tt::CBIndex::c_14;    // used for holding current matmul output

constexpr uint32_t cb_output = tt::CBIndex::c_15;

/**
 * Process a single row of the SDPA computation.
 * This function handles the full attention computation for one query row:
 * - Q @ K^T with masking and scaling
 * - Softmax with online normalization
 * - Attention @ V
 *
 * @param global_row_idx The global row index (across all batches/heads/sequences)
 */
FORCE_INLINE void process_single_row(uint32_t global_row_idx) {
    cb_wait_front(cb_query, qWt);

    // Calculate position within sequence for causal mask
    const uint32_t q_row_tile = global_row_idx % Ht;  // position within sequence (0 to Ht-1)

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // For causal mask / balanced: only process K/V tiles up to and including the diagonal
    const uint32_t num_kv_tiles_to_process = q_row_tile + 1;
#else
    // For non-causal: process all K/V tiles
    const uint32_t num_kv_tiles_to_process = Ht;
#endif

    // set up ping pong buffers
    uint32_t alias_cb_prev_max = cb_prev_max;
    uint32_t alias_cb_cur_max = cb_cur_max;
    uint32_t alias_cb_prev_sum_exp = cb_prev_sum_exp;
    uint32_t alias_cb_cur_sum_exp = cb_cur_sum_exp;
    uint32_t alias_cb_prev_mm_out = cb_prev_mm_out;
    uint32_t alias_cb_cur_mm_out = cb_cur_mm_out;

    const uint32_t matmul_accum_reg = 0;
    for (uint32_t h = 0; h < num_kv_tiles_to_process; ++h) {
        cb_wait_front(cb_key, qWt);

        reconfig_data_format(cb_query, cb_key);
        mm_init_short(cb_query, cb_key, /* transpose */ 1);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < qWt; tile_idx++) {
            matmul_tiles(
                cb_query,
                cb_key,
                /* tile_idx */ tile_idx,
                /* tile_idx */ tile_idx,
                /* dst_reg_idx*/ matmul_accum_reg);
        }

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
        // For causal mask: apply triangular mask on diagonal tile (h == q_row_tile)
        // Scaling is deferred to after max-subtraction for better precision
        if (h == q_row_tile) {
            apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, minus_one_bits, custom_inf_bits);
        }
        // Non-diagonal tiles: no mask and no scaling needed here
#elif defined(USE_ATTN_MASK)
        apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, minus_one_bits, custom_inf_bits);
#endif
        // NO MASK / non-diagonal: scores pass through unscaled.
        // Scale is applied in apply_exp_inplace_and_find_exp_sum after max subtraction.
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_attention_weights, onetile);
        pack_reconfig_data_format(cb_attention_weights);
        pack_tile(matmul_accum_reg, cb_attention_weights);
        tile_regs_release();
        cb_push_back(cb_attention_weights, onetile);
#ifdef USE_ATTN_MASK
        // For USE_ATTN_MASK: each mask tile is unique, pop after use
        cb_pop_front(cb_attn_mask, onetile);
#endif
        // Note: For CAUSAL_MASK/BALANCED_PARALLELISM, we reuse the same mask tile - don't pop it here

        // pop key data to make space for next key chunk
        cb_pop_front(cb_key, qWt);

        /**
         * to find current max value we need to perform both reduce_max and eltwise max with previous result.
         * if do_eltwise_max:
         *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
         * else:
         *  cur_max = max(qk, dim=-1)
         */
        update_cur_row_max_value<PoolType::MAX, ReduceDim::REDUCE_ROW>(
            cb_attention_weights,
            cb_reduction_scaler,
            alias_cb_cur_max,
            alias_cb_prev_max,
            /* if it first reduction in a row*/ h > 0);

        apply_exp_inplace_and_find_exp_sum(cb_attention_weights, alias_cb_cur_max, alias_cb_cur_sum_exp, scaler_bits);

        matmul_qk_by_v(qWt, block_size, cb_attention_weights, cb_value, alias_cb_cur_mm_out);
        cb_pop_front(cb_attention_weights, onetile);
        cb_pop_front(cb_value, qWt);

        /* if we process not first row of K and V:
         * we need to update exp_max_diff = exp(cur_max_value - prev_max_value)
         * we need to update previous exp sum with exp_max_diff and add it to current exp sum
         * we need to update previous matmul output with exp_max_diff and add it to current matmul output
         */
        if (h > 0) {
            update_exp_max_diff(alias_cb_prev_max, alias_cb_cur_max, cb_exp_max_diff, scaler_bits);
            cb_pop_front(alias_cb_prev_max, onetile);

            update_cur_exp_sum_inplace(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp, cb_exp_max_diff);
            cb_pop_front(alias_cb_prev_sum_exp, onetile);

            update_cur_mm_out(qWt, block_size, alias_cb_prev_mm_out, alias_cb_cur_mm_out, cb_exp_max_diff);

            cb_pop_front(cb_exp_max_diff, onetile);
            cb_pop_front(alias_cb_prev_mm_out, qWt);
        }

        std::swap(alias_cb_prev_max, alias_cb_cur_max);
        std::swap(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp);
        std::swap(alias_cb_prev_mm_out, alias_cb_cur_mm_out);
    }

    // Finalize output
    cb_wait_front(alias_cb_prev_mm_out, qWt);

    row_reduce_tile_inplace<cb_reduction_scaler, cb_matmul_reduce>(alias_cb_prev_sum_exp);
    cb_wait_front(alias_cb_prev_sum_exp, onetile);

#ifdef RETURN_INTERMEDIATES
    // Compute lse = scale*max + log(sum_exp) in FP32 DST, pack to FP32 intermediates CB.
    // Max is unscaled; scale is applied here to produce correct LSE for backward pass.
    compute_and_pack_lse(alias_cb_prev_sum_exp, alias_cb_prev_max, cb_intermediates, cb_matmul_reduce, scaler_bits);
#endif

    // recip(sum_exp) still needed for output normalization: O = mm_out * (1/sum_exp)
    recip_tile_inplace(alias_cb_prev_sum_exp);
    cb_wait_front(alias_cb_prev_sum_exp, onetile);

    cb_reserve_back(cb_output, qWt);
    pack_reconfig_data_format(cb_output);
    for (uint32_t tile_idx = 0; tile_idx < qWt; tile_idx += block_size) {
        tile_regs_acquire();

        // Load mm_out tiles via UnpackToDestFp32 (full FP32 in DST)
        reconfig_data_format(alias_cb_prev_mm_out, alias_cb_prev_mm_out);
        copy_tile_init(alias_cb_prev_mm_out);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(alias_cb_prev_mm_out, tile_idx + block_idx, block_idx);
        }

        // Load 1/sum_exp with column broadcast to DST[block_size]
        init_unary_bcast_col(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
        unary_bcast<BroadcastType::COL>(alias_cb_prev_sum_exp, 0, block_size);

        // SFPU multiply: DST[i] = mm_out[i] * (1/sum_exp)
        mul_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_binary_tile(block_idx, block_size, block_idx);
        }

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_output);
        }
        tile_regs_release();
    }
    cb_push_back(cb_output, qWt);

    cb_pop_front(alias_cb_prev_max, onetile);
    cb_pop_front(alias_cb_prev_sum_exp, onetile);
    cb_pop_front(alias_cb_prev_mm_out, qWt);
    cb_pop_front(cb_query, qWt);
}

void kernel_main() {
    // Runtime args
    // For standard mode: arg0 = start_row
    // For balanced mode: arg0 = start_pair_idx, arg1 = num_pairs
    const uint32_t start_idx = get_arg_val<uint32_t>(0);

#ifdef BALANCED_PARALLELISM
    const uint32_t num_pairs = get_arg_val<uint32_t>(1);  // Runtime arg for balanced mode
#endif

    init_sfpu(cb_query, cb_output);
    binary_op_init_common(cb_query, cb_key, cb_value);
    mm_init(cb_query, cb_key, cb_attention_weights);

    cb_wait_front(cb_reduction_scaler, onetile);

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Wait for causal mask tile ONCE - it's generated by writer and will be reused for every diagonal
    cb_wait_front(cb_attn_mask, onetile);
#endif

#ifdef BALANCED_PARALLELISM
    // Balanced parallelism mode: process pairs of rows (light + heavy)
    // Each pair consists of:
    //   - light_row: early in sequence (less work: row_in_seq + 1 K/V tiles)
    //   - heavy_row: late in sequence (more work: Ht - row_in_seq K/V tiles)
    // Together they have constant work = Ht + 1 tiles

    for (uint32_t p = 0; p < num_pairs; ++p) {
        const uint32_t global_pair_idx = start_idx + p;

        // Map pair index to sequence and position within sequence
        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        // Calculate the two row indices for this pair
        // light_row: ascending from start of sequence (pair 0 -> row 0, pair 1 -> row 1, ...)
        // heavy_row: descending from end of sequence (pair 0 -> row Ht-1, pair 1 -> row Ht-2, ...)
        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

        // Process light row first (less work)
        process_single_row(light_global_row);

        // Process heavy row second (more work)
        process_single_row(heavy_global_row);
    }
#else
    // Standard mode: process rows sequentially
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
