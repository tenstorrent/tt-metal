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

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t block_size = get_compile_time_arg_val(1);         // size of block
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t Ht = get_compile_time_arg_val(3);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);      // sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(5);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(6);  // used to transform mask from 0/-1 to 0/-1e9F

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;  // isn't used right now, for debugging only
constexpr uint32_t cb_temp_accum = tt::CBIndex::c_7;     // used for accumulating results

constexpr uint32_t cb_prev_max = tt::CBIndex::c_8;       // used to store previous max value
constexpr uint32_t cb_cur_max = tt::CBIndex::c_9;        // used to store current max value
constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_10;  // used for holding exp max diff during reduce
constexpr uint32_t cb_prev_sum_exp = tt::CBIndex::c_11;  // used for holding exp sum during reduce
constexpr uint32_t cb_cur_sum_exp = tt::CBIndex::c_12;   // used for holding exp sum during reduce
constexpr uint32_t cb_prev_mm_out = tt::CBIndex::c_13;   // used for holding previous matmul output
constexpr uint32_t cb_cur_mm_out = tt::CBIndex::c_14;    // used for holding current matmul output

constexpr uint32_t cb_output = tt::CBIndex::c_15;

constexpr uint32_t cb_mm_result_holder = tt::CBIndex::c_16;  // used for holding current matmul output

const uint32_t onetile = 1U;

const uint32_t k_chunk_size = Wt;
const uint32_t v_chunk_size = Wt;
const uint32_t kv_chunks_size = Wt;
const uint32_t kv_chunks_number = Ht;

// TODO: maybe I can move this file to compute_common.hpp file where I have other helper functions for sdpa_fw
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t cb_qk_result, uint32_t cb_identity_scaler>
void update_cur_row_max_value(uint32_t cb_cur_max_value, uint32_t cb_prev_max_value, bool do_eltwise_max = false) {
    cb_wait_front(cb_qk_result, onetile);
    cb_reserve_back(cb_cur_max_value, onetile);

    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1U;
    reduce_init<pool_type, reduce_dim>(cb_qk_result, cb_identity_scaler, cb_cur_max_value);
    tile_regs_acquire();
    reduce_tile<pool_type, reduce_dim>(
        cb_qk_result, cb_identity_scaler, /* tile_idx */ 0, /* tile_idx */ 0, /* dst_reg_idx */ reduce_dst_idx);
    reduce_uninit();

    if (do_eltwise_max) {
        DPRINT << "do eltwise max" << ENDL();
        cb_wait_front(cb_prev_max_value, onetile);
        copy_tile_init(cb_prev_max_value);
        copy_tile(cb_prev_max_value, /* tile_idx */ 0, /* register idx */ prev_max_dst_idx);

        // find max value between current max and previous max
        max_tile_init();
        max_tile(reduce_dst_idx, prev_max_dst_idx, static_cast<int>(VectorMode::C));
        // max_tile(reduce_dst_idx, prev_max_dst_idx);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_max_value);
    pack_tile(reduce_dst_idx, cb_cur_max_value);
    tile_regs_release();
    cb_push_back(cb_cur_max_value, onetile);
}

/* We process data by one tile, because we read only one row of K
 * Maybe we can read two rows of K and V and then process data by subblocks*/
template <uint32_t cb_qk_result, uint32_t cb_identity_scaler>
void apply_exp_inplace_and_find_exp_sum(uint32_t cb_cur_max_value, uint32_t cb_cur_exp_sum_holder) {
    cb_wait_front(cb_qk_result, onetile);
    cb_wait_front(cb_cur_max_value, onetile);

    const uint32_t exp_dst_idx = 0;
    sub_bcast_cols_init_short(cb_qk_result, cb_cur_max_value);
    tile_regs_acquire();
    sub_tiles_bcast_cols(
        cb_qk_result, cb_cur_max_value, /* tile_idx */ 0, /* tile_idx */ 0, /* dst_reg_idx */ exp_dst_idx);
    exp_tile<false>(exp_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    // update current qk matmul result with exp values
    cb_pop_front(cb_qk_result, onetile);
    cb_reserve_back(cb_qk_result, onetile);
    pack_reconfig_data_format(cb_qk_result);
    pack_tile(exp_dst_idx, cb_qk_result);

    /* update current exp sum with exp values
     * at the moment we pack one tile here
     * but we can use L1 accumlator to pack more tiles
     * in case we will be able to read more then one row of K and V
     */
    cb_reserve_back(cb_cur_exp_sum_holder, onetile);
    pack_reconfig_data_format(cb_cur_exp_sum_holder);
    pack_tile(exp_dst_idx, cb_cur_exp_sum_holder);
    tile_regs_release();

    cb_push_back(cb_qk_result, onetile);
    cb_push_back(cb_cur_exp_sum_holder, onetile);
}

void matmul_qk_by_v(uint32_t cb_qk_result, uint32_t cb_cur_mm_out_holder) {
    cb_wait_front(cb_qk_result, onetile);
    cb_wait_front(cb_value, kv_chunks_size);
    cb_reserve_back(cb_cur_mm_out_holder, Wt);

    // TODO[check]: check whether I can use mm_init_short here instead of full init
    // mm_init_short(cb_qk_result, cb_value, /* transpose */ 0);
    mm_init(cb_qk_result, cb_value, cb_cur_mm_out_holder, /* transpose */ 0);
    for (uint32_t tile_idx = 0; tile_idx < kv_chunks_size; tile_idx += block_size) {
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_qk_result,
                cb_value,
                /* tile_idx */ 0,
                /* tile_idx */ tile_idx + block_idx,
                block_idx,
                0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_cur_mm_out_holder);
        }
        tile_regs_release();
    }
    cb_push_back(cb_cur_mm_out_holder, Wt);
}

void update_exp_max_diff(uint32_t cb_prev_max_value, uint32_t cb_cur_max_value) {
    cb_wait_front(cb_prev_max_value, onetile);
    cb_wait_front(cb_cur_max_value, onetile);

    cb_reserve_back(cb_exp_max_diff, onetile);

    const uint32_t exp_max_diff_dst_idx = 0;
    reconfig_data_format(cb_prev_max_value, cb_cur_max_value);
    tile_regs_acquire();
    sub_tiles_init(cb_prev_max_value, cb_cur_max_value);
    sub_tiles(
        cb_prev_max_value,
        cb_cur_max_value,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* dst_reg_idx */ exp_max_diff_dst_idx);

    exp_tile_init<false>();
    exp_tile<false>(exp_max_diff_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_exp_max_diff);
    pack_tile(exp_max_diff_dst_idx, cb_exp_max_diff);
    tile_regs_release();
    cb_push_back(cb_exp_max_diff, onetile);
}

void update_cur_exp_sum_inplace(
    uint32_t cb_prev_sum_exp_holder, uint32_t cb_cur_sum_exp_holder, uint32_t cb_exp_max_diff_holder) {
    cb_wait_front(cb_prev_sum_exp_holder, onetile);
    cb_wait_front(cb_cur_sum_exp_holder, onetile);
    cb_wait_front(cb_exp_max_diff_holder, onetile);

    const uint32_t exp_sum_dst_idx = 0;
    mul_bcast_cols_init_short(cb_prev_sum_exp_holder, cb_exp_max_diff_holder);
    tile_regs_acquire();
    // multiply previous exp sum with exp_max_diff
    reconfig_data_format(cb_prev_sum_exp_holder, cb_exp_max_diff_holder);  // reconfig data format to precise
    mul_tiles_bcast_cols(cb_prev_sum_exp_holder, cb_exp_max_diff_holder, 0, 0, exp_sum_dst_idx);

    // copy current sum exp to next register
    copy_tile_init(cb_cur_sum_exp_holder);
    copy_tile(cb_cur_sum_exp_holder, /* tile_idx */ 0, /* register idx */ exp_sum_dst_idx + 1U);

    // add to updated previous exp sum with current exp sum
    add_binary_tile_init();
    add_binary_tile(exp_sum_dst_idx, exp_sum_dst_idx + 1U);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_cur_sum_exp_holder, onetile);
    cb_reserve_back(cb_cur_sum_exp_holder, onetile);
    pack_reconfig_data_format(cb_cur_sum_exp_holder);
    pack_tile(exp_sum_dst_idx, cb_cur_sum_exp_holder);
    tile_regs_release();
    cb_push_back(cb_cur_sum_exp_holder, onetile);
}

/*This uses L1 accumulation to accumulate onto cb_cur_mm_out_holder*/
// void update_cur_mm_out(uint32_t cb_prev_mm_out_holder, uint32_t cb_cur_mm_out_holder, uint32_t cb_exp_max_diff_holder) {
//     cb_wait_front(cb_prev_mm_out_holder, Wt);
//     cb_wait_front(cb_cur_mm_out_holder, Wt);
//     cb_wait_front(cb_exp_max_diff_holder, onetile);

//     PACK((llk_pack_reconfig_l1_acc(true)));  // enable L1 accumulation
//     pack_reconfig_data_format(cb_cur_mm_out_holder);

//     reconfig_data_format(cb_prev_mm_out_holder, cb_exp_max_diff_holder);
//     mul_bcast_cols_init_short(cb_prev_mm_out_holder, cb_exp_max_diff_holder);
//     for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
//         tile_regs_acquire();
//         for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
//             mul_tiles_bcast_cols(cb_prev_mm_out_holder, cb_exp_max_diff_holder, tile_idx + block_idx, 0, block_idx);
//         }
//         tile_regs_commit();
//         tile_regs_wait();
//         for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
//             pack_tile(/*dst_reg_idx*/ block_idx, cb_cur_mm_out_holder);
//         }
//         tile_regs_release();
//     }
//     PACK((llk_pack_reconfig_l1_acc(false)));  // disable L1 accumulation
//     cb_pop_front(cb_cur_mm_out_holder, Wt);
//     cb_reserve_back(cb_cur_mm_out_holder, Wt);
//     cb_push_back(cb_cur_mm_out_holder, Wt);
// }

void update_cur_mm_out(uint32_t cb_prev_mm_out_holder, uint32_t cb_cur_mm_out_holder, uint32_t cb_exp_max_diff_holder) {
    cb_wait_front(cb_prev_mm_out_holder, Wt);
    cb_wait_front(cb_cur_mm_out_holder, Wt);
    cb_wait_front(cb_exp_max_diff_holder, onetile);

    cb_reserve_back(cb_mm_result_holder, Wt);
    reconfig_data_format(cb_prev_mm_out_holder, cb_exp_max_diff_holder);
    for(uint32_t tile_idx = 0; tile_idx < Wt; tile_idx++) {
        tile_regs_acquire();
        mul_bcast_cols_init_short(cb_prev_mm_out_holder, cb_exp_max_diff_holder);
        mul_tiles_bcast_cols(cb_prev_mm_out_holder, cb_exp_max_diff_holder, tile_idx, 0, 0);

        copy_tile_init(cb_cur_mm_out_holder);
        copy_tile(cb_cur_mm_out_holder, /* tile_idx */ tile_idx, /* register idx */ 1U);
        
        add_binary_tile_init();
        add_binary_tile(0, 1U);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_mm_result_holder);
        pack_tile(0, cb_mm_result_holder);
        tile_regs_release();
    }
    cb_push_back(cb_mm_result_holder, Wt);

    cb_wait_front(cb_mm_result_holder, Wt);
    cb_pop_front(cb_cur_mm_out_holder, Wt);
    cb_reserve_back(cb_cur_mm_out_holder, Wt);
    for(uint32_t tile_idx = 0; tile_idx < Wt; tile_idx++) {
        tile_regs_acquire();
        copy_tile_init(cb_mm_result_holder);
        copy_tile(cb_mm_result_holder, /* tile_idx */ tile_idx, /* register idx */ 0);
        tile_regs_commit();
        
        tile_regs_wait();
        pack_reconfig_data_format(cb_cur_mm_out_holder);
        pack_tile(0, cb_cur_mm_out_holder);
        tile_regs_release();
    }
    cb_push_back(cb_cur_mm_out_holder, Wt);
    cb_pop_front(cb_mm_result_holder, Wt);
}

template <PoolType pool_type, ReduceDim reduce_dim, uint32_t cb_identity_scaler, bool fp32_transpose = false>
void reduce_and_recip_tile_inplace(uint32_t cb_in_idx) {
    cb_wait_front(cb_in_idx, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);  // generate tile for matmul row reduce)

    const uint32_t reduce_dst_idx = 0;

    reconfig_data_format(cb_in_idx, cb_matmul_reduce);  // reconfig data format to precise
    tile_regs_acquire();
    // reduce_init<pool_type, reduce_dim, false>(cb_in_idx, cb_identity_scaler, cb_in_idx);
    // reduce_tile<pool_type, reduce_dim, false>(
    //     cb_in_idx, cb_identity_scaler, /* tile_idx */ 0, /* tile_idx */ 0, /* dst_reg_idx */ reduce_dst_idx);
    // reduce_uninit();

    mm_init(cb_in_idx, cb_matmul_reduce, cb_identity_scaler, 0);
    // mm_init_short(cb_in_idx, cb_matmul_reduce, 0);
    matmul_tiles(
        cb_in_idx, cb_matmul_reduce, /* tile_idx */ 0, /* tile_idx */ 0, reduce_dst_idx, 0);
    
    // [Debug]: pack exp sum to cb_intermediates
    recip_tile_init();
    recip_tile(reduce_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_in_idx, onetile);
    cb_reserve_back(cb_in_idx, onetile);
    pack_reconfig_data_format(cb_in_idx);
    pack_tile(reduce_dst_idx, cb_in_idx);
    tile_regs_release();
    cb_push_back(cb_in_idx, onetile);
}

void pack_intermediate_result(uint32_t cb_in_idx, uint32_t cb_out_idx, uint32_t tiles_count = 1U) {
    cb_wait_front(cb_in_idx, tiles_count);
    cb_reserve_back(cb_out_idx, tiles_count);

    reconfig_data_format(cb_in_idx, cb_out_idx);

    for (uint32_t tile_idx = 0; tile_idx < tiles_count; ++tile_idx) {
        tile_regs_acquire();
        copy_tile_init(cb_in_idx);
        copy_tile(cb_in_idx, /* tile_idx */ tile_idx, /* register idx */ 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_out_idx);
        pack_tile(0, cb_out_idx);
        tile_regs_release();
    }

    // tile_regs_acquire();
    // copy_tile_init(cb_in_idx);
    // copy_tile(cb_in_idx, /* tile_idx */ 0, /* register idx */ 0);
    // tile_regs_commit();

    // tile_regs_wait();
    // pack_reconfig_data_format(cb_out_idx);
    // pack_tile(0, cb_out_idx);
    // tile_regs_release();
    cb_push_back(cb_out_idx, tiles_count);
}

void MAIN {
    init_sfpu(cb_query, cb_output);
    binary_op_init_common(cb_query, cb_key, cb_value);

    // [Debug]
    // mm_init(cb_query, cb_key, cb_temp_accum, 0);

    cb_wait_front(cb_reduction_scaler, onetile);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_query, Wt);


        // cb_wait_front(cb_attn_mask, Ht);  // wait until reader kernel has written kv_chunks_size tiles to cb_attn_mask

        // set up ping pong buffers
        // we will swap these buffer after each row processing to avoid overwriting previous results
        uint32_t alias_cb_prev_max = cb_prev_max;
        uint32_t alias_cb_cur_max = cb_cur_max;
        uint32_t alias_cb_prev_sum_exp = cb_prev_sum_exp;
        uint32_t alias_cb_cur_sum_exp = cb_cur_sum_exp;
        uint32_t alias_cb_prev_mm_out = cb_prev_mm_out;
        uint32_t alias_cb_cur_mm_out = cb_cur_mm_out;

        const uint32_t matmul_accum_reg = 0;
        const uint32_t mask_register = 1U;                 // mask register should be next to data register
        for (uint32_t h = 0; h < kv_chunks_number; ++h) {  // read all
            cb_wait_front(cb_key, kv_chunks_size);

            mm_init(cb_query, cb_key, cb_temp_accum, /* transpose */ 1);
            // TODO[check]: check whether I can use mm_init_short here instead of full init
            // mm_init_short(cb_query, cb_key, /* transpose */ 1);
            tile_regs_acquire();
            for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx++) {
                matmul_tiles(
                    cb_query,
                    cb_key,
                    /* tile_idx */ tile_idx,
                    /* tile_idx */ tile_idx,
                    /* dst_reg_idx*/ matmul_accum_reg,
                    /* transpose */ 1);  // accumulate in dest_reg 0
            }

            // TODO[improve]: maybe I need to move mask/scaler stuff to separate function

            // now we have to multiply result by scaler factor and then apply mask
            // we need to transform the attention mask for use in softmax:
            // The input `attn_mask` contains 1.0 for valid (keep) positions and 0.0 for masked (drop) positions.
            // To convert this into a format compatible with softmax masking:
            //   - Subtract 1.0 from the mask, so values become 0.0 (keep) and -1.0 (mask).
            //   - Multiply by a large negative value (e.g., 1e9F), resulting in 0.0 for valid entries and -inf for
            //   masked ones.
            // This way, after applying softmax, masked positions will effectively become zero,
            // and only the unmasked positions will retain meaningful attention weights
            cb_wait_front(cb_attn_mask, onetile);
            copy_tile_init(cb_attn_mask);
            // copy_tile(
            //     cb_attn_mask,
            //     /* tile_idx */ h,  // row of K define the column in (QK^T) matrix, so it define the column of
            //                        // attn_mask
            //     /* register idx */ mask_register);
            copy_tile(
                cb_attn_mask,
                /* tile_idx */ 0,
                /* register idx */ mask_register);

            // Apply the attention mask to Q @ K^T scores:
            // masked positions receive 0.0, unmasked positions remain unchanged
            mask_tile_init();
            mask_tile(matmul_accum_reg, mask_register);

            binop_with_scalar_tile_init();
            mul_unary_tile(matmul_accum_reg, scaler_bits);   // multiply by scaler factor
            add_unary_tile(mask_register, minus_one_bits);   // subtract 1.0 from mask, so it becomes 0.0 and -1.0
            mul_unary_tile(mask_register, custom_inf_bits);  // multiply by 1e9F to transform mask to 0.0 and
            // -1e9F

            // Add mask to scaled matmul result:
            // masked positions receive large negative values (will be 0.0 after softmax),
            // unmasked positions remain unchanged
            add_binary_tile_init();
            add_binary_tile(matmul_accum_reg, mask_register);
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_temp_accum, onetile);
            pack_reconfig_data_format(cb_temp_accum);
            pack_tile(matmul_accum_reg, cb_temp_accum);
            tile_regs_release();
            cb_push_back(cb_temp_accum, onetile);
            cb_pop_front(cb_attn_mask, onetile);

            // [Debug]: pack intermediate result to cb_intermediates
            // pack_intermediate_result(cb_temp_accum, cb_intermediates);

            // pop key data to make space for next key chunk
            cb_pop_front(cb_key, kv_chunks_size);

            /**
             * to find current max value we need to perform both reduce_max and eltwise max with previous result.
             * if do_eltwise_max:
             *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
             * else:
             *  cur_max = max(qk, dim=-1)
             */
            update_cur_row_max_value<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_temp_accum, cb_reduction_scaler>(
                alias_cb_cur_max, alias_cb_prev_max, /* if it first reduction in a row*/ h > 0);

            /* apply exp on qk_result inplace and */
            apply_exp_inplace_and_find_exp_sum<cb_temp_accum, cb_reduction_scaler>(
                alias_cb_cur_max, alias_cb_cur_sum_exp);
            
            // pack_intermediate_result(alias_cb_cur_sum_exp, cb_intermediates);

            /* wait on exp(qk_result) and multiply it by V row*/
            matmul_qk_by_v(cb_temp_accum, alias_cb_cur_mm_out);
            cb_pop_front(cb_temp_accum, onetile);  // pop exp(qk_result) to make space for next row
            cb_pop_front(cb_value, kv_chunks_size);

            // pack_intermediate_result(alias_cb_cur_mm_out, cb_intermediates, Wt);

            /* if we process not first row of K and V:
             * we need to update exp_max_diff = exp(cur_max_value - prev_max_value)
             * we need to update previous exp sum with exp_max_diff and add it to current exp sum
             * we need to update previous matmul output with exp_max_diff and add it to current matmul output
             */
            if (h > 0) {
                // update exp_max_diff = exp(cur_max_value - prev_max_value)
                update_exp_max_diff(alias_cb_prev_max, alias_cb_cur_max);
                cb_pop_front(alias_cb_prev_max, onetile);

                // update exp sum
                update_cur_exp_sum_inplace(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp, cb_exp_max_diff);
                cb_pop_front(alias_cb_prev_sum_exp, onetile);

                // update previous matmul output with exp_max_diff and add it to current matmul output
                update_cur_mm_out(alias_cb_prev_mm_out, alias_cb_cur_mm_out, cb_exp_max_diff);
                cb_pop_front(cb_exp_max_diff, onetile);
                cb_pop_front(alias_cb_prev_mm_out, Wt);
            }

            // swap buffers for next iteration
            std::swap(alias_cb_prev_max, alias_cb_cur_max);
            std::swap(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp);
            std::swap(alias_cb_prev_mm_out, alias_cb_cur_mm_out);
        }

        // update final output
        // pack_intermediate_result(alias_cb_prev_sum_exp, cb_intermediates);
        cb_wait_front(alias_cb_prev_mm_out, Wt);
        reduce_and_recip_tile_inplace<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_reduction_scaler, /* fp32_transpose */ true>(alias_cb_prev_sum_exp);
        cb_wait_front(alias_cb_prev_sum_exp, onetile);

        // [Debug]: pack exp sum to cb_intermediates
        pack_intermediate_result(alias_cb_prev_sum_exp, cb_intermediates);
        // pack_intermediate_result(alias_cb_prev_max, cb_intermediates);
        // cb_wait_front(alias_cb_prev_mm_out, Wt);

        cb_reserve_back(cb_output, Wt);
        pack_reconfig_data_format(cb_output);
        reconfig_data_format(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
        mul_bcast_cols_init_short(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
        for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_tiles_bcast_cols(alias_cb_prev_mm_out, alias_cb_prev_sum_exp, tile_idx + block_idx, 0, block_idx);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                pack_tile(block_idx, cb_output);
            }
            tile_regs_release();
        }
        cb_push_back(cb_output, Wt);

        // for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx++) {
        //     tile_regs_acquire();
        //     mul_tiles_bcast_cols(alias_cb_prev_mm_out, alias_cb_prev_sum_exp, tile_idx, 0, 0);
        //     tile_regs_commit();
        //     tile_regs_wait();
        //     pack_tile(0, cb_output);
        //     tile_regs_release();
        // }
        // cb_push_back(cb_output, Wt);

        // pack_intermediate_result(alias_cb_prev_sum_exp, cb_intermediates);
        cb_pop_front(alias_cb_prev_max, onetile);
        cb_pop_front(alias_cb_prev_sum_exp, onetile);      // pop previous exp
        cb_pop_front(alias_cb_prev_mm_out, Wt);  // pop previous matmul output to make space for next row

        // cb_pop_front(cb_attn_mask, Ht);  // pop attn_mask after processing all K and V rows
        cb_pop_front(cb_query, Wt);
    }
}

}  // namespace NAMESPACE
