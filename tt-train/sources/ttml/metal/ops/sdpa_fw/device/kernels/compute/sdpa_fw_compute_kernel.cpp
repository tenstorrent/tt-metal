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
#include "sdpa_compute_utils.hpp"
namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t block_size = get_compile_time_arg_val(1);         // size of block
constexpr uint32_t qWt = get_compile_time_arg_val(2);                // num tile in inner dim in query(d/TILE_W)
constexpr uint32_t kWt = get_compile_time_arg_val(3);               // num tile in inner dim in key and value (d/TILE_W)
constexpr uint32_t Ht = get_compile_time_arg_val(4);                // num_seq_len / TILE_H
constexpr uint32_t q_tiles_per_head = get_compile_time_arg_val(5);  // number of tiles per head in query
constexpr uint32_t q_heads = get_compile_time_arg_val(6);           // number of heads in query
constexpr uint32_t k_tiles_per_head = get_compile_time_arg_val(7);  // number of tiles per group in key and value
constexpr uint32_t heads_per_group = get_compile_time_arg_val(8);   // number of heads per group
constexpr uint32_t scaler_bits = get_compile_time_arg_val(9);       // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(10);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(11);  // used to transform mask from 0/-1 to 0/-1e9F

//[Debug]
constexpr uint32_t Wt = get_compile_time_arg_val(12);  // get old Wt

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;  // isn't used right now, for debugging only
constexpr uint32_t cb_qk_result = tt::CBIndex::c_7;      // used for accumulating results

constexpr uint32_t cb_prev_max = tt::CBIndex::c_8;       // used to store previous max value
constexpr uint32_t cb_cur_max = tt::CBIndex::c_9;        // used to store current max value
constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_10;  // used for holding exp max diff during reduce
constexpr uint32_t cb_prev_sum_exp = tt::CBIndex::c_11;  // used for holding exp sum during reduce
constexpr uint32_t cb_cur_sum_exp = tt::CBIndex::c_12;   // used for holding exp sum during reduce
constexpr uint32_t cb_prev_mm_out = tt::CBIndex::c_13;   // used for holding previous matmul output
constexpr uint32_t cb_cur_mm_out = tt::CBIndex::c_14;    // used for holding current matmul output

constexpr uint32_t cb_output = tt::CBIndex::c_15;

constexpr uint32_t cb_mm_result_holder = tt::CBIndex::c_16;  // used for holding current matmul output

constexpr uint32_t cb_test_temp_res = tt::CBIndex::c_17;  // used for debugging only

const uint32_t onetile = 1U;
const uint32_t tiles_per_head = q_tiles_per_head;  // assuming q_tiles_per_head == k_tiles_per_head

void MAIN {
    init_sfpu(cb_query, cb_output);
    binary_op_init_common(cb_query, cb_key, cb_value);

    // [Debug]
    // mm_init(cb_query, cb_key, cb_qk_result, 0);

    cb_wait_front(cb_reduction_scaler, onetile);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t q_head_idx = 0; q_head_idx < q_heads; ++q_head_idx) {
            cb_wait_front(cb_query, tiles_per_head);
            // set up ping pong buffers
            // we will swap these buffer after each row processing to avoid overwriting previous results
            uint32_t alias_cb_prev_max = cb_prev_max;
            uint32_t alias_cb_cur_max = cb_cur_max;
            uint32_t alias_cb_prev_sum_exp = cb_prev_sum_exp;
            uint32_t alias_cb_cur_sum_exp = cb_cur_sum_exp;
            uint32_t alias_cb_prev_mm_out = cb_prev_mm_out;
            uint32_t alias_cb_cur_mm_out = cb_cur_mm_out;

            const uint32_t matmul_accum_reg = 0;
            const uint32_t mask_register = 1U;   // mask register should be next to data register
            for (uint32_t h = 0; h < Ht; ++h) {  // read all
                cb_wait_front(cb_key, tiles_per_head);

                mm_init(cb_query, cb_key, cb_qk_result, /* transpose */ 1);
                // TODO[check]: check whether I can use mm_init_short here instead of full init
                // mm_init_short(cb_query, cb_key, /* transpose */ 1);
                tile_regs_acquire();
                for (uint32_t tile_idx = 0; tile_idx < tiles_per_head; tile_idx++) {
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
                apply_mask_on_reg<matmul_accum_reg>(cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
                tile_regs_commit();

                tile_regs_wait();
                cb_reserve_back(cb_qk_result, onetile);
                pack_reconfig_data_format(cb_qk_result);
                pack_tile(matmul_accum_reg, cb_qk_result);

                // [Debug]: pack mask to debug cb
                cb_reserve_back(cb_test_temp_res, onetile);
                pack_reconfig_data_format(cb_test_temp_res);
                pack_tile(matmul_accum_reg, cb_test_temp_res);  // for debugging only

                tile_regs_release();
                cb_push_back(cb_qk_result, onetile);
                cb_pop_front(cb_attn_mask, onetile);

                // [Debug] push mask to debug cb
                cb_push_back(cb_test_temp_res, onetile);  // for debugging only

                // pop key data to make space for next key chunk
                cb_pop_front(cb_key, tiles_per_head);

                /**
                 * to find current max value we need to perform both reduce_max and eltwise max with previous result.
                 * if do_eltwise_max:
                 *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                 * else:
                 *  cur_max = max(qk, dim=-1)
                 */
                update_cur_row_max_value<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_result, cb_reduction_scaler>(
                    alias_cb_cur_max, alias_cb_prev_max, /* if it first reduction in a row*/ h > 0);

                /* apply exp on qk_result inplace and */
                apply_exp_inplace_and_find_exp_sum<cb_qk_result>(alias_cb_cur_max, alias_cb_cur_sum_exp);

                /* wait on exp(qk_result) and multiply it by V row*/
                matmul_qk_by_v<tiles_per_head, block_size>(cb_qk_result, cb_value, alias_cb_cur_mm_out);
                cb_pop_front(cb_qk_result, onetile);  // pop exp(qk_result) to make space for next row
                cb_pop_front(cb_value, tiles_per_head);

                /* if we process not first row of K and V:
                 * we need to update exp_max_diff = exp(cur_max_value - prev_max_value)
                 * we need to update previous exp sum with exp_max_diff and add it to current exp sum
                 * we need to update previous matmul output with exp_max_diff and add it to current matmul output
                 */
                if (h > 0) {
                    // update exp_max_diff = exp(cur_max_value - prev_max_value)
                    update_exp_max_diff(alias_cb_prev_max, alias_cb_cur_max, cb_exp_max_diff);
                    cb_pop_front(alias_cb_prev_max, onetile);

                    // update exp sum
                    update_cur_exp_sum_inplace(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp, cb_exp_max_diff);
                    cb_pop_front(alias_cb_prev_sum_exp, onetile);

                    // update previous matmul output with exp_max_diff and add it to current matmul output
                    // update_cur_mm_out<tiles_per_head, block_size>(alias_cb_prev_mm_out, alias_cb_cur_mm_out,
                    // cb_exp_max_diff);
                    update_cur_mm_out<tiles_per_head>(
                        alias_cb_prev_mm_out, alias_cb_cur_mm_out, cb_exp_max_diff, cb_mm_result_holder);
                    cb_pop_front(cb_exp_max_diff, onetile);
                    cb_pop_front(alias_cb_prev_mm_out, tiles_per_head);
                }

                // swap buffers for next iteration
                std::swap(alias_cb_prev_max, alias_cb_cur_max);
                std::swap(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp);
                std::swap(alias_cb_prev_mm_out, alias_cb_cur_mm_out);

                // [Debug] pop mask from debug cb
                cb_pop_front(cb_test_temp_res, onetile);  // for debugging only
            }

            // update final output
            cb_wait_front(alias_cb_prev_mm_out, tiles_per_head);
            reduce_and_recip_tile_inplace<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                cb_reduction_scaler,
                cb_matmul_reduce,
                /* fp32_transpose */ true>(alias_cb_prev_sum_exp);
            cb_wait_front(alias_cb_prev_sum_exp, onetile);

            // [Debug]: pack exp sum to cb_intermediates

            if (q_head_idx == 0) {
                pack_intermediate_result(alias_cb_prev_sum_exp, cb_intermediates);
            }

            cb_reserve_back(cb_output, tiles_per_head);
            pack_reconfig_data_format(cb_output);
            reconfig_data_format(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
            mul_bcast_cols_init_short(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
            for (uint32_t tile_idx = 0; tile_idx < tiles_per_head; tile_idx += block_size) {
                tile_regs_acquire();
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                    mul_tiles_bcast_cols(
                        alias_cb_prev_mm_out, alias_cb_prev_sum_exp, tile_idx + block_idx, 0, block_idx);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                    pack_tile(block_idx, cb_output);
                }
                tile_regs_release();
            }
            cb_push_back(cb_output, tiles_per_head);

            cb_pop_front(alias_cb_prev_max, onetile);      // pop previous max to make space for next row
            cb_pop_front(alias_cb_prev_sum_exp, onetile);  // pop previous exp sum to make space for next row
            cb_pop_front(
                alias_cb_prev_mm_out, tiles_per_head);  // pop previous matmul output to make space for next row
            cb_pop_front(cb_query, tiles_per_head);
        }
    }
}

// void MAIN {
//     init_sfpu(cb_query, cb_output);
//     binary_op_init_common(cb_query, cb_key, cb_value);

//     // [Debug]
//     // mm_init(cb_query, cb_key, cb_qk_result, 0);

//     cb_wait_front(cb_reduction_scaler, onetile);
//     for (uint32_t row = 0; row < num_rows_per_core; ++row) {
//         cb_wait_front(cb_query, Wt);
//         // set up ping pong buffers
//         // we will swap these buffer after each row processing to avoid overwriting previous results
//         uint32_t alias_cb_prev_max = cb_prev_max;
//         uint32_t alias_cb_cur_max = cb_cur_max;
//         uint32_t alias_cb_prev_sum_exp = cb_prev_sum_exp;
//         uint32_t alias_cb_cur_sum_exp = cb_cur_sum_exp;
//         uint32_t alias_cb_prev_mm_out = cb_prev_mm_out;
//         uint32_t alias_cb_cur_mm_out = cb_cur_mm_out;

//         const uint32_t matmul_accum_reg = 0;
//         const uint32_t mask_register = 1U;   // mask register should be next to data register
//         for (uint32_t h = 0; h < Ht; ++h) {  // read all
//             cb_wait_front(cb_key, Wt);

//             mm_init(cb_query, cb_key, cb_qk_result, /* transpose */ 1);
//             // TODO[check]: check whether I can use mm_init_short here instead of full init
//             // mm_init_short(cb_query, cb_key, /* transpose */ 1);
//             tile_regs_acquire();
//             for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx++) {
//                 matmul_tiles(
//                     cb_query,
//                     cb_key,
//                     /* tile_idx */ tile_idx,
//                     /* tile_idx */ tile_idx,
//                     /* dst_reg_idx*/ matmul_accum_reg,
//                     /* transpose */ 1);  // accumulate in dest_reg 0
//             }

//             /*
//              * apply attention mask on dest_reg.
//              * function assumes that dest_reg is in acquired state via *acquire_dst* call
//              * function transforms mask from 1/0 to 0/-1e9F and applies it on dest_reg
//              */
//             apply_mask_on_reg<matmul_accum_reg>(cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
//             tile_regs_commit();

//             tile_regs_wait();
//             cb_reserve_back(cb_qk_result, onetile);
//             pack_reconfig_data_format(cb_qk_result);
//             pack_tile(matmul_accum_reg, cb_qk_result);
//             tile_regs_release();
//             cb_push_back(cb_qk_result, onetile);
//             cb_pop_front(cb_attn_mask, onetile);

//             // pop key data to make space for next key chunk
//             cb_pop_front(cb_key, Wt);

//             /**
//              * to find current max value we need to perform both reduce_max and eltwise max with previous result.
//              * if do_eltwise_max:
//              *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
//              * else:
//              *  cur_max = max(qk, dim=-1)
//              */
//             update_cur_row_max_value<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_result, cb_reduction_scaler>(
//                 alias_cb_cur_max, alias_cb_prev_max, /* if it first reduction in a row*/ h > 0);

//             /* apply exp on qk_result inplace and */
//             apply_exp_inplace_and_find_exp_sum<cb_qk_result>(alias_cb_cur_max, alias_cb_cur_sum_exp);

//             /* wait on exp(qk_result) and multiply it by V row*/
//             matmul_qk_by_v<Wt, block_size>(cb_qk_result, cb_value, alias_cb_cur_mm_out);
//             cb_pop_front(cb_qk_result, onetile);  // pop exp(qk_result) to make space for next row
//             cb_pop_front(cb_value, Wt);

//             /* if we process not first row of K and V:
//              * we need to update exp_max_diff = exp(cur_max_value - prev_max_value)
//              * we need to update previous exp sum with exp_max_diff and add it to current exp sum
//              * we need to update previous matmul output with exp_max_diff and add it to current matmul output
//              */
//             if (h > 0) {
//                 // update exp_max_diff = exp(cur_max_value - prev_max_value)
//                 update_exp_max_diff(alias_cb_prev_max, alias_cb_cur_max, cb_exp_max_diff);
//                 cb_pop_front(alias_cb_prev_max, onetile);

//                 // update exp sum
//                 update_cur_exp_sum_inplace(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp, cb_exp_max_diff);
//                 cb_pop_front(alias_cb_prev_sum_exp, onetile);

//                 // update previous matmul output with exp_max_diff and add it to current matmul output
//                 // update_cur_mm_out<Wt, block_size>(alias_cb_prev_mm_out, alias_cb_cur_mm_out, cb_exp_max_diff);
//                 update_cur_mm_out<Wt>(alias_cb_prev_mm_out, alias_cb_cur_mm_out, cb_exp_max_diff,
//                 cb_mm_result_holder); cb_pop_front(cb_exp_max_diff, onetile); cb_pop_front(alias_cb_prev_mm_out, Wt);
//             }

//             // swap buffers for next iteration
//             std::swap(alias_cb_prev_max, alias_cb_cur_max);
//             std::swap(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp);
//             std::swap(alias_cb_prev_mm_out, alias_cb_cur_mm_out);
//         }

//         // update final output
//         cb_wait_front(alias_cb_prev_mm_out, Wt);
//         reduce_and_recip_tile_inplace<
//             PoolType::MAX,
//             ReduceDim::REDUCE_ROW,
//             cb_reduction_scaler,
//             cb_matmul_reduce,
//             /* fp32_transpose */ true>(alias_cb_prev_sum_exp);
//         cb_wait_front(alias_cb_prev_sum_exp, onetile);

//         // [Debug]: pack exp sum to cb_intermediates
//         pack_intermediate_result(alias_cb_prev_sum_exp, cb_intermediates);

//         cb_reserve_back(cb_output, Wt);
//         pack_reconfig_data_format(cb_output);
//         reconfig_data_format(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
//         mul_bcast_cols_init_short(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
//         for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
//             tile_regs_acquire();
//             for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
//                 mul_tiles_bcast_cols(alias_cb_prev_mm_out, alias_cb_prev_sum_exp, tile_idx + block_idx, 0,
//                 block_idx);
//             }
//             tile_regs_commit();
//             tile_regs_wait();
//             for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
//                 pack_tile(block_idx, cb_output);
//             }
//             tile_regs_release();
//         }
//         cb_push_back(cb_output, Wt);

//         cb_pop_front(alias_cb_prev_max, onetile);      // pop previous max to make space for next row
//         cb_pop_front(alias_cb_prev_sum_exp, onetile);  // pop previous exp sum to make space for next row
//         cb_pop_front(alias_cb_prev_mm_out, Wt);        // pop previous matmul output to make space for next row
//         cb_pop_front(cb_query, Wt);
//     }
// }

}  // namespace NAMESPACE
