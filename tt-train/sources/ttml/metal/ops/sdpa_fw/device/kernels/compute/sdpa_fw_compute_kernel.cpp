// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/cb_api.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reconfig_data_format.h>
#include <compute_kernel_api/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/debug/dprint.h"
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
#include "sdpa_compute_utils.hpp"
namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t block_size = get_compile_time_arg_val(1);         // size of block
constexpr uint32_t qWt = get_compile_time_arg_val(2);                // num tile in inner dim in query(d/TILE_W)
constexpr uint32_t kWt = get_compile_time_arg_val(3);              // num tile in inner dim in key and value (d/TILE_W)
constexpr uint32_t Ht = get_compile_time_arg_val(4);               // num_seq_len / TILE_H
constexpr uint32_t q_heads = get_compile_time_arg_val(5);          // number of heads in query
constexpr uint32_t heads_per_group = get_compile_time_arg_val(6);  // number of heads per group
constexpr uint32_t scaler_bits = get_compile_time_arg_val(7);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(8);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(9);  // used to transform mask from 0/-1 to 0/-1e9F

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;  // isn't used right now, for debugging only
constexpr uint32_t cb_qk_result = tt::CBIndex::c_7;      // used for Q by K^t result

constexpr uint32_t cb_prev_max = tt::CBIndex::c_8;       // used to store previous max value
constexpr uint32_t cb_cur_max = tt::CBIndex::c_9;        // used to store current max value
constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_10;  // used for holding exp max diff during reduce
constexpr uint32_t cb_prev_sum_exp = tt::CBIndex::c_11;  // used for holding exp sum during reduce
constexpr uint32_t cb_cur_sum_exp = tt::CBIndex::c_12;   // used for holding exp sum during reduce
constexpr uint32_t cb_prev_mm_out = tt::CBIndex::c_13;   // used for holding previous matmul output
constexpr uint32_t cb_cur_mm_out = tt::CBIndex::c_14;    // used for holding current matmul output

constexpr uint32_t cb_output = tt::CBIndex::c_15;

const uint32_t onetile = 1U;

void MAIN {
    init_sfpu(cb_query, cb_output);
    binary_op_init_common(cb_query, cb_key, cb_value);
    mm_init(cb_query, cb_key, cb_qk_result);

    cb_wait_front(cb_reduction_scaler, onetile);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_query, qWt);
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

#ifdef USE_ATTN_MASK
            /*
             * apply attention mask on dest_reg.
             * function assumes that dest_reg is in acquired state via *acquire_dst* call
             * function transforms mask from 1/0 to 0/-1e9F and applies it on dest_reg
             */
            apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
#endif
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_qk_result, onetile);
            pack_reconfig_data_format(cb_qk_result);
            pack_tile(matmul_accum_reg, cb_qk_result);
            tile_regs_release();
            cb_push_back(cb_qk_result, onetile);
#ifdef USE_ATTN_MASK
            cb_pop_front(cb_attn_mask, onetile);
#endif

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
                cb_qk_result,
                cb_reduction_scaler,
                alias_cb_cur_max,
                alias_cb_prev_max,
                /* if it first reduction in a row*/ h > 0);

            /* apply exp on qk_result inplace and */
            apply_exp_inplace_and_find_exp_sum(cb_qk_result, alias_cb_cur_max, alias_cb_cur_sum_exp);

            /* wait on exp(qk_result) and multiply it by V row*/
            matmul_qk_by_v(qWt, block_size, cb_qk_result, cb_value, alias_cb_cur_mm_out);
            cb_pop_front(cb_qk_result, onetile);  // pop exp(qk_result) to make space for next row
            cb_pop_front(cb_value, qWt);

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
                update_cur_mm_out(qWt, block_size, alias_cb_prev_mm_out, alias_cb_cur_mm_out, cb_exp_max_diff);

                cb_pop_front(cb_exp_max_diff, onetile);
                cb_pop_front(alias_cb_prev_mm_out, qWt);
            }

            // swap buffers for next iteration
            std::swap(alias_cb_prev_max, alias_cb_cur_max);
            std::swap(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp);
            std::swap(alias_cb_prev_mm_out, alias_cb_cur_mm_out);
        }

        // update final output
        cb_wait_front(alias_cb_prev_mm_out, qWt);
        reduce_and_recip_tile_inplace<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_reduction_scaler, cb_matmul_reduce>(
            alias_cb_prev_sum_exp);
        cb_wait_front(alias_cb_prev_sum_exp, onetile);

#ifdef RETURN_INTERMEDIATES
        // Pack intermediates: max_val (col 0) and recip_sum_exp (col 32)
        // Total 2 tiles: [max_val_tile, recip_sum_exp_tile]
        // Writer will write these to shape (B, H, S, 64)
        // cb_matmul_reduce is used as mask tile (1.0 in col 0, 0.0 elsewhere) to ensure zeros elsewhere
        pack_intermediate_result(alias_cb_prev_max, cb_intermediates, cb_matmul_reduce);      // tile 0: max_val at col 0
        pack_intermediate_result(
            alias_cb_prev_sum_exp, cb_intermediates, cb_matmul_reduce);  // tile 1: recip_sum_exp at col 32
#endif

        cb_reserve_back(cb_output, qWt);
        pack_reconfig_data_format(cb_output);
        reconfig_data_format(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
        mul_bcast_cols_init_short(alias_cb_prev_mm_out, alias_cb_prev_sum_exp);
        for (uint32_t tile_idx = 0; tile_idx < qWt; tile_idx += block_size) {
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
        cb_push_back(cb_output, qWt);

        cb_pop_front(alias_cb_prev_max, onetile);      // pop previous max to make space for next row
        cb_pop_front(alias_cb_prev_sum_exp, onetile);  // pop previous exp sum to make space for next row
        cb_pop_front(alias_cb_prev_mm_out, qWt);       // pop previous matmul output to make space for next row
        cb_pop_front(cb_query, qWt);
    }
}

}  // namespace NAMESPACE
