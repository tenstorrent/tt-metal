// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "api/compute/transpose.h"
#include "sdpa_compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/sdpa_fw_work_list.hpp"

// Work item: one Q chunk of a head stack (see sdpa_fw_work_list.hpp), rows = Sq_chunk_t * heads_per_item
// tile-rows. The item walks K/V in steps of Sk_chunk_t tiles; per step it does QK^T, online softmax and
// P@V for all its rows.

constexpr uint32_t block_size = get_compile_time_arg_val(0);        // output tiles per DST pass, element-wise loops
constexpr uint32_t qWt = get_compile_time_arg_val(1);               // num tile in inner dim in query/key (d_qk/TILE_W)
constexpr uint32_t vWt = get_compile_time_arg_val(2);               // num tile in inner dim in value (d_v/TILE_W)
constexpr uint32_t Ht = get_compile_time_arg_val(3);                // num_seq_len / TILE_H
constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);       // 1/sqrt(d) softmax scale, fp32 bits
constexpr uint32_t scaler_bf16_bits = get_compile_time_arg_val(5);  // host-side RNE BF16 of it, WH fused scale+exp
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(6);    // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(7);   // used to transform mask from 0/-1 to 0/-1e9F
constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);        // query tile-rows per Q chunk
constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(9);        // K/V tiles per step
constexpr uint32_t qk_block_h = get_compile_time_arg_val(10);       // Q rows per QK^T matmul_block call
constexpr uint32_t qk_block_w = get_compile_time_arg_val(11);       // K tiles per QK^T matmul_block call
constexpr uint32_t pv_block_h = get_compile_time_arg_val(12);       // Q rows per PV matmul_block call
constexpr uint32_t pv_block_w = get_compile_time_arg_val(13);       // V feature tiles per PV matmul_block call
constexpr uint32_t sm_block_size = get_compile_time_arg_val(14);    // score tiles per DST pass, exp step
constexpr uint32_t heads_per_item = get_compile_time_arg_val(15);   // Q heads stacked in one item

constexpr uint32_t q_num_chunks = Ht / Sq_chunk_t;
constexpr uint32_t rows = Sq_chunk_t * heads_per_item;    // tile-rows of one item, stacked head after head
constexpr uint32_t q_tiles = rows * qWt;                  // Q tiles of one item
constexpr uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;  // arbitrary mask tiles of one step, shared by the heads
constexpr uint32_t score_tiles = rows * Sk_chunk_t;       // QK^T tiles of one K/V step
constexpr uint32_t out_tiles = rows * vWt;                // output / accumulator tiles of one item
constexpr uint32_t k_tiles = Sk_chunk_t * qWt;            // K tiles of one step
constexpr uint32_t v_tiles = Sk_chunk_t * vWt;            // V tiles of one step

static_assert(rows % qk_block_h == 0U && Sk_chunk_t % qk_block_w == 0U, "QK^T block must tile the item rows");

// Per-row state passes (max, alpha, sums, finalize) batch up to a DST's worth of rows into one pass.
constexpr uint32_t kRowsPerPass = sm_block_size;
constexpr uint32_t rows_in_pass(uint32_t row0, uint32_t per_pass) {
    return (rows - row0 < per_pass) ? (rows - row0) : per_pass;
}
static_assert(rows % pv_block_h == 0U, "PV block must tile the item rows");

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;  // ones-column tile: row reduce of the sum, lse mask
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_7;

constexpr uint32_t cb_prev_max = tt::CBIndex::c_8;       // used to store previous max value
constexpr uint32_t cb_cur_max = tt::CBIndex::c_9;        // used to store current max value
constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_10;  // used for holding exp max diff during reduce
constexpr uint32_t cb_prev_sum_exp = tt::CBIndex::c_11;  // used for holding exp sum during reduce
constexpr uint32_t cb_cur_sum_exp = tt::CBIndex::c_12;   // used for holding exp sum during reduce
constexpr uint32_t cb_prev_mm_out = tt::CBIndex::c_13;   // used for holding previous matmul output
constexpr uint32_t cb_cur_mm_out = tt::CBIndex::c_14;    // used for holding current matmul output

constexpr uint32_t cb_output = tt::CBIndex::c_15;
constexpr uint32_t cb_gate = tt::CBIndex::c_16;

// Finalize a chunk from its running state: out = (O / sum) [* sigmoid(gate)], and lse for the backward
// pass. Consumes (pops) the three state CBs and the gate.
FORCE_INLINE void finalize_chunk(uint32_t cb_mm_out, uint32_t cb_max, uint32_t cb_sum_exp) {
    cb_wait_front(cb_mm_out, out_tiles);
    cb_wait_front(cb_max, rows);
    cb_wait_front(cb_sum_exp, rows);
    cb_wait_front(cb_matmul_reduce, onetile);

    // The row sum so far is per 32-column tile column; reduce it to one value per row (in place).
    cb_pop_front(cb_sum_exp, rows);
    cb_reserve_back(cb_sum_exp, rows);
    for (uint32_t r0 = 0; r0 < rows; r0 += kRowsPerPass) {
        row_reduce_tiles_inplace<cb_reduction_scaler, cb_matmul_reduce>(cb_sum_exp, r0, rows_in_pass(r0, kRowsPerPass));
    }
    cb_push_back(cb_sum_exp, rows);
    cb_wait_front(cb_sum_exp, rows);

#ifdef RETURN_INTERMEDIATES
    // lse = scale*max + log(sum_exp)
    cb_reserve_back(cb_intermediates, rows);
    for (uint32_t r = 0; r < rows; ++r) {
        compute_and_pack_lse(cb_sum_exp, cb_max, cb_intermediates, cb_matmul_reduce, scaler_bits, r);
    }
    cb_push_back(cb_intermediates, rows);
#endif

    // 1/sum_exp for normalization
    cb_pop_front(cb_sum_exp, rows);
    cb_reserve_back(cb_sum_exp, rows);
    for (uint32_t r0 = 0; r0 < rows; r0 += kRowsPerPass) {
        recip_tiles_inplace(cb_sum_exp, r0, rows_in_pass(r0, kRowsPerPass));
    }
    cb_push_back(cb_sum_exp, rows);
    cb_wait_front(cb_sum_exp, rows);

    cb_reserve_back(cb_output, out_tiles);
#ifdef HAS_GATE
    cb_wait_front(cb_gate, out_tiles);
#endif
    pack_reconfig_data_format(cb_output);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t tile_idx = 0; tile_idx < vWt; tile_idx += block_size) {
            tile_regs_acquire();

            // Load mm_out tiles via UnpackToDestFp32 (full FP32 in DST)
            reconfig_data_format(cb_mm_out, cb_mm_out);
            copy_init(cb_mm_out);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                copy_tile(cb_mm_out, r * vWt + tile_idx + block_idx, block_idx);
            }

            // Load this row's 1/sum_exp with column broadcast to DST[block_size]
            init_unary_bcast_col(cb_sum_exp);
            unary_bcast<BroadcastType::COL>(cb_sum_exp, r, block_size);

            // SFPU multiply: DST[i] = mm_out[i] * (1/sum_exp)
            mul_binary_tile_init();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_binary_tile(block_idx, block_size, block_idx);
            }

#ifdef HAS_GATE
            reconfig_data_format(cb_gate, cb_gate);
            copy_init(cb_gate);
            sigmoid_tile_init();
            mul_binary_tile_init();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                copy_tile(cb_gate, r * vWt + tile_idx + block_idx, block_size);
                sigmoid_tile(block_size);
                mul_binary_tile(block_idx, block_size, block_idx);
            }
#endif

            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                pack_tile(block_idx, cb_output);
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_output, out_tiles);

    cb_pop_front(cb_max, rows);
    cb_pop_front(cb_sum_exp, rows);
    cb_pop_front(cb_mm_out, out_tiles);
#ifdef HAS_GATE
    cb_pop_front(cb_gate, out_tiles);
#endif
}

// Layouts used below. Score CB (cb_attention_weights): the rows x Sk_chunk_t score tiles of one
// step, row-major: row r, K tile n sits at r * Sk_chunk_t + n. Per-row state CBs (max, sum,
// exp_max_diff): row r sits in slot r. Output accumulators: row r starts at r * vWt.
// Runs K/V steps [k_begin, k_end) of the chunk; the softmax state starts fresh at k_begin.
FORCE_INLINE void process_q_chunk(uint32_t item, uint32_t k_begin, uint32_t k_end) {
    cb_wait_front(cb_query, q_tiles);

    const uint32_t first_row_tile = (item % q_num_chunks) * Sq_chunk_t;
#ifdef CAUSAL_MASK
    // Steps before first_masked_step lie entirely left of the diagonal and need no masking.
    const uint32_t first_masked_step = first_row_tile / Sk_chunk_t;
#endif

    // prev/cur state CBs swap roles after every step
    uint32_t alias_cb_prev_max = cb_prev_max;
    uint32_t alias_cb_cur_max = cb_cur_max;
    uint32_t alias_cb_prev_sum_exp = cb_prev_sum_exp;
    uint32_t alias_cb_cur_sum_exp = cb_cur_sum_exp;
    uint32_t alias_cb_prev_mm_out = cb_prev_mm_out;
    uint32_t alias_cb_cur_mm_out = cb_cur_mm_out;

    for (uint32_t k_step = k_begin; k_step < k_end; ++k_step) {
        const bool first_step = (k_step == k_begin);
        cb_wait_front(cb_key, k_tiles);

        reconfig_data_format(cb_query, cb_key);
        pack_reconfig_data_format(cb_attention_weights);

        // S = Q @ K^T, one qk_block_h x qk_block_w rectangle of score tiles per DST pass, packed to its
        // row-major slot. A mask is added afterwards by the packer (see below).
        matmul_block_init(
            cb_query,
            cb_key,
            /* transpose */ 1,
            /* ct_dim */ qk_block_w,
            /* rt_dim */ qk_block_h,
            /* kt_dim */ qWt);
        cb_reserve_back(cb_attention_weights, score_tiles);
        for (uint32_t row0 = 0; row0 < rows; row0 += qk_block_h) {
            for (uint32_t col0 = 0; col0 < Sk_chunk_t; col0 += qk_block_w) {
                tile_regs_acquire();
                // Accumulate over the feature dim: Q is row-major (next feature tile = +1), K is stored
                // feature-major in cb_key (next feature tile = +Sk_chunk_t).
                uint32_t q_idx = row0 * qWt;
                uint32_t k_idx = col0;
                for (uint32_t feat = 0; feat < qWt; ++feat) {
                    matmul_block(
                        cb_query,
                        cb_key,
                        /* in0 (Q) tile_idx */ q_idx,
                        /* in1 (K) tile_idx */ k_idx,
                        /* dst_idx */ 0,
                        /* transpose */ 1,
                        /* ct_dim */ qk_block_w,
                        /* rt_dim */ qk_block_h,
                        /* kt_dim */ qWt);
                    q_idx += 1U;
                    k_idx += Sk_chunk_t;
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < qk_block_h; ++i) {
                    for (uint32_t j = 0; j < qk_block_w; ++j) {
                        pack_tile</* out_of_order */ true>(
                            i * qk_block_w + j, cb_attention_weights, /* offset */ (row0 + i) * Sk_chunk_t + col0 + j);
                    }
                }
                tile_regs_release();
            }
        }
        cb_push_back(cb_attention_weights, score_tiles);

#ifdef CAUSAL_MASK
        if (k_step >= first_masked_step) {
            // Add the causal mask onto the scores already in L1. The CB holds exactly one step, so
            // pop + reserve lands on the same slots; with L1-accumulate on, pack_tile adds to the
            // tile already there instead of overwriting it.
            cb_wait_front(cb_attention_weights, score_tiles);
            cb_pop_front(cb_attention_weights, score_tiles);
            cb_reserve_back(cb_attention_weights, score_tiles);

            // Both mask tiles go to DST once; the packer then stamps them onto every masked slot.
            copy_init(cb_attn_mask);
            tile_regs_acquire();
            copy_tile(cb_attn_mask, /* diagonal pattern */ 0, /* dst */ 0);
            copy_tile(cb_attn_mask, /* all -1e9 */ 1, /* dst */ 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(1);
            const uint32_t step_first_k_tile = k_step * Sk_chunk_t;
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t row_tile = first_row_tile + r % Sq_chunk_t;  // heads share the diagonal
                // K tiles left of this row's diagonal tile stay unmasked.
                const uint32_t first_masked_n = (row_tile > step_first_k_tile) ? (row_tile - step_first_k_tile) : 0U;
                for (uint32_t n = first_masked_n; n < Sk_chunk_t; ++n) {
                    // mask[0] on the diagonal tile, mask[1] (all -1e9) past it.
                    const uint32_t mask_dst = (step_first_k_tile + n == row_tile) ? 0U : 1U;
                    pack_tile</* out_of_order */ true>(mask_dst, cb_attention_weights, /* offset */ r * Sk_chunk_t + n);
                }
            }
            pack_reconfig_l1_acc(0);
            tile_regs_release();

            cb_push_back(cb_attention_weights, score_tiles);
        }
#endif
#ifdef USE_ATTN_MASK
        // Arbitrary mask: the step's Sq_chunk_t x Sk_chunk_t mask tiles (1 keep / 0 drop) become 0 / -1e9 in
        // DST and are added onto the scores in L1 by the packer with L1-accumulate, like the causal stamp.
        // The stacked heads share the mask, so each DST tile is stamped onto every head's row.
        cb_wait_front(cb_attn_mask, mask_tiles);
        cb_wait_front(cb_attention_weights, score_tiles);
        cb_pop_front(cb_attention_weights, score_tiles);
        cb_reserve_back(cb_attention_weights, score_tiles);
        copy_init(cb_attn_mask);
        for (uint32_t t0 = 0; t0 < mask_tiles; t0 += sm_block_size) {
            const uint32_t n = (mask_tiles - t0 < sm_block_size) ? (mask_tiles - t0) : sm_block_size;
            tile_regs_acquire();
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_attn_mask, /* tile_idx */ t0 + i, /* dst */ i);
            }
            binop_with_scalar_tile_init();
            for (uint32_t i = 0; i < n; ++i) {
                add_unary_tile(i, minus_one_bits);   // 1/0 -> 0/-1
                mul_unary_tile(i, custom_inf_bits);  // 0/-1 -> 0/-1e9
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(1);
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t h = 0; h < heads_per_item; ++h) {
                    pack_tile</* out_of_order */ true>(i, cb_attention_weights, /* offset */ h * mask_tiles + t0 + i);
                }
            }
            pack_reconfig_l1_acc(0);
            tile_regs_release();
        }
        cb_push_back(cb_attention_weights, score_tiles);
        cb_pop_front(cb_attn_mask, mask_tiles);
#endif

        cb_pop_front(cb_key, k_tiles);

        // Online softmax. The per-row state helpers batch as many rows as fit in DST per pass.
        // (1) running max per row: max(previous max, max of this step's scores)
        cb_wait_front(cb_attention_weights, score_tiles);
        if (!first_step) {
            cb_wait_front(alias_cb_prev_max, rows);
        }
        cb_reserve_back(alias_cb_cur_max, rows);
        const uint32_t max_rows_per_pass = first_step ? kRowsPerPass : kRowsPerPass / 2U;
        for (uint32_t r0 = 0; r0 < rows; r0 += max_rows_per_pass) {
            update_row_max_values<PoolType::MAX, ReduceDim::REDUCE_ROW, Sk_chunk_t>(
                cb_attention_weights,
                cb_reduction_scaler,
                alias_cb_cur_max,
                alias_cb_prev_max,
                /* do_eltwise_max */ !first_step,
                r0,
                rows_in_pass(r0, max_rows_per_pass));
        }
        cb_push_back(alias_cb_cur_max, rows);
        cb_wait_front(alias_cb_cur_max, rows);

        // (2) alpha = exp(scale * (old max - new max)): the running sum and output of the previous steps
        //     were computed against the old max, so they get scaled by alpha before this step's are added.
        if (!first_step) {
            cb_reserve_back(cb_exp_max_diff, rows);
            for (uint32_t r0 = 0; r0 < rows; r0 += kRowsPerPass) {
                update_exp_max_diffs<scaler_bits, scaler_bf16_bits>(
                    alias_cb_prev_max, alias_cb_cur_max, cb_exp_max_diff, r0, rows_in_pass(r0, kRowsPerPass));
            }
            cb_push_back(cb_exp_max_diff, rows);
            cb_pop_front(alias_cb_prev_max, rows);
        }

        // (3) P = exp(scale * (S - max)) written over S (pop + reserve lands on the same slots), plus
        //     each row's sum of P
        cb_pop_front(cb_attention_weights, score_tiles);
        cb_reserve_back(cb_attention_weights, score_tiles);
        cb_reserve_back(alias_cb_cur_sum_exp, rows);
        apply_exp_inplace_and_find_exp_sum<scaler_bits, scaler_bf16_bits, rows, Sk_chunk_t, sm_block_size>(
            cb_attention_weights, alias_cb_cur_max, alias_cb_cur_sum_exp);
        cb_push_back(cb_attention_weights, score_tiles);
        cb_push_back(alias_cb_cur_sum_exp, rows);

        // (4) O = O_prev * alpha + P @ V, in one DST pass per output block
        matmul_qk_by_v<Sk_chunk_t, rows>(
            vWt,
            pv_block_h,
            pv_block_w,
            cb_attention_weights,
            cb_value,
            alias_cb_prev_mm_out,
            cb_exp_max_diff,
            alias_cb_cur_mm_out,
            /* rescale */ !first_step);

        cb_pop_front(cb_attention_weights, score_tiles);
        cb_pop_front(cb_value, v_tiles);

        // (5) sum = sum_prev * alpha + sum_step, in place (CB holds exactly rows tiles, pop + reserve lands
        //     on the same slots)
        if (!first_step) {
            cb_wait_front(alias_cb_prev_sum_exp, rows);
            cb_wait_front(alias_cb_cur_sum_exp, rows);
            cb_pop_front(alias_cb_cur_sum_exp, rows);
            cb_reserve_back(alias_cb_cur_sum_exp, rows);
            for (uint32_t r0 = 0; r0 < rows; r0 += kRowsPerPass / 2U) {
                update_cur_exp_sums_inplace(
                    alias_cb_prev_sum_exp,
                    alias_cb_cur_sum_exp,
                    cb_exp_max_diff,
                    r0,
                    rows_in_pass(r0, kRowsPerPass / 2U));
            }
            cb_push_back(alias_cb_cur_sum_exp, rows);
            cb_pop_front(alias_cb_prev_sum_exp, rows);

            cb_pop_front(cb_exp_max_diff, rows);
            cb_pop_front(alias_cb_prev_mm_out, out_tiles);
        }

        std::swap(alias_cb_prev_max, alias_cb_cur_max);
        std::swap(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp);
        std::swap(alias_cb_prev_mm_out, alias_cb_cur_mm_out);
    }

    finalize_chunk(alias_cb_prev_mm_out, alias_cb_prev_max, alias_cb_prev_sum_exp);
    cb_pop_front(cb_query, q_tiles);
}

void kernel_main() {
    compute_kernel_hw_startup(cb_query, cb_key, cb_value);
    copy_init(cb_query);
    // compute_kernel_hw_startup above does the one-time HW config; each matmul site below sets its
    // own state with reconfig_data_format + matmul_block_init (or matmul_init for the per-tile paths).
    matmul_init(cb_query, cb_key);

    cb_wait_front(cb_reduction_scaler, onetile);

#ifdef CAUSAL_MASK
    // The two causal mask tiles (see the writer) are waited for once and reused by every chunk.
    constexpr uint32_t kCausalMaskTilesFronted = 2U;
    cb_wait_front(cb_attn_mask, kCausalMaskTilesFronted);
#endif

    for_each_work_item(/* arg_idx */ 0, [](uint32_t item) {
        const uint32_t first_row_tile = (item % q_num_chunks) * Sq_chunk_t;
        process_q_chunk(item, /* k_begin */ 0, num_kv_steps_of_chunk<Ht, Sq_chunk_t, Sk_chunk_t>(first_row_tile));
    });

#ifdef CAUSAL_MASK
    cb_pop_front(cb_attn_mask, kCausalMaskTilesFronted);
#endif
}
