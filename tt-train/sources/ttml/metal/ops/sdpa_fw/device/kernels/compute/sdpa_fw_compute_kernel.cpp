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

// How to read this kernel: the reader kernel pushes Q, K and V tiles into circular buffers (CBs);
// this kernel waits for them (cb_wait_front), computes in the DST register file between
// tile_regs_acquire and tile_regs_release, packs results into CBs (cb_reserve_back + pack_tile +
// cb_push_back) and releases inputs (cb_pop_front). The writer drains cb_output / cb_intermediates.
//
// Work unit: one Q chunk = Sq_chunk_t query tile-rows of one batch/head. The chunk walks K/V in
// steps of Sk_chunk_t tiles; per step it does QK^T, online softmax and P@V for all its rows.

// For standard mode: num_units_per_core = Q chunks to process
// For balanced mode: num_units_per_core = pairs of Q chunks (max over cores; runtime arg has the actual count)
constexpr uint32_t num_units_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);       // output tiles per DST pass, element-wise loops
constexpr uint32_t qWt = get_compile_time_arg_val(2);              // num tile in inner dim in query/key (d_qk/TILE_W)
constexpr uint32_t vWt = get_compile_time_arg_val(3);              // num tile in inner dim in value (d_v/TILE_W)
constexpr uint32_t Ht = get_compile_time_arg_val(4);               // num_seq_len / TILE_H
constexpr uint32_t scaler_bits = get_compile_time_arg_val(5);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(6);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(7);  // used to transform mask from 0/-1 to 0/-1e9F
constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);       // K/V tiles per step
constexpr uint32_t pv_block_w = get_compile_time_arg_val(9);       // V feature tiles per PV matmul_block call
constexpr uint32_t scaler_bf16_bits =
    get_compile_time_arg_val(10);  // host-side RNE BF16 of scaler_bits, for the WH fused scale+exp
constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(11);     // query tile-rows per Q chunk
constexpr uint32_t qk_block_h = get_compile_time_arg_val(12);     // Q rows per QK^T matmul_block call
constexpr uint32_t qk_block_w = get_compile_time_arg_val(13);     // K tiles per QK^T matmul_block call
constexpr uint32_t pv_block_h = get_compile_time_arg_val(14);     // Q rows per PV matmul_block call
constexpr uint32_t sm_block_size = get_compile_time_arg_val(15);  // score tiles of one row per DST pass, exp step

constexpr uint32_t q_num_chunks = Ht / Sq_chunk_t;
constexpr uint32_t pairs_per_seq = q_num_chunks / 2;
constexpr uint32_t q_tiles = Sq_chunk_t * qWt;          // Q tiles of one chunk
constexpr uint32_t score_tiles = Sq_chunk_t * Sk_chunk_t;  // QK^T tiles of one K/V step
constexpr uint32_t out_tiles = Sq_chunk_t * vWt;        // output / accumulator tiles of one chunk
constexpr uint32_t k_tiles = Sk_chunk_t * qWt;          // K tiles of one step
constexpr uint32_t v_tiles = Sk_chunk_t * vWt;          // V tiles of one step

static_assert(Sq_chunk_t % qk_block_h == 0U && Sk_chunk_t % qk_block_w == 0U, "QK^T block must tile the chunk");
static_assert(Sq_chunk_t % pv_block_h == 0U, "PV block must tile the Q chunk");

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
constexpr uint32_t cb_gate = tt::CBIndex::c_16;

// Layouts used below. Score CB (cb_attention_weights): the Sq_chunk_t x Sk_chunk_t score tiles of one
// step, row-major: row r, K tile n sits at r * Sk_chunk_t + n. Per-row state CBs (max, sum,
// exp_max_diff): row r sits in slot r. Output accumulators: row r starts at r * vWt.
FORCE_INLINE void process_q_chunk(uint32_t global_first_row) {
    cb_wait_front(cb_query, q_tiles);

    const uint32_t first_row_tile = global_first_row % Ht;

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Causal: only K/V tiles up to the chunk's last diagonal tile, rounded up to whole steps.
    // Steps before first_masked_step lie entirely left of the diagonal and need no masking.
    const uint32_t num_kv_tiles_to_process = round_up(first_row_tile + Sq_chunk_t, Sk_chunk_t);
    const uint32_t first_masked_step = first_row_tile / Sk_chunk_t;
#else
    const uint32_t num_kv_tiles_to_process = Ht;
#endif
    const uint32_t num_kv_steps = num_kv_tiles_to_process / Sk_chunk_t;

    // prev/cur state CBs swap roles after every step
    uint32_t alias_cb_prev_max = cb_prev_max;
    uint32_t alias_cb_cur_max = cb_cur_max;
    uint32_t alias_cb_prev_sum_exp = cb_prev_sum_exp;
    uint32_t alias_cb_cur_sum_exp = cb_cur_sum_exp;
    uint32_t alias_cb_prev_mm_out = cb_prev_mm_out;
    uint32_t alias_cb_cur_mm_out = cb_cur_mm_out;

    for (uint32_t k_step = 0; k_step < num_kv_steps; ++k_step) {
        cb_wait_front(cb_key, k_tiles);

        reconfig_data_format(cb_query, cb_key);
        pack_reconfig_data_format(cb_attention_weights);

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
        // S = Q @ K^T, one qk_block_h x qk_block_w rectangle of score tiles per DST pass, packed to its
        // row-major slot. The causal mask is added afterwards by the packer (see below).
        matmul_block_init(
            cb_query,
            cb_key,
            /* transpose */ 1,
            /* ct_dim */ qk_block_w,
            /* rt_dim */ qk_block_h,
            /* kt_dim */ qWt);
        cb_reserve_back(cb_attention_weights, score_tiles);
        for (uint32_t row0 = 0; row0 < Sq_chunk_t; row0 += qk_block_h) {
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
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                const uint32_t row_tile = first_row_tile + r;
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
#else
        // Arbitrary / no mask: one score tile at a time, mask applied while it is still in DST.
        constexpr uint32_t matmul_accum_reg = 0U;
#ifdef USE_ATTN_MASK
        cb_wait_front(cb_attn_mask, score_tiles);
#endif
        cb_reserve_back(cb_attention_weights, score_tiles);
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
                // apply_mask_on_reg's SFPU inits clobber the FPU matmul state: re-init per tile.
                matmul_init(cb_query, cb_key, /* transpose */ 1);
                tile_regs_acquire();
                for (uint32_t tile_idx = 0; tile_idx < qWt; ++tile_idx) {
                    matmul_tiles(
                        cb_query,
                        cb_key,
                        /* Q tile */ r * qWt + tile_idx,
                        /* K tile, col-major */
                        tile_idx * Sk_chunk_t + n,
                        /* dst */ matmul_accum_reg);
                }
#ifdef USE_ATTN_MASK
                apply_mask_on_reg(
                    matmul_accum_reg,
                    cb_attn_mask,
                    minus_one_bits,
                    custom_inf_bits,
                    /* mask_tile_idx */ r * Sk_chunk_t + n);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_tile</* out_of_order */ true>(
                    matmul_accum_reg, cb_attention_weights, /* offset */ r * Sk_chunk_t + n);
                tile_regs_release();
            }
        }
        cb_push_back(cb_attention_weights, score_tiles);

#ifdef USE_ATTN_MASK
        cb_pop_front(cb_attn_mask, score_tiles);
#endif
#endif

        cb_pop_front(cb_key, k_tiles);

        // Online softmax, row by row.
        // (1) running max per row: max(previous max, max of this step's scores)
        cb_wait_front(cb_attention_weights, score_tiles);
        if (k_step > 0) {
            cb_wait_front(alias_cb_prev_max, Sq_chunk_t);
        }
        cb_reserve_back(alias_cb_cur_max, Sq_chunk_t);
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            update_cur_row_max_value<PoolType::MAX, ReduceDim::REDUCE_ROW, Sk_chunk_t>(
                cb_attention_weights,
                cb_reduction_scaler,
                alias_cb_cur_max,
                alias_cb_prev_max,
                /* do_eltwise_max */ k_step > 0,
                r);
        }
        cb_push_back(alias_cb_cur_max, Sq_chunk_t);
        cb_wait_front(alias_cb_cur_max, Sq_chunk_t);

        // (2) P = exp(scale * (S - max)) written over S (pop + reserve lands on the same slots), plus
        //     each row's sum of P
        cb_pop_front(cb_attention_weights, score_tiles);
        cb_reserve_back(cb_attention_weights, score_tiles);
        cb_reserve_back(alias_cb_cur_sum_exp, Sq_chunk_t);
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            apply_exp_inplace_and_find_exp_sum<scaler_bits, scaler_bf16_bits, Sk_chunk_t, sm_block_size>(
                cb_attention_weights, alias_cb_cur_max, alias_cb_cur_sum_exp, r);
        }
        cb_push_back(cb_attention_weights, score_tiles);
        cb_push_back(alias_cb_cur_sum_exp, Sq_chunk_t);

        // (3) O_step = P @ V
        matmul_qk_by_v<Sk_chunk_t, Sq_chunk_t>(
            vWt, pv_block_h, pv_block_w, cb_attention_weights, cb_value, alias_cb_cur_mm_out);

        cb_pop_front(cb_attention_weights, score_tiles);
        cb_pop_front(cb_value, v_tiles);

        // (4) fold into the running sum and output: both from previous steps were computed against the
        //     old max, so scale them by exp(scale * (old max - new max)) before adding this step's.
        if (k_step > 0) {
            cb_reserve_back(cb_exp_max_diff, Sq_chunk_t);
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                update_exp_max_diff<scaler_bits, scaler_bf16_bits>(
                    alias_cb_prev_max, alias_cb_cur_max, cb_exp_max_diff, r);
            }
            cb_push_back(cb_exp_max_diff, Sq_chunk_t);
            cb_pop_front(alias_cb_prev_max, Sq_chunk_t);

            cb_wait_front(cb_exp_max_diff, Sq_chunk_t);
            cb_wait_front(alias_cb_prev_sum_exp, Sq_chunk_t);
            cb_wait_front(alias_cb_cur_sum_exp, Sq_chunk_t);
            // in place (CB holds exactly Sq_chunk_t tiles, pop + reserve lands on the same slots)
            cb_pop_front(alias_cb_cur_sum_exp, Sq_chunk_t);
            cb_reserve_back(alias_cb_cur_sum_exp, Sq_chunk_t);
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                update_cur_exp_sum_inplace(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp, cb_exp_max_diff, r);
            }
            cb_push_back(alias_cb_cur_sum_exp, Sq_chunk_t);
            cb_pop_front(alias_cb_prev_sum_exp, Sq_chunk_t);

            update_cur_mm_out<Sq_chunk_t>(vWt, block_size, alias_cb_prev_mm_out, alias_cb_cur_mm_out, cb_exp_max_diff);

            cb_pop_front(cb_exp_max_diff, Sq_chunk_t);
            cb_pop_front(alias_cb_prev_mm_out, out_tiles);
        }

        std::swap(alias_cb_prev_max, alias_cb_cur_max);
        std::swap(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp);
        std::swap(alias_cb_prev_mm_out, alias_cb_cur_mm_out);
    }

    // ====== Finalize the chunk's rows: O / sum, and lse for the backward pass ======
    cb_wait_front(alias_cb_prev_mm_out, out_tiles);
    cb_wait_front(alias_cb_prev_max, Sq_chunk_t);
    cb_wait_front(alias_cb_prev_sum_exp, Sq_chunk_t);
    cb_wait_front(cb_matmul_reduce, onetile);

    // The row sum so far is per 32-column tile column; reduce it to one value per row (in place).
    cb_pop_front(alias_cb_prev_sum_exp, Sq_chunk_t);
    cb_reserve_back(alias_cb_prev_sum_exp, Sq_chunk_t);
    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
        row_reduce_tile_inplace<cb_reduction_scaler, cb_matmul_reduce>(alias_cb_prev_sum_exp, r);
    }
    cb_push_back(alias_cb_prev_sum_exp, Sq_chunk_t);
    cb_wait_front(alias_cb_prev_sum_exp, Sq_chunk_t);

#ifdef RETURN_INTERMEDIATES
    // lse = scale*max + log(sum_exp)
    cb_reserve_back(cb_intermediates, Sq_chunk_t);
    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
        compute_and_pack_lse(
            alias_cb_prev_sum_exp, alias_cb_prev_max, cb_intermediates, cb_matmul_reduce, scaler_bits, r);
    }
    cb_push_back(cb_intermediates, Sq_chunk_t);
#endif

    // 1/sum_exp for normalization
    cb_pop_front(alias_cb_prev_sum_exp, Sq_chunk_t);
    cb_reserve_back(alias_cb_prev_sum_exp, Sq_chunk_t);
    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
        recip_tile_inplace(alias_cb_prev_sum_exp, r);
    }
    cb_push_back(alias_cb_prev_sum_exp, Sq_chunk_t);
    cb_wait_front(alias_cb_prev_sum_exp, Sq_chunk_t);

    cb_reserve_back(cb_output, out_tiles);
#ifdef HAS_GATE
    cb_wait_front(cb_gate, out_tiles);
#endif
    pack_reconfig_data_format(cb_output);
    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
        for (uint32_t tile_idx = 0; tile_idx < vWt; tile_idx += block_size) {
            tile_regs_acquire();

            // Load mm_out tiles via UnpackToDestFp32 (full FP32 in DST)
            reconfig_data_format(alias_cb_prev_mm_out, alias_cb_prev_mm_out);
            copy_init(alias_cb_prev_mm_out);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                copy_tile(alias_cb_prev_mm_out, r * vWt + tile_idx + block_idx, block_idx);
            }

            // Load this row's 1/sum_exp with column broadcast to DST[block_size]
            init_unary_bcast_col(alias_cb_prev_sum_exp);
            unary_bcast<BroadcastType::COL>(alias_cb_prev_sum_exp, r, block_size);

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

    cb_pop_front(alias_cb_prev_max, Sq_chunk_t);
    cb_pop_front(alias_cb_prev_sum_exp, Sq_chunk_t);
    cb_pop_front(alias_cb_prev_mm_out, out_tiles);
    cb_pop_front(cb_query, q_tiles);
#ifdef HAS_GATE
    cb_pop_front(cb_gate, out_tiles);
#endif
}

void kernel_main() {
    // Runtime args
    // For standard mode: arg0 = first Q chunk of this core
    // For balanced mode: arg0 = first pair of this core, arg1 = number of pairs
    const uint32_t start_unit = get_arg_val<uint32_t>(0);

#ifdef BALANCED_PARALLELISM
    const uint32_t num_pairs = get_arg_val<uint32_t>(1);  // Runtime arg for balanced mode
#endif

    compute_kernel_hw_startup(cb_query, cb_key, cb_value);
    copy_init(cb_query);
    // compute_kernel_hw_startup above does the one-time HW config; each matmul site below
    // re-establishes its state with reconfig_data_format + matmul_init.
    matmul_init(cb_query, cb_key);

    cb_wait_front(cb_reduction_scaler, onetile);

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Wait for the two pre-transformed causal mask tiles ONCE — generated by the writer and
    // reused for every row:
    //   tile[0] = causal-diagonal pattern (0.0 on/below diagonal, -1e9 above)
    //   tile[1] = all -1e9 — used for K tiles strictly past the diagonal inside the diagonal
    //             chunk when Sk_chunk_t > 1.
    constexpr uint32_t kCausalMaskTilesFronted = 2U;
    cb_wait_front(cb_attn_mask, kCausalMaskTilesFronted);
#endif

#ifdef BALANCED_PARALLELISM
    // Balanced: pairs of Q chunks, heavy chunk first, then light.
    for (uint32_t p = 0; p < num_pairs; ++p) {
        const uint32_t global_pair_idx = start_unit + p;
        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t heavy_chunk_in_seq = q_num_chunks - 1U - pair_in_seq;
        const uint32_t light_chunk_in_seq = pair_in_seq;

        process_q_chunk(seq_idx * Ht + heavy_chunk_in_seq * Sq_chunk_t);
        process_q_chunk(seq_idx * Ht + light_chunk_in_seq * Sq_chunk_t);
    }
#else
    // Standard mode: consecutive Q chunks
    for (uint32_t u = 0; u < num_units_per_core; ++u) {
        const uint32_t global_q_chunk_idx = start_unit + u;
        const uint32_t seq_idx = global_q_chunk_idx / q_num_chunks;
        const uint32_t chunk_in_seq = global_q_chunk_idx % q_num_chunks;
        process_q_chunk(seq_idx * Ht + chunk_in_seq * Sq_chunk_t);
    }
#endif

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Pop the two causal mask tiles after all rows are processed (reused for every diagonal/chunk).
    cb_pop_front(cb_attn_mask, kCausalMaskTilesFronted);
#endif
}
