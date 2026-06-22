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
#include "api/compute/transpose_wh.h"
#include "sdpa_compute_utils.hpp"

// For standard mode: num_rows_per_core = rows to process
// For balanced mode: num_rows_per_core = num_pairs (each pair = 2 rows)
constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);       // SFPU block size (cap = dst_size - 1)
constexpr uint32_t qWt = get_compile_time_arg_val(2);              // num tile in inner dim in query/key (d_qk/TILE_W)
constexpr uint32_t vWt = get_compile_time_arg_val(3);              // num tile in inner dim in value (d_v/TILE_W)
constexpr uint32_t Ht = get_compile_time_arg_val(4);               // num_seq_len / TILE_H
constexpr uint32_t scaler_bits = get_compile_time_arg_val(5);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(6);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(7);  // used to transform mask from 0/-1 to 0/-1e9F
[[maybe_unused]] constexpr uint32_t Sk_chunk_t =
    get_compile_time_arg_val(8);  // multi-tile K/V chunking factor (1 = single-tile inner loop)
constexpr uint32_t pv_block_size = get_compile_time_arg_val(9);  // PV matmul_block ct_dim (cap = dst_size)
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
    // Causal / balanced: process K/V tiles up to and including the diagonal tile of this row.
    // The number of K tiles is rounded UP to the next multiple of Sk_chunk_t (the program
    // factory asserts Ht % Sk_chunk_t == 0, so this round-up never reads past the sequence).
    // The "trailing" tiles strictly past the diagonal inside the diagonal chunk are masked
    // out via the all-(-1e9) mask tile (cb_attn_mask[1]).
    const uint32_t num_kv_tiles_to_process = round_up(q_row_tile + 1U, Sk_chunk_t);
#else
    // Non-causal: process every K/V chunk in the sequence.
    const uint32_t num_kv_tiles_to_process = Ht;
#endif
    const uint32_t num_kv_chunks = num_kv_tiles_to_process / Sk_chunk_t;

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Chunk that contains the diagonal tile and the position of the diagonal inside it.
    // For Sk_chunk_t == 1 this is just (q_row_tile, 0) and the post-diagonal branch never fires.
    const uint32_t diag_chunk = q_row_tile / Sk_chunk_t;
    const uint32_t diag_pos_in_chunk = q_row_tile % Sk_chunk_t;
#endif

    // set up ping pong buffers
    uint32_t alias_cb_prev_max = cb_prev_max;
    uint32_t alias_cb_cur_max = cb_cur_max;
    uint32_t alias_cb_prev_sum_exp = cb_prev_sum_exp;
    uint32_t alias_cb_cur_sum_exp = cb_cur_sum_exp;
    uint32_t alias_cb_prev_mm_out = cb_prev_mm_out;
    uint32_t alias_cb_cur_mm_out = cb_cur_mm_out;

    for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
        cb_wait_front(cb_key, Sk_chunk_t * qWt);

        reconfig_data_format(cb_query, cb_key);
        pack_reconfig_data_format(cb_attention_weights);

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
        // ====== Block matmul QK^T, then packer L1-accumulate mask stamp ======
        //
        // Phase 1: matmul_block produces all Sk_chunk_t score tiles into DST[0..Sk_chunk_t-1] and
        // packs them to cb_attention_weights (FP32 CB) unmasked. K is col-major in cb_key (feat
        // outer, seq inner) so matmul_block's MOP steps in1_idx by +Sk_chunk_t per feat
        // (TTNN's `matmul_blocks` idiom in `compute_common.hpp`).
        //
        // Phase 2 (diagonal chunk only): mask is *L1-accumulated* onto the already-packed
        // scores using TTNN's `apply_causal_mask_lightweight` pattern from
        // `compute_common.hpp`. We pop+reserve cb_attention_weights to re-enter reserved state
        // on the same physical L1 slots (CB is single-buffered, size=Sk_chunk_t, so the wr_ptr
        // wraps back). Then `pack_reconfig_l1_acc(1)` enables the packer's read-modify-write:
        // read FP32 score from L1, add the DST tile (mask) in FP32, write back FP32. Score
        // stays at full FP32 precision the whole time — no DST→SRC conversion truncation
        // to TF32.
        matmul_block_init(
            cb_query,
            cb_key,
            /* transpose */ 1,
            /* ct_dim */ Sk_chunk_t,
            /* rt_dim */ 1,
            /* kt_dim */ qWt);
        tile_regs_acquire();
        {
            // One matmul_block call does ONE K-step (1 Q tile × ct_dim K tiles → ct_dim
            // accumulated outputs). Iterate qWt times to contract the full feature dim.
            uint32_t q_idx = 0;
            uint32_t k_idx = 0;
            for (uint32_t feat = 0; feat < qWt; ++feat) {
                matmul_block(
                    cb_query,
                    cb_key,
                    /* in0 (Q) tile_idx */ q_idx,
                    /* in1 (K) tile_idx */ k_idx,
                    /* dst_idx */ 0,
                    /* transpose */ 1,
                    /* ct_dim */ Sk_chunk_t,
                    /* rt_dim */ 1,
                    /* kt_dim */ qWt);
                q_idx += 1U;
                k_idx += Sk_chunk_t;
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_attention_weights, Sk_chunk_t);
        for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
            pack_tile(n, cb_attention_weights);
        }
        cb_push_back(cb_attention_weights, Sk_chunk_t);
        tile_regs_release();

        if (k_chunk == diag_chunk) {
            // Re-enter reserved state on the same L1 slots (single-buffered CB wraps back).
            cb_wait_front(cb_attention_weights, Sk_chunk_t);
            cb_pop_front(cb_attention_weights, Sk_chunk_t);
            cb_reserve_back(cb_attention_weights, Sk_chunk_t);

            // Unpacker → mask CB (BF16); packer stays on cb_attention_weights (FP32).
            copy_tile_to_dst_init_short(cb_attn_mask);
            pack_reconfig_l1_acc(1);
            for (uint32_t n = diag_pos_in_chunk; n < Sk_chunk_t; ++n) {
                // n == diag_pos_in_chunk → mask[0] (causal diagonal pattern).
                // n  > diag_pos_in_chunk → mask[1] (all -1e9, strictly past the diagonal).
                const uint32_t mask_tile_idx = (n == diag_pos_in_chunk) ? 0U : 1U;
                tile_regs_acquire();
                copy_tile(cb_attn_mask, mask_tile_idx, /* dst */ 0);
                tile_regs_commit();
                tile_regs_wait();
                // pack_tile<true> packs DST[0] to cb_attention_weights[n]; with l1_acc=1 the
                // packer reads the existing FP32 score, adds the FP32 mask in DST, writes FP32.
                pack_tile</* out_of_order */ true>(/* dst */ 0, cb_attention_weights, /* offset */ n);
                tile_regs_release();
            }
            pack_reconfig_l1_acc(0);

            cb_push_back(cb_attention_weights, Sk_chunk_t);
        }
#else
        // USE_ATTN_MASK (provided mask) — uses a per-`n` matmul_tiles + apply_mask_on_reg path.
        // The per-n mask is unique (no reusable transformed tiles) and apply_mask_on_reg operates
        // on a single DST tile + scratch, which fits comfortably here. K is laid out col-major
        // in cb_key (uniform reader layout), so the K tile index is `feat*Sk_chunk_t + n`.
        constexpr uint32_t matmul_accum_reg = 0U;
        for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
            matmul_init(cb_query, cb_key, /* transpose */ 1);
            tile_regs_acquire();
            for (uint32_t tile_idx = 0; tile_idx < qWt; ++tile_idx) {
                matmul_tiles(
                    cb_query,
                    cb_key,
                    /* Q tile */ tile_idx,
                    /* K tile (col-major within chunk): tile (n, feat) at CB pos feat*Sk_chunk_t + n */
                    tile_idx * Sk_chunk_t + n,
                    /* dst */ matmul_accum_reg);
            }
            apply_mask_on_reg(matmul_accum_reg, cb_attn_mask, minus_one_bits, custom_inf_bits, /* mask_tile_idx */ n);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_attention_weights, onetile);
            pack_tile(matmul_accum_reg, cb_attention_weights);
            tile_regs_release();
            cb_push_back(cb_attention_weights, onetile);
        }

        // Every mask tile is unique per (row, k tile). Pop the whole chunk's worth so the
        // reader can stage the next chunk.
        cb_pop_front(cb_attn_mask, Sk_chunk_t);
#endif
        // CAUSAL_MASK/BALANCED_PARALLELISM: the two mask tiles stay permanently fronted; no pop.

        // Done with this chunk's K block.
        cb_pop_front(cb_key, Sk_chunk_t * qWt);

        // Online softmax step over this chunk (max, exp, partial sum) and PV matmul.
        update_cur_row_max_value<PoolType::MAX, ReduceDim::REDUCE_ROW, Sk_chunk_t>(
            cb_attention_weights,
            cb_reduction_scaler,
            alias_cb_cur_max,
            alias_cb_prev_max,
            /* do_eltwise_max */ k_chunk > 0);

        apply_exp_inplace_and_find_exp_sum<scaler_bits, Sk_chunk_t>(
            cb_attention_weights, alias_cb_cur_max, alias_cb_cur_sum_exp);

        matmul_qk_by_v<Sk_chunk_t>(vWt, pv_block_size, cb_attention_weights, cb_value, alias_cb_cur_mm_out);

        cb_pop_front(cb_attention_weights, Sk_chunk_t);
        cb_pop_front(cb_value, Sk_chunk_t * vWt);

        // Online correction against the previous chunk's running stats.
        if (k_chunk > 0) {
            update_exp_max_diff<scaler_bits>(alias_cb_prev_max, alias_cb_cur_max, cb_exp_max_diff);
            cb_pop_front(alias_cb_prev_max, onetile);

            update_cur_exp_sum_inplace(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp, cb_exp_max_diff);
            cb_pop_front(alias_cb_prev_sum_exp, onetile);

            update_cur_mm_out(vWt, block_size, alias_cb_prev_mm_out, alias_cb_cur_mm_out, cb_exp_max_diff);

            cb_pop_front(cb_exp_max_diff, onetile);
            cb_pop_front(alias_cb_prev_mm_out, vWt);
        }

        std::swap(alias_cb_prev_max, alias_cb_cur_max);
        std::swap(alias_cb_prev_sum_exp, alias_cb_cur_sum_exp);
        std::swap(alias_cb_prev_mm_out, alias_cb_cur_mm_out);
    }

    // Finalize output
    cb_wait_front(alias_cb_prev_mm_out, vWt);

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

    cb_reserve_back(cb_output, vWt);
    pack_reconfig_data_format(cb_output);
    for (uint32_t tile_idx = 0; tile_idx < vWt; tile_idx += block_size) {
        tile_regs_acquire();

        // Load mm_out tiles via UnpackToDestFp32 (full FP32 in DST)
        reconfig_data_format(alias_cb_prev_mm_out, alias_cb_prev_mm_out);
        copy_tile_init(alias_cb_prev_mm_out);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(alias_cb_prev_mm_out, tile_idx + block_idx, block_idx);
        }

        // Load 1/sum_exp with column broadcast to DST[block_size]
        init_unary_bcast_col(alias_cb_prev_sum_exp);
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
    cb_push_back(cb_output, vWt);

    cb_pop_front(alias_cb_prev_max, onetile);
    cb_pop_front(alias_cb_prev_sum_exp, onetile);
    cb_pop_front(alias_cb_prev_mm_out, vWt);
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
    // binary_op_init_common above does the one-time HW config; each matmul site below
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

        process_single_row(heavy_global_row);
        process_single_row(light_global_row);
    }
#else
    // Standard mode: process rows sequentially
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        const uint32_t global_row_idx = start_idx + row;
        process_single_row(global_row_idx);
    }
#endif

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Pop the two causal mask tiles after all rows are processed (reused for every diagonal/chunk).
    cb_pop_front(cb_attn_mask, kCausalMaskTilesFronted);
#endif
}
