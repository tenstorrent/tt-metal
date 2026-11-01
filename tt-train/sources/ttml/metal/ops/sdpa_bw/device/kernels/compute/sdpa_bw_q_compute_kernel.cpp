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
#include "sdpa_bw_compute_utils.hpp"

// ----------------------------------------------------------------------
// SDPA Backward Compute Kernel for Query Gradient (dQ)
// ----------------------------------------------------------------------
//
// Forward pass (for reference):
//   P = softmax(Q @ K^T / sqrt(d) + mask)    // attention weights [B, H, S, S]
//   O = P @ V                                 // attention output [B, H, S, D]
//   Y = O ⊙ σ(G)                              // [HAS_GATE] gated output; otherwise Y = O
//
// Backward pass computes dQ given dY (upstream gradient of the forward's output):
//   dO = dY ⊙ σ(G)                            // [HAS_GATE] otherwise dO = dY
//   dG = dY ⊙ Y ⊙ (1 − σ(G))                  // [HAS_GATE] gradient w.r.t. the gate
//   dP = dO @ V^T                             // gradient w.r.t. attention weights
//   u  = rowsum(dO ⊙ O) = rowsum(dY ⊙ Y)      // per-row scalar for softmax backward
//   dS = P ⊙ (dP - u)                         // softmax backward (element-wise)
//   dQ = (1/sqrt(d)) * dS @ K                 // gradient w.r.t. query
//
// Note: We apply the scale factor inside dS computation for numerical stability.
//
// With HAS_GATE the kernel receives dY as grad_output and Y as attn_output (the ungated O is
// never materialized). dO is packed twice: into a compute-local CB used as the dP operand, and
// into a second CB that the writer kernel stores to DRAM for the KV kernel. u is computed
// directly from dY and Y using the identity above.
//
// Processing order:
//   for each query row q:
//     compute u_scalar = rowsum(dY[q] ⊙ Y[q])
//     [HAS_GATE] compute dO[q] = dY[q] ⊙ σ(G[q]), dG[q] = dY[q] ⊙ Y[q] ⊙ (1 − σ(G[q]))
//     for each block of kv_block K/V rows k0 .. k0 + cols (the chain chunk under CHAIN):
//       P[q,k]  = softmax(Q[q] @ K[k]^T / sqrt(d) + mask[q,k])  // recomputed, two columns per DEST pass
//       dP[q,k] = dO[q] @ V[k]^T                                 // two columns per DEST pass
//       dS[q,k] = P[q,k] * (dP[q,k] - u_scalar) * scale          // two columns per DEST pass
//       dQ[q] += dS[q, k0..] @ K[k0.., :]                        // one DEST pass per output block (iteration 22)
// The fp32 DEST holds four tiles: two score tiles plus the mask / lse tile, or two dP and two P tiles. The
// dQ pass accumulates the block's columns in DEST and adds to the L1 accumulator once per block instead of
// once per column.
// ----------------------------------------------------------------------

// For standard mode: num_rows_per_core = rows to process
// For balanced mode: num_rows_per_core = num_pairs (each pair = 2 rows)
constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t qWt = get_compile_time_arg_val(1);              // Q/K inner dim in tiles
constexpr uint32_t vWt = get_compile_time_arg_val(2);              // V/dO/O inner dim in tiles
constexpr uint32_t Ht = get_compile_time_arg_val(3);               // num_seq_len / TILE_H
constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(5);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(6);  // used to transform mask from 0/-1 to 0/-inf
constexpr uint32_t block_size = get_compile_time_arg_val(7);       // block size for update_grad_query (divides qWt)
constexpr uint32_t kv_block = get_compile_time_arg_val(8);         // K/V rows per block (the chain chunk under CHAIN)
static_assert(kv_block >= 1U && kv_block <= 4U, "a K/V block is at most four rows");

constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;  // Gradient w.r.t. output
constexpr uint32_t cb_attn_output = tt::CBIndex::c_1;  // (Gated) attention output from forward pass
constexpr uint32_t cb_query = tt::CBIndex::c_2;        // Original query
constexpr uint32_t cb_key = tt::CBIndex::c_3;          // Original key
constexpr uint32_t cb_value = tt::CBIndex::c_4;        // Original value
#if defined(CAUSAL_MASK) || defined(USE_ATTN_MASK)
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;  // Attention mask (causal or arbitrary)
#endif
constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;       // Forward pass intermediates
constexpr uint32_t cb_mat_mul_reduction = tt::CBIndex::c_7;   // Temporary computations
constexpr uint32_t cb_grad_query_accum = tt::CBIndex::c_8;    // L1 accumulator for grad_query
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_9;   // Recomputed attention weights = softmax(QK^T / sqrt(Et))
constexpr uint32_t cb_grad_attn_weights = tt::CBIndex::c_10;  // Gradient w.r.t. attention: dL/dP
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_11;        // Gradient w.r.t. QK scores
constexpr uint32_t cb_u_scalar_row = tt::CBIndex::c_12;       // u_scalar per row
constexpr uint32_t cb_grad_query = tt::CBIndex::c_13;         // Output: grad_Q
constexpr uint32_t cb_u_scaler_output = tt::CBIndex::c_14;    // Output: u_scaler to DRAM for KV kernel
#ifdef HAS_GATE
constexpr uint32_t cb_gate = tt::CBIndex::c_15;                     // Input: gate G
constexpr uint32_t cb_grad_gate = tt::CBIndex::c_16;                // Output: grad_gate
constexpr uint32_t cb_gated_grad_output = tt::CBIndex::c_17;        // Output: gated dO to DRAM for KV kernel
constexpr uint32_t cb_gated_grad_output_local = tt::CBIndex::c_18;  // Compute-local gated dO for dP matmul
// Operand for dP = dO @ V^T
constexpr uint32_t cb_dp_grad_output = cb_gated_grad_output_local;
#else
constexpr uint32_t cb_dp_grad_output = cb_grad_output;
#endif

const uint32_t qk_tiles = qWt;            // Q/K inner dim tiles
const uint32_t v_tiles = vWt;             // V/dO/O inner dim tiles
const uint32_t num_of_interm_tiles = 1U;  // single FP32 logsumexp tile per Q row

/**
 * Process a single row of the SDPA backward Q computation.
 * This function handles the full backward attention computation for one query row:
 * - Recompute P = softmax(Q @ K^T / sqrt(d) + mask) using stored intermediates
 * - Compute dP = dO @ V^T
 * - Compute dS = P * (dP - u) * scale (softmax backward)
 * - Accumulate dQ += dS @ K
 *
 * @param global_row_idx The global row index (across all batches/heads/sequences)
 */
FORCE_INLINE void process_single_row(uint32_t global_row_idx) {
    cb_wait_front(cb_attn_output, v_tiles);
    cb_wait_front(cb_grad_output, v_tiles);
    cb_wait_front(cb_query, qk_tiles);

    compute_u_scalar_row(
        cb_grad_output,
        cb_attn_output,
        cb_u_scalar_row,
        cb_mat_mul_reduction,
        v_tiles,
        scaler_bits,
        cb_u_scaler_output);

#ifdef HAS_GATE
    compute_gated_grad_output_and_grad_gate(
        cb_grad_output,
        cb_attn_output,
        cb_gate,
        cb_gated_grad_output_local,
        cb_gated_grad_output,
        cb_grad_gate,
        v_tiles);
    cb_wait_front(cb_gated_grad_output_local, v_tiles);
    // The gate helper leaves PACK in BF16; the loop below only reconfigures conditionally
    // from FP32, so restore FP32 here.
    pack_reconfig_data_format(cb_attention_weights);
#endif

    const uint32_t q_row_tile = global_row_idx % Ht;

#ifdef CAUSAL_MASK
    const uint32_t num_kv_tiles_to_process = q_row_tile + 1;
#else
    const uint32_t num_kv_tiles_to_process = Ht;
#endif

    cb_wait_front(cb_intermediates, num_of_interm_tiles);

    // K/V rows in blocks of kv_block: the P, dP and dS tiles of a block's columns are produced two per
    // DEST pass and kept in fixed kv_block-tile CB slots (a partial last block leaves its tail slots
    // unused, so every push / pop moves kv_block tiles and never straddles the CB end).
    for (uint32_t h0 = 0; h0 < num_kv_tiles_to_process; h0 += kv_block) {
        const uint32_t cols = (num_kv_tiles_to_process - h0 < kv_block) ? (num_kv_tiles_to_process - h0) : kv_block;
#ifdef CHAIN
        cb_wait_front(cb_key, kv_block * qk_tiles);  // the whole chunk arrives at once
        cb_wait_front(cb_value, kv_block * v_tiles);
#else
        cb_wait_front(cb_key, cols * qk_tiles);
        cb_wait_front(cb_value, cols * v_tiles);
#endif

        // (A) P = exp(scale * Q K^T [+ mask] - lse): scores in DST[0 .. n), the mask then the lse tile in DST[n].
        cb_reserve_back(cb_attention_weights, kv_block);
        for (uint32_t c0 = 0; c0 < cols; c0 += 2U) {
            const uint32_t n = (cols - c0 < 2U) ? (cols - c0) : 2U;
            reconfig_data_format(cb_query, cb_key);
            matmul_init(cb_query, cb_key, /*transpose*/ 1);
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t f = 0; f < qk_tiles; ++f) {
                    matmul_tiles(cb_query, cb_key, /* in0 */ f, /* in1 */ (c0 + j) * qk_tiles + f, /* dst */ j);
                }
            }
            for (uint32_t j = 0; j < n; ++j) {
#ifdef CAUSAL_MASK
                if (h0 + c0 + j == q_row_tile) {
                    apply_mask_on_reg_at(j, n, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
                } else {
                    binop_with_scalar_tile_init();
                    mul_unary_tile(j, scaler_bits);
                }
#elif defined(USE_ATTN_MASK)
                apply_mask_on_reg_at(j, n, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
                cb_pop_front(cb_attn_mask, onetile);
#else
                binop_with_scalar_tile_init();
                mul_unary_tile(j, scaler_bits);
#endif
            }
            // Fused softmax on DST: P = exp(S - lse), full fp32, no CB round trip.
            apply_softmax_statistics_on_dst_n(/* first_reg */ 0, n, /* lse_reg */ n, cb_intermediates);
            tile_regs_commit();
            tile_regs_wait();
            // 2-arg pack_reconfig: skip reprogram if new PACK format matches the previously-configured one.
            pack_reconfig_data_format(cb_grad_query_accum, cb_attention_weights);
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_attention_weights);
            }
            tile_regs_release();
        }
        cb_push_back(cb_attention_weights, kv_block);

        // (B) dP = dO V^T, two columns per DEST pass.
        cb_reserve_back(cb_grad_attn_weights, kv_block);
        reconfig_data_format(cb_dp_grad_output, cb_value);
        matmul_init(cb_dp_grad_output, cb_value, /* transpose */ 1);
        for (uint32_t c0 = 0; c0 < cols; c0 += 2U) {
            const uint32_t n = (cols - c0 < 2U) ? (cols - c0) : 2U;
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t f = 0; f < v_tiles; ++f) {
                    matmul_tiles(
                        cb_dp_grad_output, cb_value, /* in0 */ f, /* in1 */ (c0 + j) * v_tiles + f, /* dst */ j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_attention_weights, cb_grad_attn_weights);
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_grad_attn_weights);
            }
            tile_regs_release();
        }
        cb_push_back(cb_grad_attn_weights, kv_block);

        // (C) dS = P * (dP - u) * scale: DST[0 .. n) = dP - u, DST[2 .. 2 + n) = P (fp32 copy), product in place.
        cb_wait_front(cb_attention_weights, kv_block);
        cb_wait_front(cb_grad_attn_weights, kv_block);
        cb_wait_front(cb_u_scalar_row, onetile);
        cb_reserve_back(cb_grad_scores, kv_block);
        for (uint32_t c0 = 0; c0 < cols; c0 += 2U) {
            const uint32_t n = (cols - c0 < 2U) ? (cols - c0) : 2U;
            tile_regs_acquire();
            reconfig_data_format(cb_grad_attn_weights, cb_u_scalar_row);
            sub_bcast_cols_init(cb_grad_attn_weights, cb_u_scalar_row);
            for (uint32_t j = 0; j < n; ++j) {
                sub_tiles_bcast_cols(cb_grad_attn_weights, cb_u_scalar_row, /* tile_idx */ c0 + j, /* tile_idx */ 0, j);
            }
            reconfig_data_format_srca(cb_grad_attn_weights, cb_attention_weights);
            copy_init(cb_attention_weights, /*transpose=*/0);
            for (uint32_t j = 0; j < n; ++j) {
                copy_tile(cb_attention_weights, /* tile_idx */ c0 + j, /* register idx */ 2U + j);
            }
            mul_binary_tile_init();
            for (uint32_t j = 0; j < n; ++j) {
                mul_binary_tile(j, 2U + j, j);
            }
            binop_with_scalar_tile_init();
            for (uint32_t j = 0; j < n; ++j) {
                mul_unary_tile(j, scaler_bits);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_grad_attn_weights, cb_grad_scores);
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_grad_scores);
            }
            tile_regs_release();
        }
        cb_push_back(cb_grad_scores, kv_block);
        cb_pop_front(cb_attention_weights, kv_block);
        cb_pop_front(cb_grad_attn_weights, kv_block);

        // (D) dQ += dS[1 x cols] @ K[cols x qWt]: the block's columns accumulate in DST, the L1 accumulator
        //     is added to once per block (first block: plain write).
        cb_wait_front(cb_grad_scores, kv_block);
        pack_reconfig_data_format(cb_grad_scores, cb_grad_query_accum);
        const bool accumulate = (h0 > 0);
        if (!accumulate) {
            cb_reserve_back(cb_grad_query_accum, qk_tiles);
        } else {
            pack_reconfig_l1_acc(true);
        }
        // matmul in1 (K, bf16) is unpacked through SrcA, in0 (dS, fp32) through SrcB.
        reconfig_data_format(/* srca */ cb_key, /* srcb */ cb_grad_scores);
        matmul_init(cb_grad_scores, cb_key, /*transpose*/ 0);
        for (uint32_t t0 = 0; t0 < qk_tiles; t0 += block_size) {
            tile_regs_acquire();
            for (uint32_t b = 0; b < block_size; ++b) {
                for (uint32_t j = 0; j < cols; ++j) {
                    matmul_tiles(cb_grad_scores, cb_key, /* in0 */ j, /* in1 */ j * qk_tiles + t0 + b, /* dst */ b);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t b = 0; b < block_size; ++b) {
                pack_tile(b, cb_grad_query_accum);
            }
            tile_regs_release();
        }
        if (accumulate) {
            pack_reconfig_l1_acc(false);
            cb_pop_front(cb_grad_query_accum, qk_tiles);
            cb_reserve_back(cb_grad_query_accum, qk_tiles);
        }
        cb_push_back(cb_grad_query_accum, qk_tiles);
        cb_wait_front(cb_grad_query_accum, qk_tiles);
        cb_pop_front(cb_grad_scores, kv_block);

#ifdef CHAIN
        // The whole chunk goes: its rows past the diagonal are dropped with it.
        cb_pop_front(cb_key, kv_block * qk_tiles);
        cb_pop_front(cb_value, kv_block * v_tiles);
#else
        cb_pop_front(cb_key, cols * qk_tiles);
        cb_pop_front(cb_value, cols * v_tiles);
#endif
    }
#ifdef CHAIN
    // Chain mode streams every K/V chunk of the sequence through this core: drop the ones past the diagonal.
    for (uint32_t h = ((num_kv_tiles_to_process + kv_block - 1U) / kv_block) * kv_block; h < Ht; h += kv_block) {
        cb_wait_front(cb_key, kv_block * qk_tiles);
        cb_pop_front(cb_key, kv_block * qk_tiles);
        cb_wait_front(cb_value, kv_block * v_tiles);
        cb_pop_front(cb_value, kv_block * v_tiles);
    }
#endif

    pack_tiles_to_output(cb_grad_query_accum, cb_grad_query, qk_tiles);

#ifdef HAS_GATE
    cb_pop_front(cb_gated_grad_output_local, v_tiles);
    cb_pop_front(cb_gate, v_tiles);
#endif
    cb_pop_front(cb_u_scalar_row, onetile);
    cb_pop_front(cb_intermediates, num_of_interm_tiles);
    cb_pop_front(cb_query, qk_tiles);
    cb_pop_front(cb_attn_output, v_tiles);
    cb_pop_front(cb_grad_output, v_tiles);
}

void kernel_main() {
    // Runtime args
    // For standard mode: arg0 = start_row
    // For balanced mode: arg0 = start_pair_idx, arg1 = num_pairs
    // For work-list mode: arg0 = the item count (reused as `start_idx` below), args 1.. = the items
    const uint32_t start_idx = get_arg_val<uint32_t>(0);

#ifdef BALANCED_PARALLELISM
    const uint32_t num_pairs = get_arg_val<uint32_t>(1);
#endif

    compute_kernel_hw_startup(cb_grad_output, cb_query, cb_key);
    copy_init(cb_query);

    cb_wait_front(cb_mat_mul_reduction, onetile);
    // compute_kernel_hw_startup above does the one-time HW config; each matmul site below
    // re-establishes its state with reconfig_data_format + matmul_init.
    matmul_init(cb_query, cb_key);

#ifdef CAUSAL_MASK
    // Wait for causal mask tile ONCE - it's generated by writer and will be reused for every diagonal
    cb_wait_front(cb_attn_mask, onetile);
#endif

#ifdef WORK_LIST
    // Work-list mode: arg0 = number of rows of this core, args 1.. = their global row indices (heaviest first).
    // Chain mode: the same list padded with 0xFFFFFFFF rounds in which this core only consumes the K/V stream.
    for (uint32_t i = 0; i < start_idx; ++i) {
        const uint32_t row = get_arg_val<uint32_t>(1U + i);
#ifdef CHAIN
        if (row == 0xFFFFFFFFU) {
            for (uint32_t h = 0; h < Ht; h += kv_block) {
                cb_wait_front(cb_key, kv_block * qk_tiles);
                cb_pop_front(cb_key, kv_block * qk_tiles);
                cb_wait_front(cb_value, kv_block * v_tiles);
                cb_pop_front(cb_value, kv_block * v_tiles);
            }
            continue;
        }
#endif
        process_single_row(row);
    }
#elif defined(BALANCED_PARALLELISM)
    constexpr uint32_t pairs_per_seq = Ht / 2;

    for (uint32_t p = 0; p < num_pairs; ++p) {
        const uint32_t global_pair_idx = start_idx + p;

        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

        process_single_row(heavy_global_row);
        process_single_row(light_global_row);
    }
#else
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        const uint32_t global_row_idx = start_idx + row;
        process_single_row(global_row_idx);
    }
#endif

#ifdef CAUSAL_MASK
    // Pop the causal mask tile after all rows are processed (was reused for every diagonal)
    cb_pop_front(cb_attn_mask, onetile);
#endif
}
