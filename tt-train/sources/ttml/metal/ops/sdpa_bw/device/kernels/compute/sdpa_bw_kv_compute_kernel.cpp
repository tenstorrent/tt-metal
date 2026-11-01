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
#include "sdpa_bw_compute_utils.hpp"

// ----------------------------------------------------------------------
// SDPA Backward Compute Kernel for Key and Value Gradients (dK, dV)
// ----------------------------------------------------------------------
//
// Forward pass (for reference):
//   P = softmax(Q @ K^T / sqrt(d) + mask)    // attention weights [B, H, S, S]
//   O = P @ V                                 // output [B, H, S, D]
//
// Backward pass computes dK and dV given dO (upstream gradient):
//   dP = dO @ V^T                             // gradient w.r.t. attention weights
//   u  = rowsum(dO ⊙ O)                       // per-row scalar for softmax backward
//   dS = P ⊙ (dP - u)                         // softmax backward (element-wise)
//   dV = P^T @ dO                             // gradient w.r.t. value
//   dK = (1/sqrt(d)) * dS^T @ Q               // gradient w.r.t. key
//
// Note: We apply the scale factor inside dS computation for numerical stability.
//
// Grouped Query Attention (GQA):
//   Multiple query heads share the same K/V head. Gradients from all query heads
//   in a group are accumulated into their shared K/V head's gradient.
//
// Processing order:
//   for each K/V row k:
//     for each query head h in group:
//       for each query row q:
//         P[q,k] = softmax(Q[q] @ K[k]^T / sqrt(d) + mask[q,k])  // recomputed
//         dP[q,k] = dO[q] @ V[k]^T
//         dS[q,k] = P[q,k] * (dP[q,k] - u_scalar) * scale         // + P^T, dS^T in DST
//         dV[k] += P[q,k]^T @ dO[q]                               // accumulate
//         dK[k] += dS[q,k]^T @ Q[q]                               // accumulate
//
// P^T and dS^T are produced by in-DST 32-bit transpose_dest right after the dS
// elementwise chain (P is already resident in DST for the dS multiply), so neither
// transpose needs a pack->CB->unpack roundtrip. u_scalar = rowsum(dO ⊙ O) is
// precomputed by the Q kernel.
// ----------------------------------------------------------------------

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size_q = get_compile_time_arg_val(1);     // block size for Q/K matmuls (divides qWt)
constexpr uint32_t block_size_v = get_compile_time_arg_val(2);     // block size for V/dO matmuls (divides vWt)
constexpr uint32_t qWt = get_compile_time_arg_val(3);              // Q/K inner dim in tiles
constexpr uint32_t vWt = get_compile_time_arg_val(4);              // V/dO/O inner dim in tiles
constexpr uint32_t Ht = get_compile_time_arg_val(5);               // num_seq_len / TILE_H
constexpr uint32_t heads_per_group = get_compile_time_arg_val(6);  // number of heads per group
constexpr uint32_t scaler_bits = get_compile_time_arg_val(7);      // sqrt(Et) - sdpa scaler factor
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(8);   // used to transform mask from 1/0 to 0/-1
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(9);  // used to transform mask from 0/-1 to 0/-inf

constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;  // Gradient w.r.t. output
constexpr uint32_t cb_query = tt::CBIndex::c_2;        // Original query
constexpr uint32_t cb_key = tt::CBIndex::c_3;          // Original key
constexpr uint32_t cb_value = tt::CBIndex::c_4;        // Original value
#if defined(CAUSAL_MASK) || defined(USE_ATTN_MASK)
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;  // Original mask
#endif
constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;       // Forward pass intermediates
constexpr uint32_t cb_mat_mul_reduction = tt::CBIndex::c_7;   // Temporary computations
constexpr uint32_t cb_grad_value_accum = tt::CBIndex::c_8;    // L1 accumulator for grad_value
constexpr uint32_t cb_grad_key_accum = tt::CBIndex::c_9;      // L1 accumulator for grad_key
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_10;  // Recomputed attention weights = softmax(QK^T / sqrt(Et))
constexpr uint32_t cb_grad_attn_weights = tt::CBIndex::c_11;  // Gradient w.r.t. attention: dL/dP
constexpr uint32_t cb_grad_scores_transposed = tt::CBIndex::c_12;   // dS^T, packed directly from DST
constexpr uint32_t cb_attn_weights_transposed = tt::CBIndex::c_13;  // P^T, packed directly from DST
constexpr uint32_t cb_u_scalar_row = tt::CBIndex::c_14;             // u_scalar per row (precomputed by Q kernel)
constexpr uint32_t cb_grad_key = tt::CBIndex::c_15;                 // Output: grad_K
constexpr uint32_t cb_grad_value = tt::CBIndex::c_16;               // Output: grad_V

const uint32_t qk_tiles = qWt;            // Q/K inner dim tiles
const uint32_t v_tiles = vWt;             // V/dO/O inner dim tiles
const uint32_t num_of_interm_tiles = 1U;  // single FP32 logsumexp tile per Q row
// Query rows per block (iteration 28): the chain chunk under CHAIN (the reader pushes whole chunks), 1 otherwise.
constexpr uint32_t q_block = get_compile_time_arg_val(10);

// Broadcasts column 0 of `tile_idx` of `cb` (fp32 column vector: lse or u) into DST[dst] (B2D datacopy, no SrcA
// change). Must be called inside a tile_regs_acquire block.
FORCE_INLINE void bcast_col_into_dst(const uint32_t cb, const uint32_t tile_idx, const uint32_t dst) {
    reconfig_data_format_srcb(cb);
    UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(false, false, cb)));
    MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, DST_ACCUM_MODE, BroadcastType::COL>(cb)));
    unary_bcast<BroadcastType::COL>(cb, tile_idx, dst);
}

// P = exp(S - lse_row) for the m score tiles in DST[0 .. m) (one query row each, lse tiles first_tile ..),
// in place on the SFPU at full fp32.
FORCE_INLINE void exp_rows_on_dst(
    const uint32_t m, const uint32_t lse_reg, const uint32_t cb_lse, const uint32_t first_tile) {
    for (uint32_t i = 0; i < m; ++i) {
        bcast_col_into_dst(cb_lse, first_tile + i, lse_reg);
        sub_binary_tile_init();  // after the datacopy init, as apply_softmax_statistics_on_dst orders it
        sub_binary_tile(i, lse_reg, i);
    }
    sdpa_exp_tile_init();
    for (uint32_t i = 0; i < m; ++i) {
        sdpa_exp_tile(i);
    }
}

// acc[tiles] (+)= A^T[n rows]^T-block @ B[n rows x tiles]: the n fp32 transposed tiles of `cb_a` (slots 0 .. n)
// against the bf16 rows r0 .. r0 + n of `cb_b`, the block contracted in DEST, one L1 accumulate per block.
FORCE_INLINE void accumulate_block(
    const uint32_t cb_a,
    const uint32_t cb_b,
    const uint32_t cb_acc,
    const uint32_t tiles,
    const uint32_t block_size,
    const uint32_t r0,
    const uint32_t n,
    const bool do_accumulate) {
    pack_reconfig_data_format(cb_acc);
    if (!do_accumulate) {
        cb_reserve_back(cb_acc, tiles);
    } else {
        pack_reconfig_l1_acc(true);
    }
    // matmul: in1 (the bf16 rows) through SrcA, in0 (the fp32 transposed block) through SrcB.
    reconfig_data_format(/* srca */ cb_b, /* srcb */ cb_a);
    matmul_init(cb_a, cb_b, /* transpose */ 0);
    for (uint32_t t0 = 0; t0 < tiles; t0 += block_size) {
        tile_regs_acquire();
        for (uint32_t b = 0; b < block_size; ++b) {
            for (uint32_t i = 0; i < n; ++i) {
                matmul_tiles(cb_a, cb_b, /* in0 */ i, /* in1 */ (r0 + i) * tiles + t0 + b, /* dst */ b);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < block_size; ++b) {
            pack_tile(b, cb_acc);
        }
        tile_regs_release();
    }
    if (do_accumulate) {
        pack_reconfig_l1_acc(false);
        cb_pop_front(cb_acc, tiles);
        cb_reserve_back(cb_acc, tiles);
    }
    cb_push_back(cb_acc, tiles);
}

/**
 * One block of n query rows (rows r0 .. r0 + n of the resident chunk, global rows h0 + r0 ..) against the resident
 * K/V row (iteration 28):
 *   (A) P = exp(scale * Q K^T [+ mask] - lse), two rows per DEST pass, packed to cb_attention_weights;
 *   (B) per row: dP = dO V^T in DEST, dS = P (dP - u) scale on the SFPU, dS^T and P^T transposed in DEST and
 *       packed to the block CBs (the dP tile never leaves DEST);
 *   (C) dV += P^T @ dO and (D) dK += dS^T @ Q contracted over the block in DEST, one L1 accumulate each.
 */
FORCE_INLINE void process_block(
    const uint32_t h0, const uint32_t r0, const uint32_t n, const uint32_t k_row_tile, const bool do_accumulate) {
    // (A)
    cb_reserve_back(cb_attention_weights, q_block);
    for (uint32_t i0 = 0; i0 < n;) {
        // A row that takes a mask tile runs alone in its pass: mask_tile needs the mask in the register after the
        // data register (DST[1]), which a two-row pass holds the second row's scores in.
#ifdef CAUSAL_MASK
        const bool masked_pass = (h0 + r0 + i0 == k_row_tile);
#elif defined(USE_ATTN_MASK)
        const bool masked_pass = true;
#else
        const bool masked_pass = false;
#endif
        const uint32_t m = masked_pass ? 1U : ((n - i0 < 2U) ? (n - i0) : 2U);
        reconfig_data_format(cb_query, cb_key);
        matmul_init(cb_query, cb_key, /* transpose */ 1);
        tile_regs_acquire();
        for (uint32_t j = 0; j < m; ++j) {
            for (uint32_t f = 0; f < qk_tiles; ++f) {
                matmul_tiles(cb_query, cb_key, /* in0 */ (r0 + i0 + j) * qk_tiles + f, /* in1 */ f, /* dst */ j);
            }
        }
        for (uint32_t j = 0; j < m; ++j) {
#ifdef CAUSAL_MASK
            if (masked_pass) {
                apply_mask_on_reg_at(0, /* mask reg */ 1U, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
            } else {
                binop_with_scalar_tile_init();
                mul_unary_tile(j, scaler_bits);
            }
#elif defined(USE_ATTN_MASK)
            apply_mask_on_reg_at(0, /* mask reg */ 1U, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
            cb_pop_front(cb_attn_mask, onetile);
#else
            binop_with_scalar_tile_init();
            mul_unary_tile(j, scaler_bits);
#endif
        }
        exp_rows_on_dst(m, /* lse reg */ 2U, cb_intermediates, /* first lse tile */ r0 + i0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_attention_weights);
        for (uint32_t j = 0; j < m; ++j) {
            pack_tile(j, cb_attention_weights);
        }
        tile_regs_release();
        i0 += m;
    }
    cb_push_back(cb_attention_weights, q_block);
    cb_wait_front(cb_attention_weights, q_block);

    // (B)
    cb_reserve_back(cb_grad_scores_transposed, q_block);
    cb_reserve_back(cb_attn_weights_transposed, q_block);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t r = r0 + i;
        reconfig_data_format(cb_grad_output, cb_value);
        matmul_init(cb_grad_output, cb_value, /* transpose */ 1);
        tile_regs_acquire();
        for (uint32_t f = 0; f < v_tiles; ++f) {
            matmul_tiles(cb_grad_output, cb_value, /* in0 */ r * v_tiles + f, /* in1 */ f, /* dst */ 0);
        }
        bcast_col_into_dst(cb_u_scalar_row, r, /* dst */ 2U);
        sub_binary_tile_init();
        sub_binary_tile(0, 2U, 0);
        reconfig_data_format_srca(cb_grad_output, cb_attention_weights);
        copy_init(cb_attention_weights);
        copy_tile(cb_attention_weights, /* tile_idx */ i, /* register idx */ 1U);
        mul_binary_tile_init();
        mul_binary_tile(0, 1U, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scaler_bits);
        transpose_dest_init</* is_32bit */ true>(cb_attention_weights);
        transpose_dest</* is_32bit */ true>(0);   // dS -> dS^T
        transpose_dest</* is_32bit */ true>(1U);  // P  -> P^T
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_grad_scores_transposed);
        pack_tile(0, cb_grad_scores_transposed);
        pack_tile(1U, cb_attn_weights_transposed);
        tile_regs_release();
    }
    cb_push_back(cb_grad_scores_transposed, q_block);
    cb_push_back(cb_attn_weights_transposed, q_block);
    cb_pop_front(cb_attention_weights, q_block);
    cb_wait_front(cb_grad_scores_transposed, q_block);
    cb_wait_front(cb_attn_weights_transposed, q_block);

    // (C) dV += P^T @ dO
    accumulate_block(
        cb_attn_weights_transposed, cb_grad_output, cb_grad_value_accum, v_tiles, block_size_v, r0, n, do_accumulate);
    cb_wait_front(cb_grad_value_accum, v_tiles);
    // (D) dK += dS^T @ Q
    accumulate_block(
        cb_grad_scores_transposed, cb_query, cb_grad_key_accum, qk_tiles, block_size_q, r0, n, do_accumulate);
    cb_wait_front(cb_grad_key_accum, qk_tiles);

    cb_pop_front(cb_grad_scores_transposed, q_block);
    cb_pop_front(cb_attn_weights_transposed, q_block);
}

/**
 * Process one K/V row of the SDPA backward KV computation for `num_heads` query heads of its group (the
 * reader streams the heads' Q, dO, lse and u rows in order): accumulates dK and dV over them and packs
 * the result to the output CBs. The query rows come in blocks of q_block (iteration 28): the whole block is
 * resident, the rows above the diagonal are skipped inside it, the stream pops are whole blocks.
 *
 * @param global_row_idx The global row index (across all batches/groups/sequences)
 * @param num_heads Query heads accumulated into this output (heads_per_group, or 1 with HEAD_SPLIT)
 * @param stream_start Chain mode: the first query row the chain streams this round (iteration 27); otherwise the
 *        reader streams from the diagonal row.
 */
FORCE_INLINE void process_row_heads(uint32_t global_row_idx, uint32_t num_heads, uint32_t stream_start = 0) {
    cb_wait_front(cb_key, qk_tiles);
    cb_wait_front(cb_value, v_tiles);

    // The last PACK targets of the previous row are the BF16 outputs (pack_tiles_to_output); every PACK target
    // inside the block is Float32 and reconfigured unconditionally there.
#ifdef CAUSAL_MASK
    const uint32_t k_row_tile = global_row_idx % Ht;
    const uint32_t q_start_tile = k_row_tile;
#else
    const uint32_t k_row_tile = Ht;  // no diagonal row
    const uint32_t q_start_tile = 0;
#endif
#ifndef CHAIN
    stream_start = q_start_tile;  // the non-chain readers stream the rows from the diagonal on
#endif

    bool first_block = true;
    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (uint32_t h0 = stream_start; h0 < Ht; h0 += q_block) {
            cb_wait_front(cb_query, q_block * qk_tiles);
            cb_wait_front(cb_grad_output, q_block * v_tiles);
            cb_wait_front(cb_intermediates, q_block * num_of_interm_tiles);
            cb_wait_front(cb_u_scalar_row, q_block);
            if (h0 + q_block > q_start_tile) {  // the block has rows on or below the diagonal
                const uint32_t r0 = (h0 < q_start_tile) ? (q_start_tile - h0) : 0U;
                process_block(h0, r0, q_block - r0, k_row_tile, !first_block || head_idx > 0);
                first_block = false;
            }
            cb_pop_front(cb_query, q_block * qk_tiles);
            cb_pop_front(cb_grad_output, q_block * v_tiles);
            cb_pop_front(cb_intermediates, q_block * num_of_interm_tiles);
            cb_pop_front(cb_u_scalar_row, q_block);
        }
    }

    pack_tiles_to_output(cb_grad_value_accum, cb_grad_value, v_tiles);
    pack_tiles_to_output(cb_grad_key_accum, cb_grad_key, qk_tiles);

    cb_pop_front(cb_key, qk_tiles);
    cb_pop_front(cb_value, v_tiles);
}

FORCE_INLINE void process_single_row(uint32_t global_row_idx) {
    process_row_heads(global_row_idx, heads_per_group);
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
    cb_wait_front(cb_attn_mask, onetile);
#endif

#ifdef WORK_LIST
    // Work-list mode: arg0 = item count, args 1.. = items (heaviest first); an item is a K/V row (times the
    // heads of its group with HEAD_SPLIT, one head per item).
#ifdef HEAD_SPLIT
    constexpr uint32_t items_per_row = heads_per_group;
    constexpr uint32_t heads_per_item = 1U;
#else
    constexpr uint32_t items_per_row = 1U;
    constexpr uint32_t heads_per_item = heads_per_group;
#endif
    for (uint32_t i = 0; i < start_idx; ++i) {
        const uint32_t item = get_arg_val<uint32_t>(1U + i);
#ifdef CHAIN
        // Per round the first query row the chain streams (iteration 27).
        const uint32_t round_start = get_arg_val<uint32_t>(1U + start_idx + i);
        if (item == 0xFFFFFFFFU) {  // a round without an item: consume the stream
            for (uint32_t h = round_start; h < Ht; ++h) {
                cb_wait_front(cb_query, qk_tiles);
                cb_pop_front(cb_query, qk_tiles);
                cb_wait_front(cb_grad_output, v_tiles);
                cb_pop_front(cb_grad_output, v_tiles);
                cb_wait_front(cb_intermediates, num_of_interm_tiles);
                cb_pop_front(cb_intermediates, num_of_interm_tiles);
                cb_wait_front(cb_u_scalar_row, onetile);
                cb_pop_front(cb_u_scalar_row, onetile);
            }
            continue;
        }
        process_row_heads(item / items_per_row, heads_per_item, round_start);
#else
        process_row_heads(item / items_per_row, heads_per_item);
#endif
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
    cb_pop_front(cb_attn_mask, onetile);
#endif
}
