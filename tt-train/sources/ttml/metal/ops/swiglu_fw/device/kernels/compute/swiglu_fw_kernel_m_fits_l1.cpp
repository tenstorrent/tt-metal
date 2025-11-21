// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <compute_kernel_api/eltwise_binary_sfpu.h>
#include <compute_kernel_api/reconfig_data_format.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
namespace NAMESPACE {

// ----------------------------------------------------------------------
// Problem:
//
// Given:
//   X   shape [R, P]
//   W1  shape [P, K]
//   W2  shape [K, C]
//   W3  shape [P, K]
// We want:
//   Y[r, c] = sum_k( M[r, k] * W2[k, c] )
//   where
//       M[r, k] = sum_p( X[r, p] * W1[p, k] ) * sum_p( X[r, p] * W3[p, k] )
//
// We compute in 3 nested block loops over c, k, p
//   p is split into p_blocks for streaming X, W1 and W3
//   k is split into k_blocks for streaming M and W2
//   c is split into c_blocks to fit Y accumulators in registers
// ----------------------------------------------------------------------

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);         // total C
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(3);  // total K

const uint32_t hidden_Wt_rounded_up =
    ((hidden_Wt + block_size - 1) / block_size) * block_size;  // Round up to nearest block_size

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k]
// CBs with intermediate computations
constexpr auto cb_xw1_partial_idx = tt::CBIndex::c_4;  // Partial (X @ W1)[r, k_block] between p_blocks
constexpr auto cb_xw3_partial_idx = tt::CBIndex::c_5;  // Partial (X @ W3)[r, k_block] between p_blocks
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;          // (X @ W1)[r, k_block]
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;          // (X @ W3)[r, k_block]
constexpr auto cb_m_idx = tt::CBIndex::c_8;            // M[r, k_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;    // Partial Y[r, c_block] between k_blocks
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, c_block]

constexpr uint32_t onetile = 1U;

// ============================================================================
// Abstracted operation to compute
// C[r, c : c + c_block_size] = A[r, k : k + ab_block_size] * B[k : k + ab_block_size, c : c + c_block_size]
//
//   if first_ab_block is true C := ...
//   else C += ...
//   if last_ab_block is true, store C to cb_c_final_idx
//   else store C to cb_c_partial_idx
//
// NOTE: This function does not wait for nor pop cb A. It only waits for and pops B. The caller is responsible for
// waiting for and popping A.
// ============================================================================
inline void mul_AxB_accumulate_C(
    tt::CBIndex cb_a_idx,
    tt::CBIndex cb_b_idx,
    tt::CBIndex cb_c_partial_idx,
    tt::CBIndex cb_c_final_idx,
    uint32_t a_start_idx,
    uint32_t ab_block_size,
    uint32_t c_block_size,
    bool first_ab_block,
    bool last_ab_block) {
    tile_regs_acquire();

    // Initialize or load C accumulators
    if (!first_ab_block) {
        cb_wait_front(cb_c_partial_idx, block_size);
        copy_tile_init(cb_c_partial_idx);
        for (uint32_t c = 0; c < c_block_size; ++c) {
            copy_tile(cb_c_partial_idx, c, c);  // CB tile -> REG
        }
        cb_pop_front(cb_c_partial_idx, block_size);
    }

    mm_init_short(cb_a_idx, cb_b_idx, false);

    // Process each column of B sequentially
    for (uint32_t c = 0; c < c_block_size; ++c) {
        // Wait for B data: block_size tiles per column
        cb_wait_front(cb_b_idx, block_size);

        // Compute C[r, c] += sum_k( A[r, k] * B[k, c] )
        for (uint32_t k = 0; k < ab_block_size; ++k) {
            matmul_tiles(cb_a_idx, cb_b_idx, a_start_idx + k, k, c, false);
        }

        cb_pop_front(cb_b_idx, block_size);  // Done with all B data
    }

    tile_regs_commit();

    // Store result to appropriate CB
    const auto output_cb_idx = last_ab_block ? cb_c_final_idx : cb_c_partial_idx;
    pack_and_push_block(output_cb_idx, block_size);
}

// ============================================================================
// Flash-attention optimization: Accumulate XW results across k_block batches
// This function accumulates X[r, p_block] * W[p_block, k_block] into partial/final buffers
// avoiding re-reading X for each k_block
// ============================================================================
inline void mul_XW_accumulate_k_block(
    tt::CBIndex cb_x_idx,        // X[r, p_block]
    tt::CBIndex cb_w_idx,        // W[p_block, k_block] (W1 or W3)
    tt::CBIndex cb_partial_idx,  // Partial results buffer (cb_xw1_partial or cb_xw3_partial)
    tt::CBIndex cb_final_idx,    // Final results buffer (cb_xw1 or cb_xw3)
    uint32_t p_block_size,
    uint32_t k_block_size,
    bool first_p_block,
    bool last_p_block) {
    tile_regs_acquire();

    // Initialize or load previous partial results for this k_block
    if (!first_p_block) {
        cb_wait_front(cb_partial_idx, block_size);

        copy_tile_init(cb_partial_idx);
        // Only load k_block_size values (actual data), ignore padding
        for (uint32_t k = 0; k < k_block_size; ++k) {
            copy_tile(cb_partial_idx, k, k);  // CB tile -> REG
        }

        cb_pop_front(cb_partial_idx, block_size);
    }

    mm_init_short(cb_x_idx, cb_w_idx, false);

    // Process each p in this p_block (to match reader's organization)
    for (uint32_t p = 0; p < p_block_size; ++p) {
        // Wait for W data: W[p, k_block] (entire row, matching reader pattern)
        cb_wait_front(cb_w_idx, block_size);

        // Accumulate: result[k] += X[p] * W[p, k] for all k in k_block
        for (uint32_t k = 0; k < k_block_size; ++k) {
            matmul_tiles(cb_x_idx, cb_w_idx, p, k, k, false);
        }

        cb_pop_front(cb_w_idx, block_size);
    }

    tile_regs_commit();

    // Store result to appropriate CB
    const auto output_cb_idx = last_p_block ? cb_final_idx : cb_partial_idx;
    pack_and_push_block(output_cb_idx, block_size);
}

// ============================================================================
// Compute XW1[r, k] and XW3[r, k] over all p_blocks using flash-attention optimization
//   XW1[r, k] = sum_p( X[r, p] * W1[p, k] )
//   XW3[r, k] = sum_p( X[r, p] * W3[p, k] )
//
// This function implements a flash-attention-like optimization where X[r, p_block] is read
// only once per p_block and reused across all k_blocks, significantly reducing memory reads.
// ============================================================================

inline void compute_XW1_XW3_for_r() {
    // Flash-attention optimization: Read X[r, p_block] only once per p_block
    // Loop order: outer p_blocks, inner k_blocks (but process k_blocks one at a time due to register limits)
    for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
        const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;
        const bool first_p_block = (p_block_start == 0);
        const bool last_p_block = (p_block_start + block_size >= Wt);

        // Read X[r, p_block] once for this p_block
        cb_wait_front(cb_input_idx, block_size);

        // Process each k_block separately (due to register constraints)
        for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
            const uint32_t k_block_size =
                (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

            // Accumulate contributions to XW1[r, k_block] and XW3[r, k_block]
            // On last_p_block, results go directly to final CBs
            mul_XW_accumulate_k_block(
                /* cb_x_idx */ cb_input_idx,
                /* cb_w_idx */ cb_w1_idx,
                /* cb_partial_idx */ cb_xw1_partial_idx,
                /* cb_final_idx */ cb_xw1_idx,
                /* p_block_size */ p_block_size,
                /* k_block_size */ k_block_size,
                /* first_p_block */ first_p_block,
                /* last_p_block */ last_p_block);

            mul_XW_accumulate_k_block(
                /* cb_x_idx */ cb_input_idx,
                /* cb_w_idx */ cb_w3_idx,
                /* cb_partial_idx */ cb_xw3_partial_idx,
                /* cb_final_idx */ cb_xw3_idx,
                /* p_block_size */ p_block_size,
                /* k_block_size */ k_block_size,
                /* first_p_block */ first_p_block,
                /* last_p_block */ last_p_block);
        }

        // Done with X[r, p_block]
        cb_pop_front(cb_input_idx, block_size);
    }
}

// ============================================================================
// Compute M[r, :] = SiLU(XW1[r, :]) * XW3[r, :]
//   where XW1 and XW3 are precomputed and stored in cb_xw1_idx and cb_xw3_idx
// ============================================================================
inline void compute_M_for_r() {
    // Wait for XW1 and XW3 to be ready
    cb_wait_front(cb_xw1_idx, hidden_Wt_rounded_up);
    cb_wait_front(cb_xw3_idx, hidden_Wt_rounded_up);

    for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
        const uint32_t k_block_size =
            (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;
        // Compute M[r, k_block_start : k_block_start + k_block_size]
        // NOTE(maciek): We process each k in the block sequentially since we need to apply SiLU activation
        // and we have limited number of registers. Processing all k in the block in parallel would require
        // additional temporary CBs and packing/unpacking. I don't expect this to be a performance bottleneck.
        for (uint32_t k = 0; k < k_block_size; ++k) {
            const uint32_t xw1_reg = 0U;   // REG0 will hold (X @ W1)[r, k]
            const uint32_t xw3_reg = 1U;   // REG1 will hold (X @ W3)[r, k]
            const uint32_t silu_reg = 2U;  // REG2 will hold SiLU(X @ W1)[r, k]
            const uint32_t m_reg = 3U;     // REG3 will hold M

            const uint32_t tile_offset = k_block_start + k;

            tile_regs_acquire();  // acquire working regs
            // Copy XW1 and XW3 to registers
            copy_tile_init(cb_xw1_idx);
            copy_tile(cb_xw1_idx, tile_offset, xw1_reg);
            copy_tile_init(cb_xw3_idx);
            copy_tile(cb_xw3_idx, tile_offset, xw3_reg);

            // Apply SiLU activation to compute SiLU(XW1)
            copy_dest_values_init();
            copy_dest_values(silu_reg, xw1_reg);
            sigmoid_tile_init();
            sigmoid_tile(silu_reg);
            // Multiply XW1 * sigmoid(XW1) to get SiLU(XW1)
            mul_binary_tile_init();
            mul_binary_tile(xw1_reg, silu_reg, silu_reg);
            // Final multiplication: M = SiLU(XW1) * XW3
            mul_binary_tile(silu_reg, xw3_reg, m_reg);

            tile_regs_commit();
            pack_and_push(m_reg, cb_m_idx);
        }
        if (k_block_size != block_size) {
            // Push empty/invalid Ms for k >= k_block_size
            tile_regs_acquire();
            tile_regs_commit();
            pack_and_push_block(cb_m_idx, block_size - k_block_size);
        }
    }

    cb_pop_front(cb_xw1_idx, hidden_Wt_rounded_up);
    cb_pop_front(cb_xw3_idx, hidden_Wt_rounded_up);
}

// ================= Compute kernel structure (M fits in L1) ==================
// FLASH-ATTENTION OPTIMIZATION: Read X only once per p_block!
// for r in rows:
//     # Phase A: Compute XW1[r, :] and XW3[r, :] - Flash-attention trick
//     for p_block in p_blocks:                         # OUTER LOOP - read X once per p_block
//         read X[r, p_block]
//         for k_block in k_blocks:                     # INNER LOOP - reuse X for all k_blocks
//             # Process W1 first for all p in p_block
//             XW1_partial[r, k_block] += X[r, p_block] * W1[p_block, k_block]
//             # Process W3 second for all p in p_block
//             XW3_partial[r, k_block] += X[r, p_block] * W3[p_block, k_block]
//         # After last p_block: XW1_partial[r, k_block] → XW1[r, k_block]
//
//     # Phase B: Compute M[r,:] once
//     for k_block in k_blocks:                         # iterate over hidden dimension
//         for k in k_block:
//             M[r, k] = SiLU( XW1[r, k] ) * XW3[r, k]
//
//     # Phase C: Use M[r, :] for all c-blocks to compute Y[r, :]
//     for c_block in c_blocks:                         # iterate over output dimension
//         for k_block in k_blocks:                     # iterate over hidden dimension
//             Y_partial[r, c] += M[r, k] * W2[k, c]
//         store Y_partial[r, c] → Y[r, c]
// ============================================================================
inline void MAIN {
    init_sfpu(cb_input_idx, cb_y_idx);
    binary_op_init_common(cb_input_idx, cb_w1_idx, cb_y_idx);

    for (uint32_t r = 0; r < num_rows_per_core; ++r) {
        // ---- Phase A: Accumulate XW1[r,:] and XW3[r,:] in tiles over p ----
        // XW1[r,k] = sum_p( X[r,p] * W1[p,k] )
        // XW3[r,k] = sum_p( X[r,p] * W3[p,k] )
        compute_XW1_XW3_for_r();

        // ---- Phase B: Compute M[r,:] once ----
        compute_M_for_r();
        cb_wait_front(cb_m_idx, hidden_Wt_rounded_up);  // M[r, :] ready

        // ---- Phase C: Use M[r,:] for all c_blocks ----
        // Y[r, :] = sum_k( M[r,k] * W2[k,c] )
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;
            // Compute Y[r, c_block_start : c_block_start + c_block_size]
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;
                const bool first_k_block = (k_block_start == 0);
                const bool last_k_block = (k_block_start + block_size >= hidden_Wt);
                mul_AxB_accumulate_C(
                    /* cb_a_idx */ cb_m_idx,
                    /* cb_b_idx */ cb_w2_idx,
                    /* cb_c_partial_idx */ cb_y_partial_idx,
                    /* cb_c_final_idx */ cb_y_idx,
                    /* a_start_idx */ k_block_start,
                    /* ab_block_size */ k_block_size,
                    /* c_block_size */ c_block_size,
                    /* first_ab_block */ first_k_block,
                    /* last_ab_block */ last_k_block);
                // TODO(maciek): consider double buffering row of M. This would require additional space in the CB.
                // and maybe some smarter usage of cb_m_idx.
            }
        }

        // M[r, :] is no longer needed
        cb_pop_front(cb_m_idx, hidden_Wt_rounded_up);
    }
}

}  // namespace NAMESPACE
