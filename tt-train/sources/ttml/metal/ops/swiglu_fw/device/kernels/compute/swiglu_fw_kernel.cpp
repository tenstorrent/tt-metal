// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <compute_kernel_api/eltwise_binary_sfpu.h>
#include <debug/dprint.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/ops/common/compute_utils.hpp"
namespace NAMESPACE {

// ----------------------------------------------------------------------
// Problem:
//
// Given:
//   X   shape [R, P]
//   W1  shape [P, K]
//   W2  shape [K, C]
// We want:
//   Y[r, c] = sum_k( M[r, k] * W2[k, c] )
//   where
//       M[r, k] = sum_p( X[r, p] * W1[p, k] )
//
// We compute in 3 nested block loops over c, k, p
//   p is split into p_blocks for streaming X and W1
//   k is split into k_blocks for streaming M and W2
//   c is split into c_blocks to fit Y accumulators in registers
// ----------------------------------------------------------------------

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);         // total C
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(3);  // total K

// Circular buffer indices
constexpr auto cb_input_idx = tt::CBIndex::c_0;      // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;         // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;         // W2[k_block, c_block]
constexpr auto cb_zero_idx = tt::CBIndex::c_3;       // Tile of zeros
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;        // M[r, k_block]
constexpr auto cb_y_idx = tt::CBIndex::c_7;          // Final Y[r, c_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_8;  // Partial Y[r, c_block] between k_blocks

constexpr uint32_t onetile = 1U;

// ============================================================================
// Phase A: Compute full M[r, k] over all p_s
//   M[r, k] = sum_p( X[r, p] * W1[p, k] )
// ============================================================================
inline void compute_full_M_for_k() {
    const uint32_t accum_reg = 0U;  // REG0 will hold the accumulated M[r, k]
    tile_regs_acquire();            // acquire working regs

    // M[r, k] accumulator is REG0
    // For each p_block:
    //   For each p in p_block:
    //      REG0 += matmul( X[r, p], W1[p, k] )
    for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
        const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

        cb_wait_front(cb_input_idx, block_size);
        cb_wait_front(cb_w1_idx, block_size);

        mm_init(cb_input_idx, cb_w1_idx, cb_xw1_idx);
        for (uint32_t block_idx = 0; block_idx < p_block_size; ++block_idx) {
            // TODO: Confirm if accumulation is guaranteed without explicit init and add.
            matmul_tiles(cb_input_idx, cb_w1_idx, block_idx, block_idx, accum_reg, false);
        }

        cb_pop_front(cb_input_idx, block_size);
        cb_pop_front(cb_w1_idx, block_size);
    }
    tile_regs_commit();

    // At this point, accum_reg (REG0) holds full M[r, k]
    pack_and_push(accum_reg, cb_xw1_idx);
}

// ============================================================================
// Phase B: Complete Y computation and store
//   Y[r,c] += sum_k( M[r,k] * W2[k,c] )
//   Stores result to output_cb_idx (either Y_partial or Y_final CB)
// ============================================================================
inline void mul_MxW2_accumulate_Y(uint32_t k_block_size, uint32_t c_block_size, bool first_k_block, bool last_k_block) {
    MATH(DPRINT << " First k_block=" << (int)first_k_block << ", last_k_block=" << (int)last_k_block << ENDL());
    tile_regs_acquire();
    // mm_init(cb_xw1_idx, cb_w2_idx, cb_y_idx);
    // Initialize or load Y accumulators
    if (!first_k_block) {
        cb_wait_front(cb_y_partial_idx, block_size);
        copy_tile_init(cb_y_partial_idx);
        for (uint32_t c = 0; c < c_block_size; ++c) {
            copy_tile(cb_y_partial_idx, c, c);  // CB tile -> REG
        }
        cb_pop_front(cb_y_partial_idx, block_size);
    }

    cb_wait_front(cb_xw1_idx, block_size);  // M[r,k] values ready
    // mm_init(cb_xw1_idx, cb_w2_idx, cb_y_idx);
    mm_init_short(cb_xw1_idx, cb_w2_idx, false);
    // Process each column of W2 sequentially
    for (uint32_t c = 0; c < c_block_size; ++c) {
        // Wait for W2 data: block_size tiles per column
        cb_wait_front(cb_w2_idx, block_size);

        // Compute Y[r, c] = sum_k( M[r,k] * W2[k, c] )
        for (uint32_t k = 0; k < k_block_size; ++k) {
            // TODO: Verify matmul_tiles overwrite vs. accumulate behavior.
            matmul_tiles(cb_xw1_idx, cb_w2_idx, k, k, c, false);
        }
        cb_pop_front(cb_w2_idx, block_size);  // Done with all W2 data
    }
    // Pop all and M[r,k] values at once
    cb_pop_front(cb_xw1_idx, block_size);  // Done with M[r,k] values

    tile_regs_commit();
    // Store result to appropriate CB
    const auto output_cb_idx = last_k_block ? cb_y_idx : cb_y_partial_idx;
    pack_and_push_block(output_cb_idx, block_size);
}

// ========================= Compute kernel structure =========================
// for r in rows:
//   for c_block in c_blocks:
//     for k_block in k_blocks:
//       for k in k_block:
//         compute M[r,k] = sum_p( X[r,p] * W1[p,k] )
//       for c in c_block:
//         Y_partial[r,c] += sum_k( M[r,k] * W2[k,c] )
//     store final Y_partial to Y
// ============================================================================
inline void MAIN {
    init_sfpu(cb_input_idx, cb_y_idx);
    binary_op_init_common(cb_input_idx, cb_w1_idx, cb_y_idx);

    // Wait for global constants to be ready before starting computation.
    cb_wait_front(cb_zero_idx, onetile);

    for (uint32_t r = 0; r < num_rows_per_core; ++r) {
        // Loop over c_blocks with tail handling
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;

            // Loop over k_blocks with tail handling
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                const bool first_k_block = (k_block_start == 0);
                const bool last_k_block = (k_block_start + block_size >= hidden_Wt);

                // ---- Phase A: compute M[r,k] for all k in this k_block ----
                // M[r,k] = sum_p( X[r,p] * W1[p,k] )
                for (uint32_t k = 0; k < block_size; ++k) {
                    if (k < k_block_size) {
                        compute_full_M_for_k();
                    } else {
                        // TODO: Do we need acquire/commit or can we push garbage directly?
                        // Push empty/invalid M for k >= k_block_size
                        pack_and_push(3U, cb_xw1_idx);
                    }
                }
                // ---- Phase B: accumulate into Y[r, c_block] and store ----
                mul_MxW2_accumulate_Y(k_block_size, c_block_size, first_k_block, last_k_block);
            }
        }
    }
}

}  // namespace NAMESPACE
