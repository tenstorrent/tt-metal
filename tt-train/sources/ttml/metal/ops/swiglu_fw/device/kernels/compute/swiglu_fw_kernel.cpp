// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reconfig_data_format.h"
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
// tt::CBIndex::c_4 and tt::CBIndex::c_5 are unused in this implementation.
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;        // (X @ W1)[r, k_block]
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;        // (X @ W3)[r, k_block]
constexpr auto cb_m_idx = tt::CBIndex::c_8;          // M[r, k_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;  // Partial Y[r, c_block] between k_blocks
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, c_block]

constexpr uint32_t onetile = 1U;

// ============================================================================
// Compute full M[r, k] over all p_s
//   M[r, k] = SiLU(sum_p( X[r, p] * W1[p, k] )) * sum_p( X[r, p] * W3[p, k] )
// ============================================================================
inline void compute_M_for_k() {
    const uint32_t xw1_accum_reg = 0U;  // REG0 will hold the accumulated (X @ W1)[r, k]
    const uint32_t xw3_accum_reg = 1U;  // REG1 will hold the accumulated (X @ W3)[r, k]
    const uint32_t silu_xw1_reg = 2U;   // REG2 will hold the SiLU(X @ W1)[r, k]
    const uint32_t m_reg = 3U;          // REG3 will hold M[r, k] = SiLU(X @ W1) * (X @ W3)

    tile_regs_acquire();  // acquire working regs

    reconfig_data_format(cb_input_idx, cb_w1_idx);
    mm_init(cb_input_idx, cb_w1_idx, cb_xw1_idx);

    // Accumulate XW1 and XW3 over all p_blocks
    for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
        const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

        cb_wait_front(cb_input_idx, block_size);
        cb_wait_front(cb_w1_idx, block_size);
        cb_wait_front(cb_w3_idx, block_size);

        for (uint32_t block_idx = 0; block_idx < p_block_size; ++block_idx) {
            // Compute XW1
            matmul_tiles(cb_input_idx, cb_w1_idx, block_idx, block_idx, xw1_accum_reg);
            // Compute XW3 using same X tile
            matmul_tiles(cb_input_idx, cb_w3_idx, block_idx, block_idx, xw3_accum_reg);
        }

        cb_pop_front(cb_input_idx, block_size);
        cb_pop_front(cb_w1_idx, block_size);
        cb_pop_front(cb_w3_idx, block_size);
    }
    // Copy xw1_accum_reg to silu_xw1_reg
    copy_dest_values_init();
    copy_dest_values(silu_xw1_reg, xw1_accum_reg);
    // Apply sigmoid activation to compute sigmoid(XW1)
    sigmoid_tile_init();
    sigmoid_tile(silu_xw1_reg);
    // Multiply XW1 * sigmoid(XW1) to get SiLU(XW1)
    mul_binary_tile_init();
    mul_binary_tile(xw1_accum_reg, silu_xw1_reg, silu_xw1_reg);
    // Final multiplication: M = SiLU(XW1) * XW3
    mul_binary_tile(silu_xw1_reg, xw3_accum_reg, m_reg);

    tile_regs_commit();
    pack_and_push(m_reg, cb_m_idx);
}

// ============================================================================
// Compute Y and accumulate or store
//   Y[r, c] += sum_k( M[r, k] * W2[k, c] )
//   Stores result to output_cb_idx (either Y_partial or Y_final CB)
// ============================================================================
inline void mul_MxW2_accumulate_Y(uint32_t k_block_size, uint32_t c_block_size, bool first_k_block, bool last_k_block) {
    tile_regs_acquire();
    // Initialize or load Y accumulators
    if (!first_k_block) {
        cb_wait_front(cb_y_partial_idx, block_size);
        reconfig_data_format(cb_y_partial_idx, cb_y_partial_idx);
        copy_tile_init(cb_y_partial_idx);
        for (uint32_t c = 0; c < c_block_size; ++c) {
            copy_tile(cb_y_partial_idx, c, c);  // CB tile -> REG
        }
        cb_pop_front(cb_y_partial_idx, block_size);
    }
    cb_wait_front(cb_m_idx, block_size);  // M[r, k] values ready
    reconfig_data_format(cb_m_idx, cb_w2_idx);
    // NOTE(maciek): Using mm_init_short since we do not want to zero registers. mm_init does zero them.
    mm_init_short(cb_m_idx, cb_w2_idx, false);
    // Process each column of W2 sequentially
    for (uint32_t c = 0; c < c_block_size; ++c) {
        // Wait for W2 data: block_size tiles per column
        cb_wait_front(cb_w2_idx, block_size);

        // Compute Y[r, c] = sum_k( M[r, k] * W2[k, c] )
        for (uint32_t k = 0; k < k_block_size; ++k) {
            matmul_tiles(cb_m_idx, cb_w2_idx, k, k, c);
        }
        cb_pop_front(cb_w2_idx, block_size);  // Done with all W2 data
    }
    // Pop all and M[r, k] values at once
    cb_pop_front(cb_m_idx, block_size);  // Done with M[r, k] values

    tile_regs_commit();
    // Store result to appropriate CB
    const auto output_cb_idx = last_k_block ? cb_y_idx : cb_y_partial_idx;
    pack_and_push_block(output_cb_idx, block_size);
}

// ========================= Compute kernel structure =========================
// for r in rows:
//   for c_block in c_blocks:
//     for k_block in k_blocks:
//       # Phase A: compute M[r, k] for all k in this k_block
//       for k in k_block:
//         XW1 = sum_p( X[r, p] * W1[p, k] )
//         XW3 = sum_p( X[r, p] * W3[p, k] )
//         M[r, k] = SiLU(XW1) * XW3
//
//       # Phase B: accumulate into Y[r, c_block] and store
//       for c in c_block:
//         Y_partial[r, c] += sum_k( M[r, k] * W2[k, c] )
//     store Y_partial[r,c] → Y[r,c]
// ============================================================================
inline void MAIN {
    init_sfpu(cb_input_idx, cb_y_idx);
    binary_op_init_common(cb_input_idx, cb_w1_idx, cb_y_idx);
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

                // ---- Phase A: compute M[r, k] for all k in this k_block ----
                // M[r, k] = sum_p( X[r, p] * W1[p, k] )
                for (uint32_t k = 0; k < k_block_size; ++k) {
                    compute_M_for_k();
                }
                if (k_block_size != block_size) {
                    // Push empty/invalid Ms for k >= k_block_size
                    tile_regs_acquire();
                    tile_regs_commit();
                    pack_and_push_block(cb_m_idx, block_size - k_block_size);
                }
                // ---- Phase B: accumulate into Y[r, c_block] and store ----
                mul_MxW2_accumulate_Y(k_block_size, c_block_size, first_k_block, last_k_block);
            }
        }
    }
}

}  // namespace NAMESPACE
