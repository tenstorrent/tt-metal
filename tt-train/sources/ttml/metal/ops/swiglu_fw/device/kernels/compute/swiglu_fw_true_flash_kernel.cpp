// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// TRUE FLASH SwiGLU COMPUTE KERNEL
//
// This kernel implements the "True Flash" optimization for SwiGLU:
//   Y = (SiLU(X @ W1) * (X @ W3)) @ W2
//
// Key insight: Instead of materializing the full M row (hidden_dim tiles),
// we compute M tiles on-demand for each k_block.
//
// Algorithm (loop inversion from original):
//   for r in rows:
//     for k_block in K_blocks:                    # OUTER - compute M one k_block at a time
//       XW1_partial = 0, XW3_partial = 0
//       for p_block in P_blocks:                  # INNER - accumulate across embed_dim
//         XW1_partial += X[r, p_block] @ W1[p_block, k_block]
//         XW3_partial += X[r, p_block] @ W3[p_block, k_block]
//       M_partial = SiLU(XW1_partial) * XW3_partial   # Only block_size tiles!
//       for c_block in C_blocks:
//         Y_partial[c_block] += M_partial @ W2[k_block, c_block]
//     store Y[r, :]
//
// Memory savings:
//   - XW1_partial: block_size tiles (vs full hidden_Wt)
//   - XW3_partial: block_size tiles (vs full hidden_Wt)
//   - M_partial: block_size tiles (vs full hidden_Wt)
//   - Y_partial: full Wt tiles (need to accumulate all output columns)
//
// Trade-off: X is read K_blocks × P_blocks times per row (8× more for NanoLlama3).
// This is mitigated by X caching in Phase 3 of the implementation.
// ============================================================================

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
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t max_rows_for_sync = get_compile_time_arg_val(1);  // Loop iterations for multicast sync
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);         // embed_dim / 32 (output width)
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(4);  // hidden_dim / 32

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k_block]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k_block]
// CBs with intermediate computations (TRUE FLASH: only block_size tiles!)
constexpr auto cb_xw1_partial_idx = tt::CBIndex::c_4;  // Partial (X @ W1)[r, k_block] - block_size tiles
constexpr auto cb_xw3_partial_idx = tt::CBIndex::c_5;  // Partial (X @ W3)[r, k_block] - block_size tiles
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;          // Final (X @ W1)[r, k_block] after p-sum - block_size tiles
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;          // Final (X @ W3)[r, k_block] after p-sum - block_size tiles
constexpr auto cb_m_idx = tt::CBIndex::c_8;            // M[r, k_block] = SiLU(XW1) * XW3 - block_size tiles
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;    // Partial Y[r, :] across k_blocks - FULL Wt tiles!
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, :] - FULL Wt tiles

constexpr uint32_t onetile = 1U;

// ============================================================================
// Accumulate XW result for one k_block across p_blocks
// This is called for each (k_block, p_block) pair to build up the full p-sum
// for the current k_block before applying SiLU.
//
// Phase 2 Step 1: Uses matmul_block with ct_dim=block_size to process
// all output columns (k_block_size) in one matmul_block call per inner tile.
// This reduces the number of matmul calls from p×k to just p.
// ============================================================================
inline void accumulate_XW_for_k_block(
    const tt::CBIndex cb_x_idx,
    const tt::CBIndex cb_w_idx,
    const tt::CBIndex cb_partial_idx,
    const tt::CBIndex cb_final_idx,
    const uint32_t p_block_size,
    const uint32_t k_block_size,
    const bool first_p_block,
    const bool last_p_block) {
    tile_regs_acquire();

    // Load previous partial results if not first p_block
    if (!first_p_block) {
        cb_wait_front(cb_partial_idx, block_size);
        copy_tile_init(cb_partial_idx);
        for (uint32_t k = 0; k < k_block_size; ++k) {
            copy_tile(cb_partial_idx, k, k);
        }
        cb_pop_front(cb_partial_idx, block_size);
    }

    // Wait for W tiles (batched: block_size rows × block_size tiles)
    // W layout in CB: row-major [p0_k0, p0_k1, p0_k2, p0_k3, p1_k0, p1_k1, ...]
    constexpr uint32_t tiles_per_batch = block_size * block_size;
    cb_wait_front(cb_w_idx, tiles_per_batch);

    // Initialize matmul_block: ct_dim=block_size (all k output cols), rt_dim=1 (single row)
    // kt_dim=p_block_size tells hardware the inner dimension for striding
    mm_block_init_short(
        cb_x_idx,
        cb_w_idx,
        /*transpose=*/false,
        /*ct_dim=*/block_size,
        /*rt_dim=*/1,
        /*kt_dim=*/p_block_size);

    // Accumulate using matmul_block: one call per inner dimension tile
    // X: single row, tiles at indices 0, 1, 2, ... (p_block_size tiles)
    // W: p_block_size rows × block_size cols, row-major [p0_k0..p0_k3, p1_k0..p1_k3, ...]
    // For each inner tile p, W row p starts at index p*block_size
    uint32_t in0_index = 0;  // X tile index (increments by 1)
    uint32_t in1_index = 0;  // W tile index (increments by block_size per row)
    for (uint32_t p = 0; p < p_block_size; ++p) {
        matmul_block(
            cb_x_idx,
            cb_w_idx,
            in0_index,
            in1_index,
            /*dst_index=*/0,
            /*transpose=*/false,
            /*ct_dim=*/block_size,
            /*rt_dim=*/1,
            /*kt_dim=*/p_block_size);
        in0_index++;              // next X tile
        in1_index += block_size;  // next W row (row-major stride)
    }

    cb_pop_front(cb_w_idx, tiles_per_batch);

    tile_regs_commit();

    // Store to partial or final CB
    const auto output_cb_idx = last_p_block ? cb_final_idx : cb_partial_idx;
    pack_and_push_block(output_cb_idx, block_size);
}

// ============================================================================
// Compute M_partial = SiLU(XW1_partial) * XW3_partial for one k_block
// Both XW1 and XW3 have been fully accumulated (complete p-sum) at this point.
// ============================================================================
inline void compute_M_partial_for_k_block(const uint32_t k_block_size) {
    // Wait for XW1 and XW3 final results for this k_block
    cb_wait_front(cb_xw1_idx, block_size);
    cb_wait_front(cb_xw3_idx, block_size);

    for (uint32_t k = 0; k < k_block_size; ++k) {
        constexpr uint32_t xw1_reg = 0U;
        constexpr uint32_t xw3_reg = 1U;
        constexpr uint32_t silu_reg = 2U;
        constexpr uint32_t m_reg = 3U;

        tile_regs_acquire();

        // Load XW1[k] and XW3[k]
        copy_tile_init(cb_xw1_idx);
        copy_tile(cb_xw1_idx, k, xw1_reg);
        copy_tile_init(cb_xw3_idx);
        copy_tile(cb_xw3_idx, k, xw3_reg);

        // SiLU(XW1): sigmoid(XW1) * XW1
        copy_dest_values_init();
        copy_dest_values(xw1_reg, silu_reg);
        sigmoid_tile_init();
        sigmoid_tile(silu_reg);
        mul_binary_tile_init();
        mul_binary_tile(xw1_reg, silu_reg, silu_reg);

        // M = SiLU(XW1) * XW3
        mul_binary_tile(silu_reg, xw3_reg, m_reg);

        tile_regs_commit();
        pack_and_push(m_reg, cb_m_idx);
    }

    // Push padding tiles if k_block_size < block_size
    if (k_block_size != block_size) {
        tile_regs_acquire();
        tile_regs_commit();
        pack_and_push_block(cb_m_idx, block_size - k_block_size);
    }

    cb_pop_front(cb_xw1_idx, block_size);
    cb_pop_front(cb_xw3_idx, block_size);
}

// ============================================================================
// Accumulate Y_partial += M_partial @ W2 for one c_block within a k_block
// Y_partial accumulates across all k_blocks for each c_block position.
//
// CB strategy for Y accumulation:
// - Y_partial CB holds Wt tiles (full output row)
// - For first k_block: just compute and push c_block tiles
// - For subsequent k_blocks: load previous Y[c_block], add, push back
// - On last k_block: output goes to final Y CB
// ============================================================================
inline void accumulate_Y_for_c_block(
    const uint32_t k_block_size,
    const uint32_t c_block_size,
    const uint32_t c_block_offset,  // Starting tile position for this c_block
    const bool first_k_block,
    const bool last_k_block) {
    // Wait for W2 batch (block_size cols × block_size rows = block_size² tiles)
    // W2 layout in CB: column-major within batch [c0_k0, c0_k1, ..., c0_k3, c1_k0, ...]
    constexpr uint32_t tiles_per_batch = block_size * block_size;
    cb_wait_front(cb_w2_idx, tiles_per_batch);

    // Process each output tile in this c_block
    for (uint32_t c = 0; c < c_block_size; ++c) {
        tile_regs_acquire();

        // Load previous Y_partial[c_block_offset + c] if not first k_block
        if (!first_k_block) {
            cb_wait_front(cb_y_partial_idx, 1);
            copy_tile_init(cb_y_partial_idx);
            copy_tile(cb_y_partial_idx, 0, 0);  // Load single tile into reg 0
            cb_pop_front(cb_y_partial_idx, 1);
        }

        // Y[c] += sum_k M[k] * W2[k, c]
        mm_init_short(cb_m_idx, cb_w2_idx, false);
        const uint32_t cb_col_offset = c * block_size;
        for (uint32_t k = 0; k < k_block_size; ++k) {
            matmul_tiles(cb_m_idx, cb_w2_idx, k, cb_col_offset + k, 0);  // Accumulate into reg 0
        }

        tile_regs_commit();

        // Store result to appropriate CB
        const auto output_cb_idx = last_k_block ? cb_y_idx : cb_y_partial_idx;
        pack_and_push(0, output_cb_idx);
    }

    cb_pop_front(cb_w2_idx, tiles_per_batch);
}

// ============================================================================
// TRUE FLASH MAIN KERNEL
// ============================================================================
void kernel_main() {
    init_sfpu(cb_input_idx, cb_y_idx);
    binary_op_init_common(cb_input_idx, cb_w1_idx, cb_y_idx);

    // Calculate number of blocks
    const uint32_t num_k_blocks = (hidden_Wt + block_size - 1) / block_size;
    const uint32_t num_p_blocks = (Wt + block_size - 1) / block_size;
    const uint32_t num_c_blocks = (Wt + block_size - 1) / block_size;

    for (uint32_t r = 0; r < max_rows_for_sync; ++r) {
        const bool is_padding_row = (r >= num_rows_per_core);

        // ======== TRUE FLASH: k_block OUTER, p_block INNER ========
        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            const uint32_t k_block_start = k_block_idx * block_size;
            const uint32_t k_block_size =
                (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;
            const bool first_k_block = (k_block_idx == 0);
            const bool last_k_block = (k_block_idx == num_k_blocks - 1);

            // ---- Phase A: Compute XW1[k_block] and XW3[k_block] (full p-sum) ----
            for (uint32_t p_block_idx = 0; p_block_idx < num_p_blocks; ++p_block_idx) {
                const uint32_t p_block_start = p_block_idx * block_size;
                const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;
                const bool first_p_block = (p_block_idx == 0);
                const bool last_p_block = (p_block_idx == num_p_blocks - 1);

                // Read X[r, p_block] (pushed by dataflow kernel)
                cb_wait_front(cb_input_idx, block_size);

                // Accumulate XW1 for this (p_block, k_block)
                accumulate_XW_for_k_block(
                    cb_input_idx,
                    cb_w1_idx,
                    cb_xw1_partial_idx,
                    cb_xw1_idx,
                    p_block_size,
                    k_block_size,
                    first_p_block,
                    last_p_block);

                // Accumulate XW3 for this (p_block, k_block)
                accumulate_XW_for_k_block(
                    cb_input_idx,
                    cb_w3_idx,
                    cb_xw3_partial_idx,
                    cb_xw3_idx,
                    p_block_size,
                    k_block_size,
                    first_p_block,
                    last_p_block);

                cb_pop_front(cb_input_idx, block_size);
            }

            // ---- Phase B: Compute M_partial = SiLU(XW1) * XW3 ----
            compute_M_partial_for_k_block(k_block_size);

            // ---- Phase C: Accumulate Y_partial += M_partial @ W2 for ALL c_blocks ----
            cb_wait_front(cb_m_idx, block_size);  // M_partial ready

            for (uint32_t c_block_idx = 0; c_block_idx < num_c_blocks; ++c_block_idx) {
                const uint32_t c_block_start = c_block_idx * block_size;
                const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;

                if (is_padding_row) {
                    // PADDING ROW: Consume W2 to stay in sync
                    constexpr uint32_t tiles_per_batch = block_size * block_size;
                    cb_wait_front(cb_w2_idx, tiles_per_batch);
                    cb_pop_front(cb_w2_idx, tiles_per_batch);
                } else {
                    // ACTUAL ROW: Accumulate Y
                    accumulate_Y_for_c_block(
                        k_block_size,
                        c_block_size,
                        c_block_start,  // Starting tile position
                        first_k_block,
                        last_k_block);
                }
            }

            cb_pop_front(cb_m_idx, block_size);  // Done with M_partial
        }

        // Y[r, :] is now complete in cb_y_idx (writer will consume it)
    }
}
