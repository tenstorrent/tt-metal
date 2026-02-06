// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// TRUE FLASH SwiGLU COMPUTE KERNEL with rt_dim=2 and X Row Caching (Phase 3)
//
// This kernel processes 2 rows at a time using matmul_block with rt_dim=2.
// This doubles throughput by computing both rows in a single matmul operation.
//
// Phase 3 optimization: X Row Caching
//   - Full X row(s) are cached in CB at the start of each row pair
//   - X is reused across all k_blocks (K_blocks × fewer DRAM reads)
//   - CB holds rt_dim × Wt tiles = 2 × Wt tiles
//
// Key optimization: matmul_block(rt_dim=2, ct_dim=2)
//   - DST layout: [row0_k0, row0_k1, row1_k0, row1_k1] (4 tiles)
//   - Process k-dimension in 2 chunks: k=0,1 then k=2,3
//
// Algorithm with X caching:
//   for r in 0..rows step 2:           # Process row pairs
//     wait for X[r:r+2, :] (full row cached in CB)
//     for k_block in K_blocks:
//       for p_block in P_blocks:
//         # Read X[r:r+2, p_block] from CACHED CB (not DRAM!)
//         matmul_block(X[r:r+2, p], W1[p, k=0:2], rt_dim=2, ct_dim=2)
//         matmul_block(X[r:r+2, p], W1[p, k=2:4], rt_dim=2, ct_dim=2)
//       M[r:r+2, k_block] = SiLU(XW1[r:r+2]) * XW3[r:r+2]
//       for c_block in C_blocks:
//         Y[r:r+2, c_block] += M[r:r+2] @ W2[k_block, c_block]
//     pop X (once per row pair)
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
constexpr uint32_t max_rows_for_sync = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(4);

// rt_dim for matmul_block: process this many rows at once
constexpr uint32_t rt_dim = 2;
// ct_dim for matmul_block: process this many output columns at once (4/rt_dim due to DST limit)
constexpr uint32_t ct_dim = block_size / rt_dim;  // = 2 when block_size=4, rt_dim=2

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r:r+rt_dim, :] - FULL cached row (rt_dim × Wt tiles)
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k_block]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k_block]
// CBs with intermediate computations (sized for rt_dim rows)
constexpr auto cb_xw1_partial_idx = tt::CBIndex::c_4;  // Partial XW1[r:r+rt_dim, k_block]
constexpr auto cb_xw3_partial_idx = tt::CBIndex::c_5;  // Partial XW3[r:r+rt_dim, k_block]
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;          // Final XW1[r:r+rt_dim, k_block]
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;          // Final XW3[r:r+rt_dim, k_block]
constexpr auto cb_m_idx = tt::CBIndex::c_8;            // M[r:r+rt_dim, k_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;    // Partial Y[r:r+rt_dim, c_block]
constexpr auto cb_y_idx = tt::CBIndex::c_10;           // Final Y[r:r+rt_dim, c_block]

// ============================================================================
// Accumulate XW result for one k_block across p_blocks (rt_dim=2 version)
//
// Phase 3: X is cached in CB for full row. x_p_block_offset tells us where
// in the cached X to read for this p_block.
//
// With rt_dim=2, ct_dim=2:
// - Process 2 rows × 2 k-columns per matmul_block call
// - Need 2 calls per p to cover all 4 k-columns (k=0,1 and k=2,3)
// - DST layout after each call: [r0_k0, r0_k1, r1_k0, r1_k1]
//
// X CB layout (cached full rows): [r0_p0, r0_p1, ..., r0_pWt-1, r1_p0, r1_p1, ..., r1_pWt-1]
// For p_block starting at p_block_start: read from offset p_block_start
//
// Output CB layout (row-major): [r0_k0, r0_k1, r0_k2, r0_k3, r1_k0, r1_k1, r1_k2, r1_k3]
// Total: rt_dim × block_size = 2 × 4 = 8 tiles
// ============================================================================
inline void accumulate_XW_for_k_block_rt2(
    const tt::CBIndex cb_x_idx,
    const tt::CBIndex cb_w_idx,
    const tt::CBIndex cb_partial_idx,
    const tt::CBIndex cb_final_idx,
    const uint32_t x_p_block_offset,  // Offset into cached X row for this p_block
    const uint32_t p_block_size,
    const uint32_t k_block_size,
    const uint32_t actual_rows,  // 1 or 2 (for handling odd total rows)
    const bool first_p_block,
    const bool last_p_block) {
    constexpr uint32_t tiles_per_row = block_size;                // k-dimension tiles per row
    constexpr uint32_t total_output_tiles = rt_dim * block_size;  // 8 tiles for 2 rows × 4 k

    tile_regs_acquire();

    // Load previous partial results if not first p_block
    // Layout in CB: row-major [r0_k0, r0_k1, r0_k2, r0_k3, r1_k0, r1_k1, r1_k2, r1_k3]
    if (!first_p_block) {
        cb_wait_front(cb_partial_idx, total_output_tiles);
        copy_tile_init(cb_partial_idx);
        // Load row 0's k tiles into DST 0,1,2,3
        for (uint32_t k = 0; k < block_size; ++k) {
            copy_tile(cb_partial_idx, k, k);
        }
        // Load row 1's k tiles into DST 4,5,6,7
        for (uint32_t k = 0; k < block_size; ++k) {
            copy_tile(cb_partial_idx, block_size + k, block_size + k);
        }
        cb_pop_front(cb_partial_idx, total_output_tiles);
    }

    // Wait for W tiles (batched: block_size rows × block_size cols)
    // W layout in CB: row-major [p0_k0, p0_k1, p0_k2, p0_k3, p1_k0, ...]
    constexpr uint32_t tiles_per_w_batch = block_size * block_size;
    cb_wait_front(cb_w_idx, tiles_per_w_batch);

    // Process in two halves: first k=0,1, then k=2,3
    // Each half uses matmul_block(rt_dim=2, ct_dim=2) producing 4 DST tiles

    // Initialize for rt_dim=2, ct_dim=2
    mm_block_init_short(
        cb_x_idx,
        cb_w_idx,
        /*transpose=*/false,
        /*ct_dim=*/ct_dim,
        /*rt_dim=*/rt_dim,
        /*kt_dim=*/p_block_size);

    // Phase 3: X is cached as full rows in CB
    // X layout in CB: [r0_p0, r0_p1, ..., r0_pWt-1, r1_p0, r1_p1, ..., r1_pWt-1]
    // For this p_block: X[r0, p] at x_p_block_offset + p, X[r1, p] at Wt + x_p_block_offset + p
    // For rt_dim=2: in0 reads tiles at in0_index (r0) and in0_index + Wt (r1)

    // --- First k-half: k=0,1 ---
    // DST output positions: [0,1,2,3] = [r0_k0, r0_k1, r1_k0, r1_k1]
    for (uint32_t p = 0; p < p_block_size; ++p) {
        // in0_index: X tile for this p within the cached row
        // X[r0, p_block_start + p] at (x_p_block_offset + p)
        // X[r1, p_block_start + p] at (Wt + x_p_block_offset + p)
        const uint32_t in0_index = x_p_block_offset + p;
        // in1_index: W[p, k=0] - first k-half
        const uint32_t in1_index = p * block_size + 0;  // W row p, cols 0,1
        matmul_block(
            cb_x_idx,
            cb_w_idx,
            in0_index,
            in1_index,
            /*dst_index=*/0,
            /*transpose=*/false,
            /*ct_dim=*/ct_dim,
            /*rt_dim=*/rt_dim,
            /*kt_dim=*/p_block_size);
    }

    // --- Second k-half: k=2,3 ---
    // DST output positions: [4,5,6,7] mapped as [r0_k2, r0_k3, r1_k2, r1_k3]
    // But we need to pack as: [r0_k0, r0_k1, r0_k2, r0_k3, r1_k0, r1_k1, r1_k2, r1_k3]
    // So second half goes to DST 2,3 for r0 and DST 6,7 for r1
    //
    // Actually, matmul_block with rt_dim=2 produces:
    // DST[dst_index + 0] = row0, col0
    // DST[dst_index + 1] = row0, col1  (when ct_dim=2)
    // DST[dst_index + ct_dim] = row1, col0
    // DST[dst_index + ct_dim + 1] = row1, col1
    //
    // So for dst_index=0, ct_dim=2: [0]=r0_k0, [1]=r0_k1, [2]=r1_k0, [3]=r1_k1
    // For dst_index=4, ct_dim=2: [4]=r0_k2, [5]=r0_k3, [6]=r1_k2, [7]=r1_k3
    //
    // We want final layout: [r0_k0, r0_k1, r0_k2, r0_k3, r1_k0, r1_k1, r1_k2, r1_k3]
    // But matmul_block gives: [r0_k0, r0_k1, r1_k0, r1_k1, r0_k2, r0_k3, r1_k2, r1_k3]
    //
    // Need to rearrange before packing, or adjust the pack order

    for (uint32_t p = 0; p < p_block_size; ++p) {
        // Phase 3: Use cached X offset
        const uint32_t in0_index = x_p_block_offset + p;
        const uint32_t in1_index = p * block_size + ct_dim;  // W row p, cols 2,3
        matmul_block(
            cb_x_idx,
            cb_w_idx,
            in0_index,
            in1_index,
            /*dst_index=*/4,
            /*transpose=*/false,  // Second half of DST
            /*ct_dim=*/ct_dim,
            /*rt_dim=*/rt_dim,
            /*kt_dim=*/p_block_size);
    }

    cb_pop_front(cb_w_idx, tiles_per_w_batch);

    tile_regs_commit();

    // Pack to CB in row-major order: [r0_k0, r0_k1, r0_k2, r0_k3, r1_k0, r1_k1, r1_k2, r1_k3]
    // DST has: [r0_k0, r0_k1, r1_k0, r1_k1, r0_k2, r0_k3, r1_k2, r1_k3]
    // Pack order: 0, 1, 4, 5, 2, 3, 6, 7
    const auto output_cb_idx = last_p_block ? cb_final_idx : cb_partial_idx;

    // Row 0: DST 0, 1, 4, 5 -> k0, k1, k2, k3
    pack_and_push(0, output_cb_idx);
    pack_and_push(1, output_cb_idx);
    pack_and_push(4, output_cb_idx);
    pack_and_push(5, output_cb_idx);

    // Row 1: DST 2, 3, 6, 7 -> k0, k1, k2, k3 (only if actual_rows == 2)
    if (actual_rows == rt_dim) {
        pack_and_push(2, output_cb_idx);
        pack_and_push(3, output_cb_idx);
        pack_and_push(6, output_cb_idx);
        pack_and_push(7, output_cb_idx);
    }
}

// ============================================================================
// Compute M = SiLU(XW1) * XW3 for rt_dim rows
// ============================================================================
inline void compute_M_partial_for_k_block_rt2(const uint32_t k_block_size, const uint32_t actual_rows) {
    constexpr uint32_t tiles_per_row = block_size;
    const uint32_t total_tiles = actual_rows * tiles_per_row;

    cb_wait_front(cb_xw1_idx, total_tiles);
    cb_wait_front(cb_xw3_idx, total_tiles);

    // Process all tiles (both rows if actual_rows == 2)
    for (uint32_t tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
        // Only process valid k tiles within each row
        const uint32_t row = tile_idx / block_size;
        const uint32_t k = tile_idx % block_size;
        if (k >= k_block_size) {
            continue;  // Skip padding tiles
        }

        constexpr uint32_t xw1_reg = 0U;
        constexpr uint32_t xw3_reg = 1U;
        constexpr uint32_t silu_reg = 2U;
        constexpr uint32_t m_reg = 3U;

        tile_regs_acquire();

        copy_tile_init(cb_xw1_idx);
        copy_tile(cb_xw1_idx, tile_idx, xw1_reg);
        copy_tile_init(cb_xw3_idx);
        copy_tile(cb_xw3_idx, tile_idx, xw3_reg);

        // SiLU(XW1)
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

    // Push padding tiles for incomplete k_block
    const uint32_t valid_tiles = actual_rows * k_block_size;
    const uint32_t padding_tiles = total_tiles - valid_tiles;
    if (padding_tiles > 0) {
        tile_regs_acquire();
        tile_regs_commit();
        pack_and_push_block(cb_m_idx, padding_tiles);
    }

    cb_pop_front(cb_xw1_idx, total_tiles);
    cb_pop_front(cb_xw3_idx, total_tiles);
}

// ============================================================================
// Accumulate Y += M @ W2 for one c_block (rt_dim rows version)
// ============================================================================
inline void accumulate_Y_for_c_block_rt2(
    const uint32_t k_block_size,
    const uint32_t c_block_size,
    const uint32_t actual_rows,
    const bool first_k_block,
    const bool last_k_block) {
    constexpr uint32_t tiles_per_w2_batch = block_size * block_size;
    const uint32_t m_tiles = actual_rows * block_size;

    cb_wait_front(cb_w2_idx, tiles_per_w2_batch);

    // Process each output column one at a time (same as before, but for rt_dim rows)
    // TODO: Optimize with rt_dim=2, ct_dim=2 for Y accumulation too
    for (uint32_t c = 0; c < c_block_size; ++c) {
        for (uint32_t row = 0; row < actual_rows; ++row) {
            tile_regs_acquire();

            // Load previous Y_partial if not first k_block
            if (!first_k_block) {
                cb_wait_front(cb_y_partial_idx, 1);
                copy_tile_init(cb_y_partial_idx);
                copy_tile(cb_y_partial_idx, 0, 0);
                cb_pop_front(cb_y_partial_idx, 1);
            }

            // Y[row, c] += M[row, :] @ W2[:, c]
            mm_init_short(cb_m_idx, cb_w2_idx, false);
            const uint32_t m_row_offset = row * block_size;
            const uint32_t w2_col_offset = c * block_size;  // W2 is column-major in CB
            for (uint32_t k = 0; k < k_block_size; ++k) {
                matmul_tiles(cb_m_idx, cb_w2_idx, m_row_offset + k, w2_col_offset + k, 0);
            }

            tile_regs_commit();

            const auto output_cb_idx = last_k_block ? cb_y_idx : cb_y_partial_idx;
            pack_and_push(0, output_cb_idx);
        }
    }

    cb_pop_front(cb_w2_idx, tiles_per_w2_batch);
}

// ============================================================================
// TRUE FLASH MAIN KERNEL with rt_dim=2 and X Row Caching (Phase 3)
// ============================================================================
void kernel_main() {
    init_sfpu(cb_input_idx, cb_y_idx);
    binary_op_init_common(cb_input_idx, cb_w1_idx, cb_y_idx);

    const uint32_t num_k_blocks = (hidden_Wt + block_size - 1) / block_size;
    const uint32_t num_p_blocks = (Wt + block_size - 1) / block_size;
    const uint32_t num_c_blocks = (Wt + block_size - 1) / block_size;

    // X CB now holds full cached rows: rt_dim × Wt tiles
    constexpr uint32_t x_cache_tiles = rt_dim * Wt;

    // Process rows in pairs (rt_dim=2)
    for (uint32_t r = 0; r < max_rows_for_sync; r += rt_dim) {
        // Determine how many rows to actually process (1 or 2)
        // For padding rows and odd last row, we may have fewer than rt_dim
        uint32_t actual_rows = rt_dim;
        if (r + rt_dim > num_rows_per_core) {
            actual_rows = (r < num_rows_per_core) ? (num_rows_per_core - r) : 0;
        }
        const bool is_padding_pair = (actual_rows == 0);

        // ---- Phase 3: Wait for FULL X rows (cached in CB) ----
        // X is read once per row pair and reused for all k_blocks
        cb_wait_front(cb_input_idx, x_cache_tiles);

        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            const uint32_t k_block_start = k_block_idx * block_size;
            const uint32_t k_block_size =
                (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;
            const bool first_k_block = (k_block_idx == 0);
            const bool last_k_block = (k_block_idx == num_k_blocks - 1);

            // ---- Phase A: Compute XW1 and XW3 for rt_dim rows ----
            // X is already cached - iterate through p_blocks reading from cache
            for (uint32_t p_block_idx = 0; p_block_idx < num_p_blocks; ++p_block_idx) {
                const uint32_t p_block_start = p_block_idx * block_size;
                const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;
                const bool first_p_block = (p_block_idx == 0);
                const bool last_p_block = (p_block_idx == num_p_blocks - 1);

                // X is cached - use p_block_start as offset into cached row
                accumulate_XW_for_k_block_rt2(
                    cb_input_idx,
                    cb_w1_idx,
                    cb_xw1_partial_idx,
                    cb_xw1_idx,
                    p_block_start,  // Offset into cached X row
                    p_block_size,
                    k_block_size,
                    actual_rows > 0 ? actual_rows : rt_dim,
                    first_p_block,
                    last_p_block);

                accumulate_XW_for_k_block_rt2(
                    cb_input_idx,
                    cb_w3_idx,
                    cb_xw3_partial_idx,
                    cb_xw3_idx,
                    p_block_start,  // Offset into cached X row
                    p_block_size,
                    k_block_size,
                    actual_rows > 0 ? actual_rows : rt_dim,
                    first_p_block,
                    last_p_block);

                // NOTE: Don't pop X here - it's reused for all k_blocks!
            }

            // ---- Phase B: Compute M = SiLU(XW1) * XW3 ----
            const uint32_t rows_for_m = actual_rows > 0 ? actual_rows : rt_dim;
            compute_M_partial_for_k_block_rt2(k_block_size, rows_for_m);

            // ---- Phase C: Accumulate Y += M @ W2 ----
            const uint32_t m_tiles = rows_for_m * block_size;
            cb_wait_front(cb_m_idx, m_tiles);

            for (uint32_t c_block_idx = 0; c_block_idx < num_c_blocks; ++c_block_idx) {
                const uint32_t c_block_start = c_block_idx * block_size;
                const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;

                if (is_padding_pair) {
                    // Consume W2 without computing
                    constexpr uint32_t tiles_per_batch = block_size * block_size;
                    cb_wait_front(cb_w2_idx, tiles_per_batch);
                    cb_pop_front(cb_w2_idx, tiles_per_batch);
                } else {
                    accumulate_Y_for_c_block_rt2(k_block_size, c_block_size, actual_rows, first_k_block, last_k_block);
                }
            }

            cb_pop_front(cb_m_idx, m_tiles);
        }

        // ---- Pop cached X once after all k_blocks are done ----
        cb_pop_front(cb_input_idx, x_cache_tiles);
    }
}
