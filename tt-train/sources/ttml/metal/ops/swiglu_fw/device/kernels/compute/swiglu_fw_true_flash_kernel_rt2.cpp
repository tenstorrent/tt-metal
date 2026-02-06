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
// Phase 4: L1 Accumulation
//   - Instead of loading partial from CB into DST, we use packer L1 accumulation
//   - Packer reads from CB, adds DST value, writes back to CB
//   - Reserve CB ONCE at first_p_block, push ONCE at last_p_block
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

    const uint32_t output_tiles = actual_rows * block_size;

    // Phase 4: L1 Accumulation
    // - Always accumulate into partial CB using L1 acc
    // - On last_p_block: copy final result to final CB
    //
    // Why not accumulate directly to final CB?
    // - Partial CB is configured for L1 acc (may have different memory layout)
    // - Final CB is consumer-facing and needs standard layout

    // Reserve CB space ONCE at start of p_block accumulation
    if (first_p_block) {
        cb_reserve_back(cb_partial_idx, output_tiles);
    }

    tile_regs_acquire();

    // Phase 4: No need to manually load partial from CB into DST!
    // The packer will handle this via L1 accumulation

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

    // --- First k-half: k=0,1 ---
    for (uint32_t p = 0; p < p_block_size; ++p) {
        const uint32_t in0_index = x_p_block_offset + p;
        const uint32_t in1_index = p * block_size + 0;
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
    for (uint32_t p = 0; p < p_block_size; ++p) {
        const uint32_t in0_index = x_p_block_offset + p;
        const uint32_t in1_index = p * block_size + ct_dim;
        matmul_block(
            cb_x_idx,
            cb_w_idx,
            in0_index,
            in1_index,
            /*dst_index=*/4,
            /*transpose=*/false,
            /*ct_dim=*/ct_dim,
            /*rt_dim=*/rt_dim,
            /*kt_dim=*/p_block_size);
    }

    cb_pop_front(cb_w_idx, tiles_per_w_batch);

    tile_regs_commit();
    tile_regs_wait();

    // Phase 4: Configure L1 accumulation before packing
    // - first_p_block: l1_acc=0 (write, not accumulate)
    // - subsequent p_blocks: l1_acc=1 (read-add-write)
    pack_reconfig_data_format(cb_partial_idx);
    if (!first_p_block) {
        PACK((llk_pack_reconfig_l1_acc(1)));
    }

    // Pack to CB in row-major order: [r0_k0, r0_k1, r0_k2, r0_k3, r1_k0, r1_k1, r1_k2, r1_k3]
    // DST has: [r0_k0, r0_k1, r1_k0, r1_k1, r0_k2, r0_k3, r1_k2, r1_k3]
    // Use pack_tile<true> to pack to specific positions for L1 accumulation
    // dst_index -> cb_position mapping:
    //   DST 0 -> CB pos 0 (r0_k0)
    //   DST 1 -> CB pos 1 (r0_k1)
    //   DST 4 -> CB pos 2 (r0_k2)
    //   DST 5 -> CB pos 3 (r0_k3)
    //   DST 2 -> CB pos 4 (r1_k0)
    //   DST 3 -> CB pos 5 (r1_k1)
    //   DST 6 -> CB pos 6 (r1_k2)
    //   DST 7 -> CB pos 7 (r1_k3)

    // Row 0: DST 0, 1, 4, 5 -> CB positions 0, 1, 2, 3
    pack_tile<true>(0, cb_partial_idx, 0);
    pack_tile<true>(1, cb_partial_idx, 1);
    pack_tile<true>(4, cb_partial_idx, 2);
    pack_tile<true>(5, cb_partial_idx, 3);

    // Row 1: DST 2, 3, 6, 7 -> CB positions 4, 5, 6, 7 (only if actual_rows == 2)
    if (actual_rows == rt_dim) {
        pack_tile<true>(2, cb_partial_idx, 4);
        pack_tile<true>(3, cb_partial_idx, 5);
        pack_tile<true>(6, cb_partial_idx, 6);
        pack_tile<true>(7, cb_partial_idx, 7);
    }

    tile_regs_release();

    // Phase 4: Reset L1 acc and finalize on last p_block
    if (last_p_block) {
        PACK((llk_pack_reconfig_l1_acc(0)));
        cb_push_back(cb_partial_idx, output_tiles);

        // Copy from partial to final CB
        cb_wait_front(cb_partial_idx, output_tiles);
        cb_reserve_back(cb_final_idx, output_tiles);

        tile_regs_acquire();
        copy_tile_init(cb_partial_idx);
        for (uint32_t i = 0; i < output_tiles; ++i) {
            copy_tile(cb_partial_idx, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_final_idx);
        for (uint32_t i = 0; i < output_tiles; ++i) {
            pack_tile(i, cb_final_idx);
        }
        tile_regs_release();

        cb_pop_front(cb_partial_idx, output_tiles);
        cb_push_back(cb_final_idx, output_tiles);
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
