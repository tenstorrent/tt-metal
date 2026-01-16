// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"

// Fused SiLU activation support (only when FUSE_SILU is defined)
// For m=1 tiles, we use optimized SFPU with:
// - VectorMode::R: processes only Face0+Face1 (top row), 2x faster
// - ITERATIONS=2: processes 2 rows per face instead of 8, 4x faster
// Total: ~8x faster than default silu_tile()
#ifdef FUSE_SILU
#include "compute_kernel_api.h"  // for silu_tile_init() and llk_math_eltwise_unary_sfpu_silu
#endif

/**
 * Simplified DRAM streaming matmul compute kernel with subblocking.
 *
 * Each core computes: output[M=1, N=per_core_N] = in0[M=1, K] @ in1[K, N=per_core_N]
 *
 * in0 is fully available in L1 (replicated, tensor-backed CB).
 * in1 is streamed from DRAM, subblock_k tiles at a time.
 *
 * Uses custom_mm_block which handles K-dimension reduction via MOP replay
 * for maximum throughput (eliminates software loop overhead).
 *
 * Subblocking in both K and N dimensions:
 * - subblock_k: reduces initial latency by allowing compute to start sooner
 * - subblock_w: batches output tiles to reduce cb_reserve_back and tile_regs_acquire calls
 *
 * Loop structure:
 *   for each subblock in N (per_core_N / subblock_w iterations):
 *     for each tile w in subblock:
 *       for each K subblock:
 *         wait for in1 subblock_k tiles
 *         accumulate into dest[w]
 *         pop in1 subblock_k tiles
 *     apply optional SFPU activation
 *     pack subblock_w output tiles
 */
namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t subblock_k = get_compile_time_arg_val(3);  // tiles per K subblock
    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t num_subblocks_k = get_compile_time_arg_val(6);

    constexpr uint32_t transpose = false;
    constexpr uint32_t num_subblocks_n = per_core_N / subblock_w;
    constexpr uint32_t num_tiles_k = subblock_k * num_subblocks_k;

    // Initialize SiLU if fused
#ifdef FUSE_SILU
    silu_tile_init();
#endif

    // Initialize custom matmul
    // Use subblock_k for init since that's the K tiles per call
    custom_mm_block_init(cb_id_in0, cb_id_in1, cb_id_out, transpose, subblock_k);

    // Wait for all in0 tiles (replicated, tensor-backed - always available)
    cb_wait_front(cb_id_in0, num_tiles_k);

    // Process subblocks of output tiles
    for (uint32_t sb_n = 0; sb_n < num_subblocks_n; sb_n++) {
        // Reserve all output tiles upfront
        cb_reserve_back(cb_id_out, subblock_w);

        // Accumulate subblock_w output tiles into dest registers
        tile_regs_acquire();
        for (uint32_t w = 0; w < subblock_w; w++) {
            // Intermediate K subblocks: partial accumulation (no finalization)
            for (uint32_t sb_k = 0; sb_k < num_subblocks_k - 1; sb_k++) {
                cb_wait_front(cb_id_in1, subblock_k);
                custom_mm_block<true>(cb_id_in0, cb_id_in1, sb_k * subblock_k, 0, w, transpose, subblock_k);
                cb_pop_front(cb_id_in1, subblock_k);
            }
            // Final K subblock: full accumulation with finalization
            cb_wait_front(cb_id_in1, subblock_k);
            custom_mm_block<false>(
                cb_id_in0, cb_id_in1, (num_subblocks_k - 1) * subblock_k, 0, w, transpose, subblock_k);
            cb_pop_front(cb_id_in1, subblock_k);
        }

        // Apply fused SiLU activation - optimized for m=1 tiles
        // VectorMode::R (2x) + ITERATIONS=2 (4x) = ~8x speedup
#ifdef FUSE_SILU
        for (uint32_t i = 0; i < subblock_w; i++) {
            MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE, 2>(i, (int)VectorMode::R)));
        }
#endif
        tile_regs_commit();

        // Pack all output tiles in subblock to correct CB offsets
        tile_regs_wait();
        for (uint32_t w = 0; w < subblock_w; w++) {
            pack_tile(w, cb_id_out, w);
        }
        tile_regs_release();

        // Push all output tiles at once
        cb_push_back(cb_id_out, subblock_w);
    }

    // Pop in0 (only once at the end)
    cb_pop_front(cb_id_in0, num_tiles_k);
}
}  // namespace NAMESPACE
