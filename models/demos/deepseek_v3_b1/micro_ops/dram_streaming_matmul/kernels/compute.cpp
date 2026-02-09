// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"

// Fused SiLU activation support (only when FUSE_SILU is defined)
// For tiny tiles (m <= 8), we use optimized SFPU with:
// - VectorMode::R: processes only row faces, 2x faster
// - ITERATIONS: minimum 2 required for SFPU, then scales (m<=4->2, m=8->4, m>=16->8)
// Total: significant speedup vs default silu_tile()
#ifdef FUSE_SILU
#include "api/compute/compute_kernel_api.h"  // for silu_tile_init() and llk_math_eltwise_unary_sfpu_silu
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
void kernel_main() {
    // Compile time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t subblock_k = get_compile_time_arg_val(3);  // tiles per K subblock
    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t num_subblocks_k = get_compile_time_arg_val(6);
    constexpr uint32_t tile_r_dim = get_compile_time_arg_val(7);  // tile row dimension (m)

    constexpr uint32_t transpose = false;
    constexpr uint32_t num_subblocks_n = per_core_N / subblock_w;
    constexpr uint32_t num_tiles_k = subblock_k * num_subblocks_k;

    // Initialize SiLU if fused
#ifdef FUSE_SILU
    silu_tile_init();
#endif

    // Initialize custom matmul
    // Use subblock_k for init since that's the K tiles per call
    custom_mm_block_init<false, true>(cb_id_in0, cb_id_in1, cb_id_out);

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
                custom_mm_block<false>(cb_id_in0, cb_id_in1, sb_k * subblock_k, 0, w, subblock_k);
                cb_pop_front(cb_id_in1, subblock_k);
            }
            // Final K subblock: full accumulation with finalization
            cb_wait_front(cb_id_in1, subblock_k);
            custom_mm_block<true>(cb_id_in0, cb_id_in1, (num_subblocks_k - 1) * subblock_k, 0, w, subblock_k);
            cb_pop_front(cb_id_in1, subblock_k);
        }

        // Apply fused SiLU activation - optimized for tiny tiles
        // VectorMode::R processes only row faces (2x speedup)
        // ITERATIONS: minimum 2 for m<=4, then scale up (m=8->4, m=16->8)
#ifdef FUSE_SILU
        for (uint32_t i = 0; i < subblock_w; i++) {
            // ITERATIONS must be compile-time constant for template
            // Minimum ITERATIONS=2 required for SFPU to work correctly with VectorMode::R
            if constexpr (tile_r_dim <= 4) {
                MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE, 2>(i, (int)VectorMode::R)));
            } else if constexpr (tile_r_dim == 8) {
                MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE, 4>(i, (int)VectorMode::R)));
            } else {
                // For larger tiles (16, 32), use full iterations
                MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE, 8>(i, (int)VectorMode::R)));
            }
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
