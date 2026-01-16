// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"

/**
 * Simplified DRAM streaming matmul compute kernel with subblocking.
 *
 * Each core computes: output[M=1, N=per_core_N] = in0[M=1, K] @ in1[K, N=per_core_N]
 *
 * in0 is fully available in L1 (replicated, tensor-backed CB).
 * in1 is streamed from DRAM, one Kx1 column stick at a time.
 *
 * Uses custom_mm_block which handles K-dimension reduction via MOP replay
 * for maximum throughput (eliminates software loop overhead).
 *
 * Uses subblock_w to batch tile processing, reducing cb_reserve_back and
 * tile_regs_acquire calls. subblock_w is determined by dest register availability:
 * - FP32 dest: 8 (full sync) or 4 (half sync)
 * - BF16/FP16 dest: 16 (full sync) or 8 (half sync)
 *
 * Loop structure:
 *   for each subblock in N (per_core_N / subblock_w iterations):
 *     reserve subblock_w output tiles
 *     for each tile in subblock:
 *       wait for in1 Kx1 stick
 *       accumulate across K into dest[w]
 *       pop in1 Kx1 stick
 *     pack subblock_w output tiles
 */
namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);

    constexpr uint32_t transpose = false;
    constexpr uint32_t num_subblocks = per_core_N / subblock_w;

    // Initialize custom matmul with K-dimension optimization
    custom_mm_block_init(cb_id_in0, cb_id_in1, cb_id_out, transpose, num_tiles_k);

    // Wait for all in0 tiles (replicated, tensor-backed - always available)
    cb_wait_front(cb_id_in0, num_tiles_k);

    // Reserve all output tiles upfront (tensor-backed CB)
    cb_reserve_back(cb_id_out, per_core_N);

    // Process subblocks of output tiles
    for (uint32_t sb = 0; sb < num_subblocks; sb++) {
        // Accumulate subblock_w output tiles into dest registers
        tile_regs_acquire();
        for (uint32_t w = 0; w < subblock_w; w++) {
            // Wait for in1 Kx1 stick (matching reader's production rate)
            cb_wait_front(cb_id_in1, num_tiles_k);

            // Compute output tile w, accumulating into dest[w]
            custom_mm_block(cb_id_in0, cb_id_in1, 0, 0, w, transpose, num_tiles_k);

            // Pop in1 Kx1 stick
            cb_pop_front(cb_id_in1, num_tiles_k);
        }
        tile_regs_commit();

        // Pack all output tiles in subblock to correct CB offsets
        tile_regs_wait();
        for (uint32_t w = 0; w < subblock_w; w++) {
            pack_tile(w, cb_id_out, sb * subblock_w + w);
        }
        tile_regs_release();
    }

    // Push all output tiles at once
    cb_push_back(cb_id_out, per_core_N);

    // Pop in0 (only once at the end)
    cb_pop_front(cb_id_in0, num_tiles_k);
}
}  // namespace NAMESPACE
