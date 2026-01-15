// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

/**
 * Simplified DRAM streaming matmul compute kernel.
 *
 * Each core computes: output[M=1, N=per_core_N] = in0[M=1, K] @ in1[K, N=per_core_N]
 *
 * in0 is fully available in L1 (replicated, tensor-backed CB).
 * in1 is streamed from DRAM, one Kx1 column stick at a time.
 *
 * Loop structure:
 *   for each output tile in N:
 *     wait for in1 Kx1 stick
 *     accumulate across K: output[n] = sum_k(in0[k] * in1[k,n])
 *     pack output tile
 */
namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);

    // For single tile output at a time, simple matmul accumulation
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t out_subblock_w = 1;
    constexpr uint32_t in0_block_w = 1;  // Process one K tile at a time

    // Initialize matmul
    mm_block_init(cb_id_in0, cb_id_in1, cb_id_out, false, out_subblock_w, out_subblock_h, in0_block_w);

    // Wait for all in0 tiles (replicated, tensor-backed - always available)
    cb_wait_front(cb_id_in0, num_tiles_k);

    // Process each output tile in N dimension
    for (uint32_t n = 0; n < per_core_N; n++) {
        // Wait for in1 Kx1 stick (K tiles for this output column)
        cb_wait_front(cb_id_in1, num_tiles_k);

        // Accumulate across K dimension
        tile_regs_acquire();

        for (uint32_t k = 0; k < num_tiles_k; k++) {
            // in0 tile: k (from replicated input)
            // in1 tile: k (from current Kx1 stick)
            // dst: 0 (single output tile, accumulating)
            matmul_tiles(cb_id_in0, cb_id_in1, k, k, 0);
        }

        // Pop in1 Kx1 stick
        cb_pop_front(cb_id_in1, num_tiles_k);

        tile_regs_commit();

        // Pack output tile directly
        cb_reserve_back(cb_id_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_id_out);
        tile_regs_release();
        cb_push_back(cb_id_out, 1);
    }

    // Pop in0 (only once at the end)
    cb_pop_front(cb_id_in0, num_tiles_k);
}
}  // namespace NAMESPACE
