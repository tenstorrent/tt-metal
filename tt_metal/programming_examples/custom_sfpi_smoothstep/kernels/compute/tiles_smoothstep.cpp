// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_smoothstep.h"
#endif

/**
 * High-Level SFPU API Function (Smoothstep)
 *
 * Public interface for kernel code to perform smoothstep SFPU operations.
 * Abstraction layers:
 *   1. smoothstep_tile_face()        - Low-level SFPU function (tile face)
 *   2. my_smoothstep_tile_internal() - LLK wrapper (invokes for each face)
 *   3. my_smoothstep_tiles()         - High-level API (easy to use)
 *
 * Usage:
 *   - Call after loading data into Dst registers with copy_tile()
 *   - Specify Dst register indices (not CB indices!)
 *   - Results written to specified Dst register
 */
inline void my_smoothstep_tiles(uint32_t idx_dst0, float edge0, float edge1, float inv_delta) {
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, smoothstep_tile_face, idx_dst0, VectorMode::RC, edge0, edge1, inv_delta));
}

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Input circular buffer for tiles
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    // Output circular buffer for results
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // Initialize SFPU for computation using cb_in0 and cb_out0
    compute_kernel_hw_startup(cb_in0, cb_out0);
    copy_init(cb_in0);

    // precompute inverse of (edge1 - edge0) for efficiency
    constexpr float edge0 = 0.0f;
    constexpr float edge1 = 1.0f;
    float inv_delta = 1.0f / (edge1 - edge0);

    // Loop over all tiles and apply smoothstep
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait for input tile
        cb_wait_front(cb_in0, 1);
        // Acquire tile registers (8 at a time)
        tile_regs_acquire();
        // Copy input tile from circular buffer to Dst register
        copy_tile(cb_in0, 0, 0);  // input x
        // Apply smoothstep SFPU operation
        my_smoothstep_tiles(0, edge0, edge1, inv_delta);  // <-- Custom SFPU smoothstep
        // Commit and wait for register transfer
        tile_regs_commit();
        tile_regs_wait();
        // Reserve space in output buffer
        cb_reserve_back(cb_out0, 1);
        // Pack result tile to output buffer
        pack_tile(0, cb_out0);
        // Push result and pop input
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        // Release tile registers
        tile_regs_release();
    }
}
