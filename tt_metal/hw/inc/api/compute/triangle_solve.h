// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_triangle_solve.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Broadcast a single bf16 element of an L1-resident tile to all 32 lanes of an SFPU register, via a
 * direct L1 read + SFPLOADI (no DEST round-trip or cross-lane shuffle). Useful as a building block
 * for the triangle solve — e.g. splat the pivot L[k][k] before scaling the column beneath it.
 *
 * The tile must be resident in the CB (i.e. *cb.wait_front* has been done). The DST register buffer
 * does NOT need to be acquired: this touches an SFPU LREG only, so call it inside the same
 * *tile_regs_acquire* block as the SFPU op that consumes `out_lreg`.
 *
 * | Argument  | Description                                                          | Type            |
 * |-----------|----------------------------------------------------------------------|-----------------|
 * | cb        | CircularBuffer holding the tile (bf16 / Float16_b)                   | CircularBuffer& |
 * | tile_idx  | Tile index within the CB (front-relative)                           | uint32_t        |
 * | x, y      | Logical (row, col) in the 32x32 tile, each 0..31                     | uint32_t        |
 * | out_lreg  | Destination SFPU register (p_sfpu::LREGn)                           | uint32_t        |
 */
// clang-format on
ALWI void broadcast_tile_value(CircularBuffer& cb, uint32_t tile_idx, uint32_t x, uint32_t y, uint32_t out_lreg) {
    // Resolve the L1 byte address of the tile. get_tile_address computes it on the UNPACK thread
    // (owner of the CB read pointer) and mailboxes it to MATH/PACK, so it must run on all threads —
    // i.e. OUTSIDE the MATH() guard. The volatile read + SFPLOADI then happen on MATH.
    const uint32_t l1_tile_base = cb.get_tile_address(tile_idx);
    MATH((sfpu::broadcast_tile_value_bf16(l1_tile_base, x, y, out_lreg)));
}

// clang-format off
/**
 * Per-tile forward-substitution triangle solve of  L X = RHS  for a 32x32 tile. L is the unit
 * lower-triangular matrix, supplied NEGATED (strict-lower entries pre-multiplied by -1) so the
 * per-column update is an accumulate; it is read element-by-element straight from L1 (not a DST
 * register), so only the RHS occupies a DST input. The DST register buffer must be in the acquired
 * state via *tile_regs_acquire*, and the L tile must be resident in cb_l (*cb_wait_front* done).
 * The solution is left in DST[idst_out] in standard tile layout. Blocking; compute-engine only.
 *
 * | Argument     | Description                                                                | Type            |
 * |--------------|----------------------------------------------------------------------------|-----------------|
 * | DATA_FORMAT  | Data format of the DST tiles                                               | DataFormat      |
 * | cb_l         | CircularBuffer holding the negated unit-lower-tri L tile (bf16)            | CircularBuffer& |
 * | l_tile_idx   | Tile index of L within cb_l (front-relative)                              | uint32_t        |
 * | idst_in      | DST index of the right-hand-side tile (RHS)                               | uint32_t        |
 * | idst_out     | DST index that receives the solution X of  L X = RHS                       | uint32_t        |
 */
// clang-format on
template <DataFormat DATA_FORMAT = DataFormat::Float16_b>
ALWI void triangle_solve_tile(CircularBuffer& cb_l, uint32_t l_tile_idx, uint32_t idst_in, uint32_t idst_out) {
    // Resolve L's L1 base on all threads (get_tile_address runs on UNPACK and mailboxes the byte
    // address to MATH), then run the solve on MATH. The SFPU op does its own absolute DEST addressing
    // (idst * dst_tile_size), so start_(0) programs a zero base — the same way the binary SFPU macro
    // drives an SFPU op.
    const uint32_t l1_tri_base = cb_l.get_tile_address(l_tile_idx);
    MATH((_llk_math_eltwise_sfpu_start_(0)));
    MATH((sfpu::triangle_solve<DATA_FORMAT, false /*APPROXIMATE*/>(idst_in, idst_out, l1_tri_base)));
    MATH((_llk_math_eltwise_sfpu_done_()));
}

/**
 * Initialization for triangle_solve_tile. Must be called before triangle_solve_tile.
 * Reuses SfpuType::unused with a per-op init callback (no dedicated SfpuType enum needed).
 */
ALWI void triangle_solve_tile_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::triangle_solve_init))); }

}  // namespace ckernel
