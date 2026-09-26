// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
// Blackhole-only: the SFPU triangle-solve LLK lives only in the Blackhole tt-llk sfpu section.
// Gating the include keeps this Compute API header parseable on every architecture (the public
// entry points below are likewise compiled only for Blackhole), rather than failing JIT on a
// missing header.
#if defined(ARCH_BLACKHOLE)
#include "sfpu/ckernel_sfpu_triangle_solve.h"
#endif
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// clang-format off
/**
 * Per-tile forward-substitution triangle solve of  L X = RHS  for a 32x32 tile. L is the unit
 * lower-triangular matrix, supplied plain (strict-lower entries NOT negated); the per-column update
 * subtracts L[row][col] * X[col] directly. L is read element-by-element straight from L1 (not a DST
 * register), so only the RHS occupies a DST input. The DST register buffer must be in the acquired
 * state via *tile_regs_acquire*, and the L tile must be resident in cb_l (*cb_wait_front* done).
 * The solution is left in DST[idst_out] in standard tile layout. Blocking; compute-engine only.
 *
 * | Argument     | Description                                                                | Type            |
 * |--------------|----------------------------------------------------------------------------|-----------------|
 * | DATA_FORMAT  | Data format of the DST tiles                                               | DataFormat      |
 * | cb_l         | CircularBuffer holding the unit-lower-tri L tile (bf16)                    | CircularBuffer& |
 * | l_tile_idx   | Tile index of L within cb_l (front-relative)                              | uint32_t        |
 * | idst_in      | DST index of the right-hand-side tile (RHS)                               | uint32_t        |
 * | idst_out     | DST index that receives the solution X of  L X = RHS                       | uint32_t        |
 */
// clang-format on
template <DataFormat DATA_FORMAT = DataFormat::Float16_b>
ALWI void triangle_solve_tile(CircularBuffer& cb_l, uint32_t l_tile_idx, uint32_t idst_in, uint32_t idst_out) {
    // First pass supports bf16 only: the SFPU microcode hardcodes bf16 L1 reads and the
    // SFPLOAD/SFPSTORE SRCB format mode, so any other DATA_FORMAT would silently misinterpret DST.
    static_assert(
        DATA_FORMAT == DataFormat::Float16_b, "triangle_solve_tile currently supports only DataFormat::Float16_b");
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

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
