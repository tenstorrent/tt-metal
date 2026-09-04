// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Configure the SFPU address mode used by the square op.
 *
 * Programs ADDR_MOD_6 with a dest increment of 2 (one SFPU pass writes 2 rows on Quasar).
 *
 * @note Call this before @ref calculate_square to set up the address mode it relies on.
 */
inline void init_square() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
}

/**
 * @brief Square one SFPU pass worth of rows (Quasar = 2 rows): dest = x * x.
 *
 * Loads x from dest, multiplies it by itself, and stores the result back to dest using
 * ADDR_MOD_6 to advance to the next pair of rows.
 *
 * @note ADDR_MOD_6 must already be programmed by @ref init_square.
 */
inline void _calculate_square_sfp_rows_() {
    sfpi::vFloat v = sfpi::dst_reg[0];                     // load x from dest (SFPLOAD)
    sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = v * v;  // x * x via SFPMUL, store back to dest (SFPSTORE)
}

/**
 * @brief Square a full Dest tile in place: dest = x * x.
 *
 * @tparam ITERATIONS: Number of SFPU passes (each covers 2 rows) needed to span the tile.
 * @note Call @ref init_square before this to program the address mode it depends on.
 */
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_square() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_square_sfp_rows_();
    }
}

// Square<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: square_tile / square_tile_init (compute_kernel_api.h).
// The Quasar kernel takes only an iteration count, so APPROXIMATION_MODE and DST_ACCUM are accepted for
// interface parity with WH/BH and ignored here; init runs init_square to program ADDR_MOD_6.
template <[[maybe_unused]] bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = SFPU_ITERATIONS>
struct Square : SfpuUnaryOp<Square<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_square<ITERATIONS>(); }

    static void init_kernel() { init_square(); }
};

}  // namespace sfpu
}  // namespace ckernel
