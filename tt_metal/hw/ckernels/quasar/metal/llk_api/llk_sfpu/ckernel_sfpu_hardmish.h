// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// hardmish(x) = x * clamp(x + 2, 0, 2) / 2
//             = x * clamp(0.5 * x + 1.0, 0.0, 1.0)
//
// For finite x, the piecewise form is:
//   x <= -2  =>  0         (scale clamped to 0)
//   x >= 0   =>  x         (scale clamped to 1)
//   else     =>  x*(x+2)/2 (quadratic)
//
// Non-finite inputs follow IEEE 754 operations above.
// In particular, x = -inf clamps scale to 0, and (-inf) * 0 yields NaN.
//
// Constants 0.5 and 1.0 are exactly representable in IEEE 754.
// Clamping to [0, 1] gives exact boundary values (0 or 1),
// so the final multiply produces exact 0 or exact x at transitions.
inline void hardmish_init() {
    // The paired store walks Dest through ADDR_MOD_6, advancing by the two rows the loop just wrote
    // (one sfpi row is SFP_ROWS Dest counter rows), so the loop body needs no separate increment.
    // The SFPU load/store addr_mod field is 3 bits wide on Quasar, so the store names slot 6 directly
    // (as square does).
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2 * ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    // 0.5 lives in a programmable constant register so the body does not re-load it per row.
    sfpi::vConstFloatPrgm0 = 0.5f;
}

/**
 * hardmish(x) on ITERATIONS Dest rows, two rows per iteration with the statements interleaved: each
 * row's MAD -> clamp (2x SFPSWAP) -> MUL chain is serial, so pairing two independent rows fills the
 * dependency stalls with the other row's work.
 *
 * @tparam APPROXIMATION_MODE: Unused; the kernel is exact in both modes.
 * @tparam ITERATIONS: Number of Dest rows to process; must be even.
 * @note Call @ref hardmish_init after any other SFPU init that writes vConstFloatPrgm0 or ADDR_MOD_6;
 *       this reads 0.5 from vConstFloatPrgm0 and advances Dest through ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void hardmish() {
    static_assert(ITERATIONS % 2 == 0, "hardmish processes two rows per iteration");
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::vFloat scale0 = x0 * sfpi::vConstFloatPrgm0 + 1.0f;
        sfpi::vFloat scale1 = x1 * sfpi::vConstFloatPrgm0 + 1.0f;

        scale0 = sfpi::clamp(scale0, /*lower*/ 0.0f, /*upper*/ 1.0f);
        scale1 = sfpi::clamp(scale1, 0.0f, 1.0f);

        sfpi::dst_reg[0] = x0 * scale0;
        // Row 1's store also advances Dest past both rows (ADDR_MOD_6 from hardmish_init).
        sfpi::dst_reg[1].mode(ADDR_MOD_6) = x1 * scale1;
    }
}

}  // namespace sfpu
}  // namespace ckernel
