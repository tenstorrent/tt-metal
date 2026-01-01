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
#if !(defined(MM_THROTTLE) && MM_THROTTLE > 0)
    // The paired store walks dest through physical addr_mod slot 6, advancing by the two rows the loop
    // just wrote (one sfpi row is two dest counter steps), so the loop body needs no separate increment.
    // Unary SFPU sets addr_mod_base, so the store names ADDR_MOD_2, which maps to physical slot 6.
    // A throttled matmul (MM_THROTTLE > 0) programs ADDR_MOD_6 itself, so there the body increments
    // dest explicitly and this init leaves the slot alone.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
#endif
    math::reset_counters(p_setrwc::SET_ABD_F);
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
#if defined(MM_THROTTLE) && MM_THROTTLE > 0
        sfpi::dst_reg[1] = x1 * scale1;
        sfpi::dst_reg += 2;
#else
        sfpi::dst_reg[1].mode(ADDR_MOD_2) = x1 * scale1;
#endif
    }
}

}  // namespace sfpu
}  // namespace ckernel
