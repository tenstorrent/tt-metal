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
inline void hardmish_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void hardmish() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat scale = x * 0.5f + 1.0f;

        scale = sfpi::clamp(scale, 0.0f, 1.0f);

        sfpi::dst_reg[0] = x * scale;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
