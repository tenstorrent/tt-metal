// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void sign_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sign(const std::uint32_t /*exponent_size_8*/) {
// All params are in FP16 format
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // SFPSETCC is unspecified for -0.0 (VectorUnit.md), so a bare compare can miss it.
        // abs is taken first so copysgn can overwrite v in place (SFPSETSGN reads the sign
        // from its destination); otherwise v must be copied out before the stamp.
        sfpi::vFloat a = sfpi::abs(v);
        // copysgn stamps v's sign bit onto 1.0, which is exactly the v < 0 arm. Only the
        // zero case is left for a branch, so the v_elseif and its predicate-complement
        // disappear.
        v = sfpi::copysgn(sfpi::vFloat(1.0f), v);
        v_if(a == 0.0F) { v = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
