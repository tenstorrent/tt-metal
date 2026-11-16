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
    static_assert(ITERATIONS % 2 == 0, "calculate_sign processes two rows per iteration");
    // Two rows per iteration share one dst_reg increment.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        // SFPSETCC is unspecified for -0.0 (VectorUnit.md), so a bare compare can miss it.
        // SFPSETSGN overwrites its sign operand, so abs comes first to let copysgn reuse
        // v's register without a copy.
        sfpi::vFloat a0 = sfpi::abs(v0);
        sfpi::vFloat a1 = sfpi::abs(v1);
        // copysgn stamps v's sign bit onto 1.0, which is exactly the v < 0 arm. Only the
        // zero case is left for a branch, so the v_elseif and its predicate-complement
        // disappear.
        sfpi::vFloat res0 = sfpi::copysgn(sfpi::vFloat(1.0f), v0);
        sfpi::vFloat res1 = sfpi::copysgn(sfpi::vFloat(1.0f), v1);
        v_if(a0 == 0.0F) { res0 = 0.0f; }
        v_endif;
        v_if(a1 == 0.0F) { res1 = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = res0;
        sfpi::dst_reg[1] = res1;
        sfpi::dst_reg += 2;
    }
}

}  // namespace sfpu
}  // namespace ckernel
