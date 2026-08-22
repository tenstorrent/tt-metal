// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Legacy-compat rsqrt: the compat square-root approximation (_sqrt_compat_, which is what
// "compat" is actually about) paired with the modern reciprocal. _reciprocal_compat_ is not
// used: its exponent-difference arithmetic has no pole guard, so 1/sqrt(0) came out as
// 1.7e38 instead of inf. sfpu_reciprocal_iter builds its scale factor as ~in.Exp precisely
// so both poles fall out for free. See FIX_PLAN_52930_reciprocal_compat_pole.md.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en>
inline void _calculate_rsqrt_compat_iter_(const int iterations) {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        // Round-tripped through Dest, as the legacy kernel did: on a 16-bit Dest the
        // square root is narrowed to bf16 before the reciprocal sees it.
        sfpi::dst_reg[0] = _sqrt_compat_<APPROXIMATION_MODE, 2>(sfpi::dst_reg[0]);
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat out;

        // Iteration count and the bf16 narrowing mirror _calculate_reciprocal_internal_.
        // No explicit sign flip for in < 0: sfpu_reciprocal_iter already ends in
        // copysgn(y, in), which is the same result the legacy negate produced.
        if constexpr (APPROXIMATION_MODE) {
            out = sfpu_reciprocal_iter<0>(in);
        } else if constexpr (fp32_dest_acc_en) {
            out = sfpu_reciprocal_iter<2>(in);
        } else {
            out = sfpu_reciprocal_iter<1>(in);
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat>
inline void calculate_rsqrt() {
    if constexpr (legacy_compat) {
        _calculate_rsqrt_compat_iter_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(ITERATIONS);
    } else {
        _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, true, FAST_APPROX>();
    }
}

template <bool APPROXIMATION_MODE, bool legacy_compat>
void rsqrt_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (legacy_compat) {
        // The compat path now runs sfpu_reciprocal_iter, whose polynomial coefficients live
        // in vConstFloatPrgm0..2, so they have to be programmed here too.
        sfpu_reciprocal_init<APPROXIMATION_MODE>();
    } else {
        sqrt_init<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
