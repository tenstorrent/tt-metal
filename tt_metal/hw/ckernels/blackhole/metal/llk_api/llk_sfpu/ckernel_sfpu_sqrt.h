// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// See: Kokosiński, Z., Gepner, P., Moroz, L. et al.
// Fast and accurate approximation algorithms for computing floating point square root. Numerical Algorithms (2024).
// https://doi.org/10.1007/s11075-024-01932-7

// x's bits with the sign shifted out: `_bits_without_sign_(x) != 0` is a magnitude test that
// treats -0.0 and +0.0 alike, which a bare `as<vInt>(x) != 0` does not.
sfpi_inline sfpi::vInt _bits_without_sign_(const sfpi::vFloat x) { return sfpi::as<sfpi::vInt>(x) << 1; }

// Computes the square root or reciprocal square root of a positive floating point value x.
template <bool APPROXIMATE = false, bool RECIPROCAL = false, bool FAST_APPROX = false>
sfpi_inline sfpi::vFloat _calculate_sqrt_body_(const sfpi::vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);

    if constexpr (APPROXIMATE) {
        // Algorithm SQRT_10-bits, with modifications for reciprocal.
        sfpi::vFloat c = x * y;
        sfpi::vFloat negative_y = -y;
        sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);
        sfpi::vFloat t = sfpi::vConstFloatPrgm1 + negative_y * c;
        if constexpr (RECIPROCAL) {
            sfpi::vInt x_bits = sfpi::as<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            // If x != inf and x has a non-zero magnitude.
            v_if(infinity_minus_x_bits != 0 && _bits_without_sign_(x) != 0) {
                y = y * t;
                if constexpr (!FAST_APPROX) {
                    // Every lane reaching here is sign-negative with a non-zero magnitude:
                    // this region already excludes +/-0 and +inf, so a bare sign test is
                    // enough and the magnitude test does not have to be repeated.
                    v_if(x < 0.0f) {
                        y = std::numeric_limits<float>::quiet_NaN();  // nan for fp32, inf for bf16
                    }
                    v_endif;
                }
            }
            // Otherwise x = +/-0 gives +/-inf (the subtraction carries x's sign), x = inf gives 0.
            v_else { y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits); }
            v_endif;
        } else {
            y = c;
            // If x != inf.  Otherwise, y = inf, since c = inf.
            v_if(sfpi::as<sfpi::vInt>(x) != infinity_bits) { y = y * t; }
            v_endif;
        }
    } else {
        // Algorithm SQRT_23-bits, with modifications for reciprocal.
        sfpi::vFloat xy = x * y;
        sfpi::vFloat negative_y = -y;
        sfpi::vFloat c = negative_y * xy;
        sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);
        y = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));
        xy = x * y;
        negative_y = -y;
        sfpi::vFloat one_minus_xyy = 1.0f + (negative_y * xy);

        if constexpr (RECIPROCAL) {
            sfpi::vFloat half_y = sfpi::addexp(y, -1);
            sfpi::vInt x_bits = sfpi::as<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            // If x != inf and x has a non-zero magnitude.
            v_if(infinity_minus_x_bits != 0 && _bits_without_sign_(x) != 0) {
                y = one_minus_xyy * half_y + y;
                if constexpr (!FAST_APPROX) {
                    // Every lane reaching here is sign-negative with a non-zero magnitude:
                    // this region already excludes +/-0 and +inf, so a bare sign test is
                    // enough and the magnitude test does not have to be repeated.
                    v_if(x < 0.0f) {
                        y = std::numeric_limits<float>::quiet_NaN();  // nan for fp32, inf for bf16
                    }
                    v_endif;
                }
            }
            // Otherwise x = +/-0 gives +/-inf (the subtraction carries x's sign), x = inf gives 0.
            v_else { y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits); }
            v_endif;
        } else {
            sfpi::vFloat half_xy = 0.5f * xy;
            // If x == inf, we need to skip to avoid y = inf - inf = nan; y will already be inf.
            // Keep this as `<` and not `!=`. What `<` buys is skipping *positive* NaN: under
            // `!=` the step would run on it and come back sign-flipped (0xFFD00001, measured on
            // Wormhole), which a bf16 pack then turns into -inf. It does not skip negative NaN --
            // this ordered integer compare is a wrapping subtract-and-test-sign, so 0xFF800001..
            // 0xFFFFFFFF still read as less than infinity_bits and do run the step. That is
            // harmless only because the `x < 0.0f` clamp below rewrites every sign-set lane.
            v_if(sfpi::as<sfpi::vInt>(x) < infinity_bits) { y = one_minus_xyy * half_xy + xy; }
            v_endif;
        }
    }

    // Every edge guard lives under !FAST_APPROX, as the negative clamp alone did before these
    // arms were added: sqrt_tile<true>/rsqrt_tile<true> trade all edge handling for speed, so on
    // that path a negative still returns the unclamped seed and sqrt(-0) is +0. The results
    // claimed below are therefore claims about FAST_APPROX=false -- the default of
    // sqrt_tile/rsqrt_tile, and what the edge sweep and the signed-zero regression test run.
    if constexpr (!FAST_APPROX) {
        // `x < 0.0f` is a sign-bit test, so it claims -0.0 as well; a zero magnitude is kept out
        // of it and answered on its own. Behaviour here is measured rather than specified -- the
        // derivation is with the edge sweep in test_eltwise_unary_sfpu.py.
        if constexpr (!RECIPROCAL) {
            // rsqrt's clamp is not here: it sits inside the refinement guard above, where the
            // predicate already excludes +/-0 and +inf, so it needs no magnitude test of its
            // own. Measured on a WH n150: that placement is worth ~3-4% on Rsqrt.
            // sqrt(+/-0) = +/-0: return x, because the refinement above cannot produce a signed
            // zero. test_sqrt_family_negative_zero_regression is the standing check. Two other
            // shapes were measured on Wormhole and are worse: copysgn plus a strictly-negative
            // `-x > 0.0f` clamp, and folding the zero into the refinement guard's v_else.
            v_if(_bits_without_sign_(x) == 0) { y = x; }
            v_elseif(x < 0.0f) {
                y = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
            }
            v_endif;
        }
    }

    return y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool RECIPROCAL, bool FAST_APPROX>
inline void _calculate_sqrt_internal_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat tmp = _calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL, FAST_APPROX>(sfpi::dst_reg[0]);
        if constexpr (!fp32_dest_acc_en) {
            tmp = sfpi::convert<sfpi::vFloat16b>(tmp, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = tmp;
        sfpi::dst_reg++;
    }
}

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = 8,
    bool fp32_dest_acc_en,
    bool FAST_APPROX,
    bool legacy_compat = false>
inline void calculate_sqrt() {
    if constexpr (legacy_compat) {
        _calculate_sqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(ITERATIONS);
    } else {
        _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, false, FAST_APPROX>();
    }
}

template <bool APPROXIMATION_MODE, bool legacy_compat = false>
void sqrt_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!legacy_compat) {
        if constexpr (APPROXIMATION_MODE) {
            sfpi::vConstIntPrgm0 = 0x5f0b3892;
            sfpi::vConstFloatPrgm1 = 1.89099014875f;
        } else {
            sfpi::vConstIntPrgm0 = 0x5f1110a0;
            sfpi::vConstFloatPrgm1 = 2.2825186f;
            sfpi::vConstFloatPrgm2 = 2.2533049f;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
