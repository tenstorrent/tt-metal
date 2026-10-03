// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
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
                    // This region already excludes +/-0 and +inf, so a bare sign test is enough.
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
                    // This region already excludes +/-0 and +inf, so a bare sign test is enough.
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
            // Keep this as `<`, not `!=`: it skips positive NaN, which would otherwise run the
            // step and come back sign-flipped (a bf16 pack then makes that -inf). Negative NaN
            // does run the step -- the compare wraps -- but the clamp below rewrites it.
            v_if(sfpi::as<sfpi::vInt>(x) < infinity_bits) { y = one_minus_xyy * half_xy + xy; }
            v_endif;
        }
    }

    // All edge handling is gated on !FAST_APPROX, as the negative clamp alone was before: the
    // fast path trades every edge guard for speed, so there a negative is unclamped and
    // sqrt(-0) is +0. Everything below is a FAST_APPROX=false claim.
    if constexpr (!FAST_APPROX) {
        // `x < 0.0f` is a sign-bit test, so it claims -0.0 as well; zero magnitudes are kept
        // out of it and answered separately.
        if constexpr (!RECIPROCAL) {
            // rsqrt's clamp is inside the refinement guard above, where the predicate already
            // excludes +/-0 and +inf.
            // sqrt(+/-0) = +/-0: return x, because the refinement cannot produce a signed zero.
            v_if(_bits_without_sign_(x) == 0) { y = x; }
            v_elseif(x < 0.0f) {
                y = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
            }
            v_endif;
        }
    }

    return y;
}

// The edge handling of the accurate reciprocal body for one vector; keep it in step with the RECIPROCAL branch of
// _calculate_sqrt_body_.
template <bool FAST_APPROX>
sfpi_inline void _sqrt_accurate_reciprocal_edge_(
    const sfpi::vFloat x, sfpi::vFloat& y, const sfpi::vFloat one_minus_xyy, const sfpi::vInt infinity_minus_x_bits) {
    // If x != inf and x has a non-zero magnitude.
    v_if(infinity_minus_x_bits != 0 && _bits_without_sign_(x) != 0) {
        sfpi::vFloat half_y = sfpi::addexp(y, -1);
        y = one_minus_xyy * half_y + y;
        if constexpr (!FAST_APPROX) {
            // This region already excludes +/-0 and +inf, so a bare sign test is enough.
            v_if(x < 0.0f) {
                y = std::numeric_limits<float>::quiet_NaN();  // nan for fp32, inf for bf16
            }
            v_endif;
        }
    }
    // Otherwise x = +/-0 gives +/-inf (the subtraction carries x's sign), x = inf gives 0.
    v_else { y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits); }
    v_endif;
}

// The second refinement step and edge handling of the accurate body for one vector, as in _calculate_sqrt_body_; the
// integer statements sit between the dependent float steps so that no result is read by the next instruction.
template <bool RECIPROCAL, bool FAST_APPROX>
sfpi_inline sfpi::vFloat _sqrt_accurate_second_step_(const sfpi::vFloat x, sfpi::vFloat y) {
    sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
    sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);
    sfpi::vFloat xy = x * y;

    if constexpr (RECIPROCAL) {
        sfpi::vInt x_bits = sfpi::as<sfpi::vInt>(x);
        sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
        sfpi::vFloat negative_y = -y;
        sfpi::vFloat one_minus_xyy = 1.0f + (negative_y * xy);
        _sqrt_accurate_reciprocal_edge_<FAST_APPROX>(x, y, one_minus_xyy, infinity_minus_x_bits);
    } else {
        sfpi::vFloat negative_y = -y;
        sfpi::vFloat one_minus_xyy = 1.0f + (negative_y * xy);
        // If x == inf, we need to skip to avoid y = inf - inf = nan; y will already be inf.
        // Keep this as `<`, not `!=`: it skips positive NaN, which would otherwise run the
        // step and come back sign-flipped (a bf16 pack then makes that -inf). Negative NaN
        // does run the step -- the compare wraps -- but the clamp below rewrites it.
        v_if(sfpi::as<sfpi::vInt>(x) < infinity_bits) {
            sfpi::vFloat half_xy = 0.5f * xy;
            y = one_minus_xyy * half_xy + xy;
        }
        v_endif;

        if constexpr (!FAST_APPROX) {
            // sqrt(+/-0) = +/-0: return x, because the refinement cannot produce a signed zero.
            v_if(_bits_without_sign_(x) == 0) { y = x; }
            v_elseif(x < 0.0f) {
                y = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
            }
            v_endif;
        }
    }

    return y;
}

// The seed and the first refinement step of the accurate body for two vectors, interleaved step by step.
sfpi_inline void _sqrt_accurate_first_step_x2_(
    const sfpi::vFloat x0, const sfpi::vFloat x1, sfpi::vFloat& y0, sfpi::vFloat& y1) {
    sfpi::vInt i0 = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x0) >> 1);
    sfpi::vInt i1 = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x1) >> 1);
    y0 = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i0);
    y1 = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i1);

    // Algorithm SQRT_23-bits, with modifications for reciprocal: the first step.
    sfpi::vFloat xy0 = x0 * y0;
    sfpi::vFloat xy1 = x1 * y1;
    sfpi::vFloat negative_y0 = -y0;
    sfpi::vFloat negative_y1 = -y1;
    sfpi::vFloat c0 = negative_y0 * xy0;
    sfpi::vFloat c1 = negative_y1 * xy1;
    sfpi::vFloat t0 = sfpi::vConstFloatPrgm2 + c0;
    sfpi::vFloat t1 = sfpi::vConstFloatPrgm2 + c1;
    t0 = sfpi::vConstFloatPrgm1 + c0 * t0;
    t1 = sfpi::vConstFloatPrgm1 + c1 * t1;
    y0 = y0 * t0;
    y1 = y1 * t1;
}

// The accurate body for two vectors at once, interleaved so that no instruction reads the result of the one before it;
// per lane the operations and their order are those of _calculate_sqrt_body_<false, RECIPROCAL, FAST_APPROX>.
template <bool RECIPROCAL, bool FAST_APPROX, class LoadX0, class LoadX1>
sfpi_inline void _calculate_sqrt_body_accurate_x2_(LoadX0 load_x0, LoadX1 load_x1, sfpi::vFloat& y0, sfpi::vFloat& y1) {
    if constexpr (RECIPROCAL) {
        const sfpi::vFloat x0 = load_x0();
        const sfpi::vFloat x1 = load_x1();
        _sqrt_accurate_first_step_x2_(x0, x1, y0, y1);

        sfpi::vFloat xy0 = x0 * y0;
        sfpi::vFloat xy1 = x1 * y1;
        sfpi::vFloat negative_y0 = -y0;
        sfpi::vFloat one_minus_xyy0 = 1.0f + (negative_y0 * xy0);
        sfpi::vFloat negative_y1 = -y1;
        sfpi::vFloat one_minus_xyy1 = 1.0f + (negative_y1 * xy1);

        {
            sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
            sfpi::vInt infinity_minus_x0_bits = sfpi::as<sfpi::vInt>(infinity) - sfpi::as<sfpi::vInt>(x0);
            _sqrt_accurate_reciprocal_edge_<FAST_APPROX>(x0, y0, one_minus_xyy0, infinity_minus_x0_bits);
        }
        {
            sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
            sfpi::vInt infinity_minus_x1_bits = sfpi::as<sfpi::vInt>(infinity) - sfpi::as<sfpi::vInt>(x1);
            _sqrt_accurate_reciprocal_edge_<FAST_APPROX>(x1, y1, one_minus_xyy1, infinity_minus_x1_bits);
        }
    } else {
        {
            const sfpi::vFloat x0 = load_x0();
            const sfpi::vFloat x1 = load_x1();
            _sqrt_accurate_first_step_x2_(x0, x1, y0, y1);
        }
        // The sqrt form re-reads x from DEST for the second step; the barriers keep the first read from being held.
        asm volatile("" ::: "memory");
        y0 = _sqrt_accurate_second_step_<false, FAST_APPROX>(load_x0(), y0);
        asm volatile("" ::: "memory");
        y1 = _sqrt_accurate_second_step_<false, FAST_APPROX>(load_x1(), y1);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool RECIPROCAL, bool FAST_APPROX>
inline void _calculate_sqrt_internal_() {
    if constexpr (APPROXIMATION_MODE || (ITERATIONS % 2) != 0) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat tmp = _calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL, FAST_APPROX>(sfpi::dst_reg[0]);
            if constexpr (!fp32_dest_acc_en) {
                tmp = sfpi::convert<sfpi::vFloat16b>(tmp, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = tmp;
            sfpi::dst_reg++;
        }
    } else {
        // Two vectors per step so that the refinement chains overlap; per lane the results are unchanged.
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d += 2) {
            sfpi::vFloat y0;
            sfpi::vFloat y1;
            _calculate_sqrt_body_accurate_x2_<RECIPROCAL, FAST_APPROX>(
                [] { return sfpi::vFloat(sfpi::dst_reg[0]); }, [] { return sfpi::vFloat(sfpi::dst_reg[1]); }, y0, y1);
            if constexpr (!fp32_dest_acc_en) {
                y0 = sfpi::convert<sfpi::vFloat16b>(y0, sfpi::RoundMode::Nearest);
                y1 = sfpi::convert<sfpi::vFloat16b>(y1, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = y0;
            sfpi::dst_reg[1] = y1;
            sfpi::dst_reg += 2;
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool FAST_APPROX>
inline void calculate_sqrt() {
    _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, false, FAST_APPROX>();
}

template <bool APPROXIMATION_MODE>
void sqrt_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vConstIntPrgm0 = 0x5f0b3892;
        sfpi::vConstFloatPrgm1 = 1.89099014875f;
    } else {
        sfpi::vConstIntPrgm0 = 0x5f1110a0;
        sfpi::vConstFloatPrgm1 = 2.2825186f;
        sfpi::vConstFloatPrgm2 = 2.2533049f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
