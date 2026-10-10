// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// State the intended single-rounding multiply-add explicitly. With an immediate
// coefficient, the ordinary a * b + c spelling prefers SFPMULI/SFPADDI (two
// dependent roundings) over SFPMAD. I0's polynomials admit the fused rounding,
// as verified by their numeric contract, and naming it here keeps that semantic
// choice local rather than requiring a translation-unit fast-math option.
sfpi_inline vFloat i0_mad_(vFloat a, vFloat b, float c) {
    const vFloat addend = c;
    return __builtin_rvtt_sfpmad(a.get(), b.get(), addend.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}
inline void i0_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// i0's Maclaurin series above is 11 terms in x^2 and is only useful while the
// series still converges inside fp32: it is 0.03% low at |x| = 12, 5.9% low at 13,
// 1.7% low at 15 and 1.5e16x low at 89, where it returned 1.23e21 for the true
// 1.90e37. An unbounded function cannot be repaired by clamping its argument, so
// the tail is stated by the standard Hankel asymptotic (A&S 9.7.1)
//
//   I0(x) = exp(x)/sqrt(2*pi*x) * (1 + 1/(8x) + 9/(128x^2) + 225/(3072x^3) + ...)
//
// truncated after the x^-4 term (residual 6e-7 at x = 12, far inside the bf16
// contract). Shape, seed constants and the outlined-helper discipline are lifted
// from the sibling kernel ckernel_sfpu_i1.h, whose asymptotic is the same form.
//
// exp(x) itself overflows fp32 at 88.7229 while I0 stays finite to 91.9008, so the
// exponential is taken at x/2 and applied TWICE with the 1/sqrt(x) factor between
// the two multiplies; no intermediate then leaves the normal range.
inline vFloat calculate_i0_asymptotic_(const vFloat abs_x) {
    const vFloat half_abs = 0.5f * abs_x;
#ifdef INP_FLOAT32
    const vFloat exp_half = _sfpu_exp_fp32_accurate_unsafe_(half_abs);
#else
    const vFloat exp_half = _sfpu_exp_21f_bf16_unsafe_<true>(half_abs);
#endif

    // 1/sqrt(|x|): Quake-style magic seed + two Newton refinements (i1's constants).
    const vInt rsqrt_i = sfpi::as<vInt>(sfpi::as<vUInt>(abs_x) >> 1);
    vFloat rsqrt_y     = sfpi::as<vFloat>(vInt(0x5f1110a0) - rsqrt_i);
    vFloat c0          = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * i0_mad_(c0, vFloat(2.2533049f) + c0, 2.2825186f);
    c0 = i0_mad_(-rsqrt_y, abs_x * rsqrt_y, 1.0f);
    rsqrt_y            = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    // 1/|x| = (1/sqrt|x|)^2 — reuses the refined rsqrt instead of a reciprocal.
    const vFloat u = rsqrt_y * rsqrt_y;

    // Q(u) = 1 + u/8 + 9u^2/128 + 225u^3/3072 + 11025u^4/98304, Horner.
    vFloat q = vFloat(0.112152099609375f);
    q = i0_mad_(q, u, 0.0732421875f);
    q = i0_mad_(q, u, 0.0703125f);
    q = i0_mad_(q, u, 0.125f);
    q = i0_mad_(q, u, 1.0f);

    // (exp(x/2) * 1/sqrt(2*pi*x) * Q) * exp(x/2)
    return ((exp_half * vFloat(0.3989422804f)) * rsqrt_y * q) * exp_half;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    // Series is accurate to ~3e-4 at |x| = 12 and diverges from I0 above it; the
    // asymptotic is equal or better from 10 upward, so 12 is a seamless crossover
    // that leaves every in-range lane on the byte-identical series path.
    constexpr float I0_THRESHOLD  = 12.0f;
    // |x| at which I0 reaches FLT_MAX (x - 0.5*ln(2*pi*x) = ln(FLT_MAX)): above it
    // the value is +inf, which the series happened to deliver by overflowing and
    // which the asymptotic must therefore state explicitly.
    constexpr float I0_MAX_FINITE = 91.9008f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = 0.0f;
        vFloat input = dst_reg[0];
        vFloat abs_x = sfpi::abs(input);
        vFloat x = input * input;

        vFloat p = vFloat(1.50E-22f);
        p = i0_mad_(p, x, 7.24E-20f);
        p = i0_mad_(p, x, 2.90E-17f);
        p = i0_mad_(p, x, 9.39E-15f);
        p = i0_mad_(p, x, 2.40E-12f);
        p = i0_mad_(p, x, 4.71E-10f);
        p = i0_mad_(p, x, 6.78E-08f);
        p = i0_mad_(p, x, 0.000006781684028f);
        p = i0_mad_(p, x, 0.0004340277778f);
        p = i0_mad_(p, x, 0.015625f);
        p = i0_mad_(p, x, 0.25f);
        result = i0_mad_(p, x, 1.0f);

        v_if(abs_x > I0_THRESHOLD) {
            // No bound on the argument: every lane with abs_x >= I0_MAX_FINITE
            // (inf, and NaN, which the compares admit) is overwritten just below,
            // and every other lane is already under it.
            result = calculate_i0_asymptotic_(abs_x);
            // abs_x * inf rather than the constant: +inf for every |x| past the
            // overflow point, but a NaN (which the compares above admit) stays NaN.
            v_if(abs_x >= I0_MAX_FINITE) { result = abs_x * vFloat(std::numeric_limits<float>::infinity()); }
            v_endif;
        }
        v_endif;

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
