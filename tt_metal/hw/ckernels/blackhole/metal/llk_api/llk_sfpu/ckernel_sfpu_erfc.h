// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// ============================================================================================================
// FP32: erfc(x) = 1 - erf(x)
//               = (2 / sqrt(pi)) * integral_x^inf exp(-w^2) dw
//
// Let a = min(|x|, 5). For negative inputs, erfc(-a) = 2 - erfc(a) so only a in [0, 5] is evaluated.
// Since exp(-a^2) decays rapidly over the interval [2, 5) (about 10 orders of magnitude), we can factor out exp(-a^2)
// so that we only need to fit the remaining part which changes gently.
//
// Formula:
//     erfc(a) = exp(-a^2) * t * Q(t - c),      Q is a degree-9 polynomial and t = 1 / (1 + a/2)
//
// To get the above result, we define w = a + v/(2a) and dw = dv/(2a), this helps us factor out exp(-a^2).
// On simplification, we get
//     erfc(a) = exp(-a^2) / (a * sqrt(pi)) * integral_0^inf exp(-v - v^2/(4a^2)) dv
// Since 1/a is unbounded in [0, 5], we define t = 1 / (1 + a/2).
//     a = 0 => t = 1
//     a = 5 => t = 2/7  -> t is a bounded variable in [2/7, 1]
// This lets us define a function Q(t-c) such that
//     erfc(a) ~= exp(-a^2) * t * Q(t - c),     where c = 323/512 = 0.630859375.
// The polynomial is centred at c so its coefficients are small and well-conditioned.
// Since c is a multiple of 2^-9, t - c is exact in fp32 for t in [0.25, 1).
// c = 323/512 was picked so that rounding c0 and c1 to fp32 costs at most 0.007 and 0.11 ULP.
//
// This code also uses u = 1/(2 + a) = t/2, which saves the multiply by 1/2. Since 2u = t exactly:
//     R = t * Q(t - c) = u * Q2(2u - c),     where Q2 = 2*Q (ERFC_Q_COEFFS are the coefficients of Q2)
//
// STEPS PER LANE:
//   1. a = min(|x|, 5)
//   2. u = 1/(2 + a) using an exact split of the denominator and a residual correction (erfc_recip_two_plus_a)
//   3. split a^2 = hi + lo for error reduction (hi is exactly representable in FP32 and lo retains the residual),
//      then compute exp(-(hi+lo)) = 2^j * exp(f) by Cody-Waite range reduction:
//        j = the integer nearest to -hi * log2(e)   (rounded, not floored), computed from hi alone.
//        f = -hi - j*ln2 - lo                       (a remainder in natural-log units, |f| <~ ln2/2 + |lo|),
//                                                   lo is folded into f afterwards (see erfc_exp_neg_a2).
//   4. Q2 by Horner's rule in s = 2u - c, degree 9
//   5. r = (e * u) * Q2 = erfc(a)
//   6. erfc(x) = sg * r + (1 - sg),  sg = copysign(1, x):  r for x >= 0, 2 - r for x < 0
// ============================================================================================================

constexpr float ERFC_Q_CENTER = 6.308593750e-01f;
constexpr std::array<float, 10> ERFC_Q_COEFFS = {
    1.221113920e+00f,
    1.678530216e+00f,
    1.193264365e+00f,
    8.548802882e-02f,
    -4.240161776e-01f,
    -3.436279296e-02f,
    2.180175782e-01f,
    -4.714965820e-02f,
    -1.016845703e-01f,
    6.689453124e-02f};

// Q2 by Horner's rule: c[0] + s*(c[1] + s*(c[2] + ... + s*c[9]))
sfpi_inline sfpi::vFloat erfc_q2_poly(sfpi::vFloat s) {
    sfpi::vFloat q = ERFC_Q_COEFFS[ERFC_Q_COEFFS.size() - 1];
#pragma GCC unroll 16
    for (int i = static_cast<int>(ERFC_Q_COEFFS.size()) - 2; i >= 2; i--) {
        q = q * s + ERFC_Q_COEFFS[i];
    }
    q = q * s + sfpi::vConstFloatPrgm1;  // c1
    q = q * s + sfpi::vConstFloatPrgm2;  // c0
    return q;
}

constexpr std::uint32_t ERFC_A_HI_MASK = 0xFFFFF000U;
constexpr float ERFC_CLAMP = 5.0f;

// Split a^2 = hi + lo:
//   a_hi = a with the low 12 mantissa bits cleared -> 12 significant bits (1 implicit + 11 stored)
//   hi   = a_hi * a_hi -> a 12-bit by 12-bit product needs 24 bits, exact in fp32
//   lo   = (a - a_hi) * (a + a_hi) ~= a^2 - a_hi^2     (a - a_hi is exact; |lo| < 0.02 for a <= 5)
sfpi_inline void erfc_square_split(sfpi::vFloat a, sfpi::vFloat& hi, sfpi::vFloat& lo) {
    sfpi::vFloat ah = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(a) & ERFC_A_HI_MASK);
    hi = ah * ah;
    lo = (a - ah) * (a + ah);
}

// Constants of exp(f), |f| <~ 0.37.  They are the fp16-friendly refit of _sfpu_exp_fp32_accurate_ coefficients.
constexpr float ERFC_EXP_LOG2E = 1.442382812e+00f;
constexpr float ERFC_EXP_NEG_LN2_HI = -6.933593750e-01f;
constexpr float ERFC_EXP_NEG_LN2_LO = 2.121944417e-04f;  // -(ln2 - 0.693359375)
constexpr float ERFC_EXP_P0 = 1.390457153e-03f;
constexpr float ERFC_EXP_P1 = 8.377075195e-03f;
constexpr float ERFC_EXP_C2 = 4.166688485e-02f;
constexpr float ERFC_EXP_C3 = 1.666643702e-01f;
constexpr float ERFC_EXP_C4 = 0.5f;

// exp(-(hi + lo)):
// x = -(hi+lo) is in [-25, 0] (smallest result exp(-25) = 1.4e-11)
// 1. Cody-Waite range reduction to compute the exp of a smaller value:
//    We want to represent x = j*ln2 + f so that exp(x) = 2^j * exp(f) with integer j and |f| <~ ln2/2 (+ |lo|).
//    We can rearrange this to get
//        j ~ round(-hi * log2(e))              integer in [-36, 0]
//        f ~ -hi - j*ln2_hi - j*ln2_lo - lo,   ln2 = ln2_hi + ln2_lo is closely representable in FP32.
// 2. Taylor expand exp(f) and keep the 1 + f term outside so that only a small part is rounded:
//        exp(f) = 1 + f + f^2 * rho(f)
//    where rho(f) = (e^f - 1 - f)/f^2 = 1/2 + f/6 + f^2/24 + ... varies slowly, so it is fitted by a polynomial in f
//    (Horner coefficients P0, P1, C2, C3, C4).
// 3. Multiplying by 2^j in fp representation is just 1.m * 2^(E + j - 127) which is equivalent to adding j.
//    So  2^j * exp(f) is equivalent to adding j to the biased exponent of exp(f).
sfpi_inline sfpi::vFloat erfc_exp_neg_a2(sfpi::vFloat hi, sfpi::vFloat lo) {
    sfpi::vFloat arg = -hi;
    // Rounding j to nearest integer
    constexpr float MAGIC = 12582912.0f;  // 1.5 * 2^23
    sfpi::vFloat tmp = arg * ERFC_EXP_LOG2E + MAGIC;
    sfpi::vFloat j = tmp - MAGIC;
    // f = -hi - j*ln2 - lo
    sfpi::vFloat f = j * ERFC_EXP_NEG_LN2_HI + arg;
    f = j * ERFC_EXP_NEG_LN2_LO + f;
    f = f - lo;
    // rho(f) by Horner
    sfpi::vFloat r = ERFC_EXP_P0;
    r = r * f + ERFC_EXP_P1;
    r = r * f + ERFC_EXP_C2;
    r = r * f + ERFC_EXP_C3;
    r = r * f + ERFC_EXP_C4;
    sfpi::vFloat y = r * f + 1.0f;
    r = y * f + 1.0f;
    return sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(r) + (sfpi::as<sfpi::vInt>(tmp) << 23));
}

// u = t/2 = 1/(2 + a) where t = 1/(1 + a/2):
//   denom   = 2 + a = d_hi + d_lo           d_hi = float(a + 2);  d_lo = a + (2 - d_hi) is the exact rounding error
//   u0      = 1/d_hi (to ~14 bits)          hardware estimate (~7 bits) plus one Newton step  u <- u * (2 - d_hi * u)
//   eps     = u0 * d_hi - 1 + u0 * d_lo     = u0 * denom - 1, the true residual (each multiply-add rounds once)
//   u       = u0 - u0 * eps                 = u0 * (1 - eps)
// Since 1/denom = u0/(1 + eps) = u0 * (1 - eps + eps^2 - ...), the remaining error is eps^2 ~ 2^-28, below fp32
// rounding. This residual step replaces the second Newton iteration.
sfpi_inline sfpi::vFloat erfc_recip_two_plus_a(sfpi::vFloat a) {
    sfpi::vFloat d_hi = a + sfpi::vConstFloatPrgm0;
    sfpi::vFloat d_lo = a + (sfpi::vConstFloatPrgm0 - d_hi);
    // // d_hi in [2, 7] is finite and non-zero. The [1, 2) in sfpu_reciprocal_iter's doc is stricter than this path
    // needs.
    sfpi::vFloat u0 = sfpu_reciprocal_iter</*max_iter=*/1, /*normalized=*/true>(d_hi);
    sfpi::vFloat res = u0 * d_hi - 1.0f;
    res = u0 * d_lo + res;
    return u0 - u0 * res;
}

inline bool bf16_dest_erfc();
template <int ITERATIONS>
inline void calculate_erfc_bf16();
// Whether BF16 DEST runs the generated erfc kernel as one call over the whole tile.
inline constexpr bool erfc_bf16_whole_tile = true;

template <int ITERATIONS = 8, bool is_fp32_dest_acc_en = true>
inline void calculate_erfc() {
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 32) {
        if (bf16_dest_erfc()) {
            calculate_erfc_bf16<ITERATIONS>();
            return;
        }
    }
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        // Clamp x >= 5 returns erfc(5) and x <= -5 returns 2 - erfc(5) = 2
        sfpi::vFloat a = sfpi::min(sfpi::abs(x0), ERFC_CLAMP);
        // u = 1/(2 + a)
        sfpi::vFloat u = erfc_recip_two_plus_a(a);
        sfpi::vFloat hi, lo;
        // a^2 = hi + lo
        erfc_square_split(a, hi, lo);
        // e = exp(-(hi + lo))
        sfpi::vFloat e = erfc_exp_neg_a2(hi, lo);
        // s = 2u - c = t - c
        sfpi::vFloat q = erfc_q2_poly(u * sfpi::vConstFloatPrgm0 - ERFC_Q_CENTER);
        sfpi::vFloat r = (e * u) * q;
        // r for x >= 0, 2 - r for x < 0
        sfpi::vFloat sg = sfpi::copysgn(sfpi::vFloat(1.0f), x0);
        r = sg * r + (1.0f - sg);
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

inline void init_erfc_bf16();

template <bool APPROXIMATION_MODE>
void erfc_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init</*APPROXIMATE=*/false>();  // vConstFloatPrgm0 is 2.0
    sfpi::vConstFloatPrgm1 = ERFC_Q_COEFFS[1];  // vConstFloatPrgm1 is c1 of Q2
    sfpi::vConstFloatPrgm2 = ERFC_Q_COEFFS[0];  // vConstFloatPrgm2 is c0 of Q2
}

}  // namespace ckernel::sfpu

#include "ckernel_sfpu_erfc_bf16.h"
