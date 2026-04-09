// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// High-precision implementation using three regions:
//
//   Region 1 (|u| < 0.5, u = x/cap):  Taylor series degree-13 (Horner form)
//     tanh(u) = u * (1 + u^2*(c3 + u^2*(c5 + u^2*(c7 + u^2*(c9 + u^2*(c11 + u^2*c13))))))
//
//   Region 2 (0.5 <= |u| < 9.0):  Exp-based formula
//     tanh(|u|) = (exp(2|u|) - 1) / (exp(2|u|) + 1)
//     Uses range-reduced polynomial exp2 and Newton-Raphson reciprocal
//
//   Region 3 (|u| >= 9.0):  Saturation
//     tanh(u) = sign(u), so softcap = sign(x) * cap

// Helper: compute 2^z for z >= 0 using range reduction + degree-7 Taylor polynomial.
// Decomposes z = n + r where n = round(z), |r| <= 0.5.
// Then 2^z = 2^n * 2^r, with 2^r = exp(r*ln2) approximated by polynomial.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat softcap_exp2(sfpi::vFloat z) {
    // Compute 2^z via range reduction: 2^z = 2^n * 2^r, n = round(z), |r| <= 0.5
    // n = floor(z + 0.5) via magic constant trick (valid for z in [0, 2^23))
    sfpi::vInt n_int = sfpi::reinterpret<sfpi::vInt>(z + 8388608.5f) - sfpi::vInt(0x4B000000);

    // r = z - n (fractional part, |r| <= 0.5)
    sfpi::vFloat n_float = sfpi::int32_to_float(n_int, 0);
    sfpi::vFloat r = z - n_float;

    // Direct minimax polynomial for 2^r, r in [-0.5, 0.5]
    // 2^r = 1 + r*C1 + r^2*C2 + ... (Taylor coefficients of 2^r)
    // C_k = (ln2)^k / k!
    constexpr float C1 = 0.6931471805599453f;  // ln2
    constexpr float C2 = 0.2402265069591007f;  // (ln2)^2/2
    constexpr float C3 = 0.0555041086648216f;  // (ln2)^3/6
    constexpr float C4 = 0.0096181291076285f;  // (ln2)^4/24
    constexpr float C5 = 0.0013333558146429f;  // (ln2)^5/120
    constexpr float C6 = 0.0001540353039338f;  // (ln2)^6/720
    constexpr float C7 = 0.0000152525861611f;  // (ln2)^7/5040
    constexpr float C8 = 0.0000013215486790f;  // (ln2)^8/40320

    // Horner form: 2^r = 1 + r*(C1 + r*(C2 + r*(C3 + ...)))
    sfpi::vFloat poly = C8;
    poly = poly * r + C7;
    poly = poly * r + C6;
    poly = poly * r + C5;
    poly = poly * r + C4;
    poly = poly * r + C3;
    poly = poly * r + C2;
    poly = poly * r + C1;
    poly = poly * r + 1.0f;

    // 2^z = 2^n * poly: add n to the exponent
    sfpi::vInt poly_exp = sfpi::exexp(poly);
    return sfpi::setexp(poly, 127U + poly_exp + n_int);
}

// Helper: Newton-Raphson reciprocal for positive values.
// 4 iterations give full fp32 precision (~2^-48 relative error).
inline sfpi::vFloat softcap_reciprocal(sfpi::vFloat val) {
    sfpi::vFloat nval = sfpi::setsgn(val, 1);
    sfpi::vInt orig_exp = sfpi::exexp(val);
    nval = sfpi::setexp(nval, 126);

    sfpi::vFloat ln2_recip = 1.442695f;
    sfpi::vFloat two = 2.0f;
    sfpi::vFloat result = ln2_recip * (nval * ln2_recip + two);

    result = result * (nval * result + two);
    result = result * (nval * result + two);
    result = result * (nval * result + two);
    result = result * (nval * result + two);

    sfpi::vInt new_exp = sfpi::exexp(result);
    new_exp -= orig_exp;
    new_exp += 126;

    v_if(new_exp < 0) {
        result = 0.0f;
        new_exp = 0;
    }
    v_endif;

    return sfpi::setexp(result, new_exp);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0) {
    union {
        uint32_t u;
        float f;
    } conv = {param0};
    const float cap_f = conv.f;
    const float inv_cap_f = 1.0f / cap_f;

    constexpr float log2e = 1.4426950408889634f;

    // Taylor coefficients for tanh degree-13 (Horner form on u^2)
    constexpr float c3 = -1.0f / 3.0f;
    constexpr float c5 = 2.0f / 15.0f;
    constexpr float c7 = -17.0f / 315.0f;
    constexpr float c9 = 62.0f / 2835.0f;
    constexpr float c11 = -1382.0f / 155925.0f;
    constexpr float c13 = 21844.0f / 6081075.0f;

    constexpr float taylor_bound = 0.5f;
    constexpr float sat_bound = 9.0f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat u = x * inv_cap_f;
        sfpi::vFloat abs_u = sfpi::abs(u);
        sfpi::vFloat u2 = u * u;

        // Region 1: Taylor series degree-13 (Horner form)
        sfpi::vFloat poly = c13;
        poly = poly * u2 + c11;
        poly = poly * u2 + c9;
        poly = poly * u2 + c7;
        poly = poly * u2 + c5;
        poly = poly * u2 + c3;
        poly = poly * u2 + 1.0f;
        sfpi::vFloat result = poly * u;

        // Region 2: Exp-based for 0.5 <= |u| < 9.0
        v_if(abs_u >= taylor_bound) {
            sfpi::vFloat z = abs_u * (2.0f * log2e);
            sfpi::vFloat exp_val = softcap_exp2<APPROXIMATION_MODE>(z);
            sfpi::vFloat num = exp_val - 1.0f;
            sfpi::vFloat den = exp_val + 1.0f;
            sfpi::vFloat t = num * softcap_reciprocal(den);
            v_if(u < 0.0f) { t = 0.0f - t; }
            v_endif;
            result = t;
        }
        v_endif;

        // Region 3: Saturation for |u| >= 9.0
        v_if(abs_u >= sat_bound) {
            result = sfpi::vConst1;
            v_if(u < 0.0f) { result = 0.0f - sfpi::vConst1; }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = result * cap_f;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {}

}  // namespace ckernel::sfpu
