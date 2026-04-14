// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

namespace {
inline float uint_as_float(uint32_t u) {
    union {
        uint32_t i;
        float f;
    } conv = {u};
    return conv.f;
}
}  // namespace

// softcap(x, cap) = cap * tanh(x / cap)
//
// Three paths:
//   |u| < 0.5:  Taylor degree-13, result = x * P(u²) (avoids cap round-trip)
//   0.5 ≤ |u| < 9: exp-based tanh * cap
//   |u| ≥ 9:    result = ±cap
//
// 5 vFloat LREGs: x_orig, val(u), res, nf, p

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t cap_bits, uint32_t rcap_bits) {
    using namespace sfpi;

    const float cap_f = uint_as_float(cap_bits);
    const float rcap_f = uint_as_float(rcap_bits);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat res = dst_reg[0];            // res = x (will become result)
        vFloat val = res * vFloat(rcap_f);  // val = u = x/cap

        // Default: large |u| → result = ±cap
        vFloat nf = abs(val);              // nf = |u| (temporary, reused later)
        res = setsgn(vFloat(cap_f), val);  // res = sign(u) * cap

        // --- Exp path: 0.5 ≤ |u| < 9 ---
        v_if(nf < 9.0f) {
            vFloat p = nf + nf;  // p = y = 2*|u|

            // Range reduction
            nf = p * 1.4426950408889634f + 12582912.0f;
            nf = nf - 12582912.0f;  // nf = n = round(y/ln2)
            // Cody-Waite two-part subtraction
            p = p - nf * 0.693145751953125f;
            p = p - nf * 1.428606765330187e-6f;  // p = r

            // exp(r) degree-8 Horner
            res = 0.0000248015873015873f;  // 1/40320
            res = res * p + 0.000198412698412698f;
            res = res * p + 0.001388888888888889f;
            res = res * p + 0.008333333333333333f;
            res = res * p + 0.041666666666666664f;
            res = res * p + 0.16666666666666666f;
            res = res * p + 0.5f;
            res = res * p + vConst1;
            res = res * p + vConst1;
            // res = exp(r)

            // exp(y) = res * 2^n
            res = setexp(res, reinterpret<vUInt>(exexp_nodebias(res) + reinterpret<vInt>(float_to_int16(nf, 0))));

            // tanh = (exp-1)/(exp+1)
            nf = res + vConst1;  // nf = den = exp + 1
            p = res - vConst1;   // p = num = exp - 1

            // Newton-Raphson reciprocal of nf (den)
            res = setexp(nf, 127);       // mantissa in [1,2)
            res = 1.5f + res * (-0.5f);  // linear 1/m approx
            res = setexp(res, reinterpret<vUInt>(exexp_nodebias(res) - exexp(nf)));
            res = res * (2.0f - nf * res);  // Newton 1
            res = res * (2.0f - nf * res);  // Newton 2
            res = res * (2.0f - nf * res);  // Newton 3

            res = p * res;                           // tanh = num / den
            res = setsgn(res * vFloat(cap_f), val);  // softcap = ±tanh * cap
        }
        v_endif;

        // --- Taylor path: |u| < 0.5 ---
        // tanh(u) = u * P(u²), degree-13 Taylor
        v_if(abs(val) < 0.5f) {
            nf = val * val;              // nf = u²
            res = 0.00359212803657248f;  // 21844/6081075
            res = res * nf + (-0.00886323552990220f);
            res = res * nf + 0.02186948853615520f;
            res = res * nf + (-0.05396825396825397f);
            res = res * nf + 0.1333333333333333f;
            res = res * nf + (-0.3333333333333333f);
            res = res * nf + vConst1;                           // P(u²) ≈ tanh(u)/u
            res = setsgn(abs(val) * res * vFloat(cap_f), val);  // |u| * P * cap * sign
        }
        v_endif;

        dst_reg[0] = res;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {}

}  // namespace sfpu
}  // namespace ckernel
