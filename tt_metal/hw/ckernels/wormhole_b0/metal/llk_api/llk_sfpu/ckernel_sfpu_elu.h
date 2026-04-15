// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_piecewise_polynomial.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based elu via minimax polynomial P(x) ≈ exp(x)-1
//
// BF16: Cody-Waite range reduction + degree-5 expm1 (factored Horner)
//       max abs error = 1.60e-7, target MaxULP ≈ 1-2
// FP32: degree-14 on [-10, 0], inline Horner (err < 1e-13)
// Positive path (x>=0): identity.
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t ELU_DEGREE = 14;
constexpr float ELU_COEFFS[] = {
    0.0000000000e+00f,
    1.0000000000e+00f,
    4.9999934435e-01f,
    1.6666224599e-01f,
    4.1655816138e-02f,
    8.3194561303e-03f,
    1.3780959416e-03f,
    1.9285458256e-04f,
    2.2803982574e-05f,
    2.2368417376e-06f,
    1.7566813426e-07f,
    1.0487319457e-08f,
    4.4169295998e-10f,
    1.1582759057e-11f,
    1.4128406561e-13f};
constexpr float ELU_CLAMP_LO = -10.0f;
#else
// BF16: Cody-Waite expm1 constants
// expm1(r) = r * h(r), h(r) = h0 + h1*r + h2*r^2 + h3*r^3 + h4*r^4
// Minimax on [-ln2/2, ln2/2], max abs error = 1.60e-7 (single-precision)
constexpr float EXPM1_H0 = 1.0000000000e+00f;
constexpr float EXPM1_H1 = 4.9999371171e-01f;
constexpr float EXPM1_H2 = 1.6666433215e-01f;
constexpr float EXPM1_H3 = 4.1875664145e-02f;
constexpr float EXPM1_H4 = 8.3751315251e-03f;
#endif

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_elu(uint slope) {
    sfpi::vFloat alpha = Converter::as_float(slope);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat orig_x = sfpi::dst_reg[0];
#ifdef INP_FLOAT32
        // FP32: clamp + degree-14 polynomial
        sfpi::vFloat x = orig_x;
        sfpi::vFloat lo = ELU_CLAMP_LO;
        sfpi::vec_min_max(lo, x);
        sfpi::vFloat result = alpha * piecewise_poly_eval<ELU_DEGREE>(ELU_COEFFS, x);
        v_if(orig_x >= 0.0f) { result = orig_x; }
        v_endif;
        v_if(orig_x < ELU_CLAMP_LO) { result = -alpha; }
        v_endif;
#else
        // BF16: Cody-Waite range reduction + factored expm1
        // x = k*ln(2) + r, |r| <= ln(2)/2
        constexpr float INV_LN2 = 1.4426950408889634f;
        constexpr float NEG_LN2_HI = -0.6931152343750000f;
        constexpr float NEG_LN2_LO = -3.19461832987e-05f;
        const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);

        sfpi::vFloat tmp = orig_x * INV_LN2 + c231;
        sfpi::vInt k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);
        sfpi::vFloat k_f = tmp - c231;
        sfpi::vFloat r = k_f * NEG_LN2_HI + orig_x;
        r = r + k_f * NEG_LN2_LO;

        // expm1(r) = r * h(r), Horner for h(r) degree 4
        sfpi::vFloat h = EXPM1_H4;
        h = h * r + EXPM1_H3;
        h = h * r + EXPM1_H2;
        h = h * r + EXPM1_H1;
        h = h * r + EXPM1_H0;
        sfpi::vFloat p = r * h;  // expm1(r)

        // Reconstruct: exp(x)-1 = (2^k - 1) + 2^k * p
        sfpi::vFloat two_k = sfpi::setexp(sfpi::vConst1, sfpi::exexp_nodebias(sfpi::vConst1) + k_int);
        sfpi::vFloat result = (two_k - sfpi::vConst1) + two_k * p;
        result = alpha * result;

        v_if(orig_x >= 0.0f) { result = orig_x; }
        v_endif;
#endif
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void elu_init() {}

}  // namespace ckernel::sfpu
