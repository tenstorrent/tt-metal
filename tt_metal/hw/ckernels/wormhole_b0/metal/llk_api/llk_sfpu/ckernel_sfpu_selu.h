// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_piecewise_polynomial.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based selu via minimax polynomial
//
// BF16: Cody-Waite range reduction + degree-5 expm1, scale*alpha post-mul
// FP32: degree-14 for exp(x)-1 on [-10, 0], runtime scale*alpha multiply
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t SELU_DEGREE = 14;
constexpr float SELU_COEFFS[] = {
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
constexpr float SELU_CLAMP_LO = -10.0f;
#else
// BF16: Cody-Waite expm1 constants (same as ELU)
constexpr float SELU_EXPM1_H0 = 1.0000000000e+00f;
constexpr float SELU_EXPM1_H1 = 4.9999371171e-01f;
constexpr float SELU_EXPM1_H2 = 1.6666433215e-01f;
constexpr float SELU_EXPM1_H3 = 4.1875664145e-02f;
constexpr float SELU_EXPM1_H4 = 8.3751315251e-03f;
#endif

// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS>
inline void calculate_selu(uint scale, uint alpha) {
    const sfpi::vFloat scale_val = Converter::as_float(scale);
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
#ifdef INP_FLOAT32
        sfpi::vFloat clamped = x;
        sfpi::vFloat lo = SELU_CLAMP_LO;
        sfpi::vec_min_max(lo, clamped);
        sfpi::vFloat result = scale_alpha * piecewise_poly_eval<SELU_DEGREE>(SELU_COEFFS, clamped);
        v_if(x >= 0.0f) { result = scale_val * x; }
        v_endif;
        v_if(x < SELU_CLAMP_LO) { result = -1.7580993408e+00f; }
        v_endif;
#else
        // BF16: Cody-Waite range reduction + factored expm1
        constexpr float INV_LN2 = 1.4426950408889634f;
        constexpr float NEG_LN2_HI = -0.6931152343750000f;
        constexpr float NEG_LN2_LO = -3.19461832987e-05f;
        const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);

        // Clamp to prevent exponent underflow (k < -127 wraps setexp)
        sfpi::vFloat cw_x = x;
        sfpi::vFloat lo_cw = -87.0f;
        sfpi::vec_min_max(lo_cw, cw_x);

        sfpi::vFloat tmp = cw_x * INV_LN2 + c231;
        sfpi::vInt k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);
        sfpi::vFloat k_f = tmp - c231;
        sfpi::vFloat r = k_f * NEG_LN2_HI + cw_x;
        r = r + k_f * NEG_LN2_LO;

        sfpi::vFloat h = SELU_EXPM1_H4;
        h = h * r + SELU_EXPM1_H3;
        h = h * r + SELU_EXPM1_H2;
        h = h * r + SELU_EXPM1_H1;
        h = h * r + SELU_EXPM1_H0;
        sfpi::vFloat p = r * h;

        sfpi::vFloat two_k = sfpi::setexp(sfpi::vConst1, sfpi::exexp_nodebias(sfpi::vConst1) + k_int);
        sfpi::vFloat result = (two_k - sfpi::vConst1) + two_k * p;
        result = scale_alpha * result;

        v_if(x >= 0.0f) { result = scale_val * x; }
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
void selu_init() {}

}  // namespace ckernel::sfpu
