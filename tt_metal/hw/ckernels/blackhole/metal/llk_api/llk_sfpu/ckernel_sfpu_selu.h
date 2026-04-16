// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// SELU via Cody-Waite range reduction + factored expm1 polynomial
//
// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581
// ======================================================================

// Cody-Waite constants
constexpr float SELU_CW_INV_LN2 = 1.4426950408889634f;
constexpr float SELU_CW_NEG_LN2_HI = -0.6931152343750000f;
constexpr float SELU_CW_NEG_LN2_LO = -3.19461832987e-05f;

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) {
    const sfpi::vFloat scale_val = Converter::as_float(scale);
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to prevent exponent underflow
        // Safe for x >= 0 check: max(x, -87) preserves sign for positive x
        sfpi::vFloat lo = -87.0f;
        sfpi::vec_min_max(lo, x);

        // Cody-Waite range reduction
        sfpi::vFloat tmp = x * SELU_CW_INV_LN2 + c231;
        sfpi::vFloat k_f = tmp - c231;
        sfpi::vFloat r = k_f * SELU_CW_NEG_LN2_HI + x;
        r = r + k_f * SELU_CW_NEG_LN2_LO;

        // expm1(r) = r * h(r)
#ifdef INP_FLOAT32
        sfpi::vFloat h = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,
            5.0000000000e-01f,
            1.6666504741e-01f,
            4.1666239500e-02f,
            8.3691505715e-03f,
            1.3948583510e-03f);
#else
        sfpi::vFloat h = PolynomialEvaluator::eval(
            r, sfpi::vConst1, 4.9999371171e-01f, 1.6666433215e-01f, 4.1875664145e-02f, 8.3751315251e-03f);
#endif
        h = r * h;

        // Reconstruct: exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
        constexpr int kC231Bias = 0x4B3FFF81;
        sfpi::vFloat two_k = sfpi::setexp(sfpi::vConst1, sfpi::reinterpret<sfpi::vInt>(tmp) - kC231Bias);
        sfpi::vFloat result = (two_k - sfpi::vConst1) + two_k * h;
        result = scale_alpha * result;

        v_if(x >= 0.0f) { result = scale_val * x; }
        v_endif;

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
