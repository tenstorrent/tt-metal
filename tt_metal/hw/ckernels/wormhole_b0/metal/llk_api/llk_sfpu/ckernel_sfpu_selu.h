// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) {
    constexpr float CW_INV_LN2 = 1.4426950408889634f;
    constexpr float CW_NEG_LN2_HI = -0.6931152343750000f;
    constexpr float CW_NEG_LN2_LO = -3.19461832987e-05f;

    const sfpi::vFloat scale_val = Converter::as_float(scale);
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);
    const sfpi::vFloat neg_scale_alpha = sfpi::setsgn(scale_alpha, 1);
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to prevent exponent underflow
        sfpi::vFloat lo = -87.0f;
        sfpi::vec_min_max(lo, x);

        // Cody-Waite range reduction
        sfpi::vFloat tmp = x * CW_INV_LN2 + c231;
        sfpi::vInt k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);
        sfpi::vFloat k_f = tmp - c231;
        sfpi::vFloat r = k_f * CW_NEG_LN2_HI + x;
        r = r + k_f * CW_NEG_LN2_LO;

        // expm1(r) = r * h(r)
#ifdef INP_FLOAT32
        sfpi::vFloat h = PolynomialEvaluator::eval(
            r,
            1.0000000000e+00f,
            5.0000000000e-01f,
            1.6666504741e-01f,
            4.1666239500e-02f,
            8.3691505715e-03f,
            1.3948583510e-03f);
#else
        sfpi::vFloat h = PolynomialEvaluator::eval(
            r, 1.0000000000e+00f, 4.9999371171e-01f, 1.6666433215e-01f, 4.1875664145e-02f, 8.3751315251e-03f);
#endif

        // Reconstruct: exp(x) = 2^k * exp(r), result = scale_alpha*exp(x) - scale_alpha
        sfpi::vFloat exp_r = r * h + sfpi::vConst1;
        sfpi::vFloat exp_x = sfpi::setexp(exp_r, sfpi::exexp_nodebias(exp_r) + k_int);
        sfpi::vFloat result = scale_alpha * exp_x + neg_scale_alpha;

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
