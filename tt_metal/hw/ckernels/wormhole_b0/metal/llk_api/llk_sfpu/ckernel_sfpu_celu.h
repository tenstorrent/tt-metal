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
// LUT-based celu via minimax polynomial P(x) ≈ exp(x)-1
//
// BF16: Cody-Waite range reduction + degree-5 expm1 (factored Horner)
// FP32: degree-14 on [-10, 0], inline Horner
// Evaluated on x_rescaled = x/alpha.
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t CELU_DEGREE = 14;
constexpr float CELU_COEFFS[] = {
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
constexpr float CELU_CLAMP_LO = -10.0f;
#else
// BF16: Cody-Waite expm1 constants (same as ELU)
constexpr float CELU_EXPM1_H0 = 1.0000000000e+00f;
constexpr float CELU_EXPM1_H1 = 4.9999371171e-01f;
constexpr float CELU_EXPM1_H2 = 1.6666433215e-01f;
constexpr float CELU_EXPM1_H3 = 4.1875664145e-02f;
constexpr float CELU_EXPM1_H4 = 8.3751315251e-03f;
#endif

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_celu(uint32_t param0, uint32_t param1) {
    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x_rescaled = alpha_recip * x;
#ifdef INP_FLOAT32
        sfpi::vFloat orig_xr = x_rescaled;
        sfpi::vFloat lo = CELU_CLAMP_LO;
        sfpi::vec_min_max(lo, x_rescaled);
        sfpi::vFloat result = alpha * piecewise_poly_eval<CELU_DEGREE>(CELU_COEFFS, x_rescaled);
        v_if(x >= 0.0f) { result = x; }
        v_endif;
        v_if(orig_xr < CELU_CLAMP_LO) { result = -alpha; }
        v_endif;
#else
        // BF16: Cody-Waite range reduction on x_rescaled
        constexpr float INV_LN2 = 1.4426950408889634f;
        constexpr float NEG_LN2_HI = -0.6931152343750000f;
        constexpr float NEG_LN2_LO = -3.19461832987e-05f;
        const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);

        // Clamp to prevent exponent underflow (k < -127 wraps setexp)
        sfpi::vFloat cw_xr = x_rescaled;
        sfpi::vFloat lo_cw = -87.0f;
        sfpi::vec_min_max(lo_cw, cw_xr);

        sfpi::vFloat tmp = cw_xr * INV_LN2 + c231;
        sfpi::vInt k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);
        sfpi::vFloat k_f = tmp - c231;
        sfpi::vFloat r = k_f * NEG_LN2_HI + cw_xr;
        r = r + k_f * NEG_LN2_LO;

        sfpi::vFloat h = CELU_EXPM1_H4;
        h = h * r + CELU_EXPM1_H3;
        h = h * r + CELU_EXPM1_H2;
        h = h * r + CELU_EXPM1_H1;
        h = h * r + CELU_EXPM1_H0;
        sfpi::vFloat p = r * h;

        sfpi::vFloat two_k = sfpi::setexp(sfpi::vConst1, sfpi::exexp_nodebias(sfpi::vConst1) + k_int);
        sfpi::vFloat result = (two_k - sfpi::vConst1) + two_k * p;
        result = alpha * result;

        v_if(x >= 0.0f) { result = x; }
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
void celu_init() {}

}  // namespace ckernel::sfpu
