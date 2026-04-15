// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// ======================================================================
// CELU via Cody-Waite range reduction + factored expm1 polynomial
//
// Same algorithm as ELU, evaluated on x_rescaled = x/alpha.
// celu(x) = x for x>=0, alpha*(exp(x/alpha)-1) for x<0
// ======================================================================

// Cody-Waite constants
constexpr float CELU_CW_INV_LN2 = 1.4426950408889634f;
constexpr float CELU_CW_NEG_LN2_HI = -0.6931152343750000f;
constexpr float CELU_CW_NEG_LN2_LO = -3.19461832987e-05f;

#ifdef INP_FLOAT32
constexpr uint32_t CELU_EXPM1_H_DEGREE = 5;
constexpr float CELU_EXPM1_H[] = {
    1.0000000000e+00f, 5.0000000000e-01f, 1.6666504741e-01f, 4.1666239500e-02f, 8.3691505715e-03f, 1.3948583510e-03f};
#else
constexpr uint32_t CELU_EXPM1_H_DEGREE = 4;
constexpr float CELU_EXPM1_H[] = {
    1.0000000000e+00f, 4.9999371171e-01f, 1.6666433215e-01f, 4.1875664145e-02f, 8.3751315251e-03f};
#endif

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_celu(uint32_t param0, uint32_t param1) {
    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat xr = alpha_recip * x;

        // Clamp to prevent exponent underflow
        sfpi::vFloat lo = -87.0f;
        sfpi::vec_min_max(lo, xr);

        // Cody-Waite range reduction on x_rescaled
        sfpi::vFloat tmp = xr * CELU_CW_INV_LN2 + c231;
        sfpi::vInt k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);
        sfpi::vFloat k_f = tmp - c231;
        sfpi::vFloat r = k_f * CELU_CW_NEG_LN2_HI + xr;
        r = r + k_f * CELU_CW_NEG_LN2_LO;

        // expm1(r) = r * h(r)
        sfpi::vFloat h = CELU_EXPM1_H[CELU_EXPM1_H_DEGREE];
        for (int i = static_cast<int>(CELU_EXPM1_H_DEGREE) - 1; i >= 0; i--) {
            h = h * r + CELU_EXPM1_H[i];
        }
        sfpi::vFloat p = r * h;

        // Reconstruct: exp(xr)-1 = (2^k - 1) + 2^k * expm1(r)
        sfpi::vFloat two_k = sfpi::setexp(sfpi::vConst1, sfpi::exexp_nodebias(sfpi::vConst1) + k_int);
        sfpi::vFloat result = (two_k - sfpi::vConst1) + two_k * p;
        result = alpha * result;

        v_if(x >= 0.0f) { result = x; }
        v_endif;

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
