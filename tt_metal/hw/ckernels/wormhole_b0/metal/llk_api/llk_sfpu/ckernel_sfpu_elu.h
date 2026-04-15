// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// ======================================================================
// ELU via Cody-Waite range reduction + factored expm1 polynomial
//
// Algorithm: x = k*ln(2) + r, |r| <= ln(2)/2
//   expm1(r) = r * h(r), h(r) = minimax degree 5 on [-ln2/2, ln2/2]
//   exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
//
// BF16 h degree 4: max abs error = 1.60e-7 (Sollya remez)
// FP32 h degree 5: max abs error = 8.67e-9 (Sollya remez)
// ======================================================================

// Cody-Waite constants
constexpr float CW_INV_LN2 = 1.4426950408889634f;
constexpr float CW_NEG_LN2_HI = -0.6931152343750000f;
constexpr float CW_NEG_LN2_LO = -3.19461832987e-05f;

// expm1(r) = r * h(r) coefficients
#ifdef INP_FLOAT32
// FP32: h degree 5 (max abs error = 8.67e-9)
constexpr uint32_t EXPM1_H_DEGREE = 5;
constexpr float EXPM1_H[] = {
    1.0000000000e+00f, 5.0000000000e-01f, 1.6666504741e-01f, 4.1666239500e-02f, 8.3691505715e-03f, 1.3948583510e-03f};
#else
// BF16: h degree 4 (max abs error = 1.60e-7)
constexpr uint32_t EXPM1_H_DEGREE = 4;
constexpr float EXPM1_H[] = {
    1.0000000000e+00f, 4.9999371171e-01f, 1.6666433215e-01f, 4.1875664145e-02f, 8.3751315251e-03f};
#endif

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_elu(uint slope) {
    sfpi::vFloat alpha = Converter::as_float(slope);
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat orig_x = sfpi::dst_reg[0];

        // Clamp to prevent exponent underflow (k < -127 wraps setexp)
        sfpi::vFloat cw_x = orig_x;
        sfpi::vFloat lo = -87.0f;
        sfpi::vec_min_max(lo, cw_x);

        // Cody-Waite range reduction: x = k*ln(2) + r
        sfpi::vFloat tmp = cw_x * CW_INV_LN2 + c231;
        sfpi::vInt k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);
        sfpi::vFloat k_f = tmp - c231;
        sfpi::vFloat r = k_f * CW_NEG_LN2_HI + cw_x;
        r = r + k_f * CW_NEG_LN2_LO;

        // expm1(r) = r * h(r), Horner evaluation of h
        sfpi::vFloat h = EXPM1_H[EXPM1_H_DEGREE];
        for (int i = static_cast<int>(EXPM1_H_DEGREE) - 1; i >= 0; i--) {
            h = h * r + EXPM1_H[i];
        }
        sfpi::vFloat p = r * h;

        // Reconstruct: exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
        sfpi::vFloat two_k = sfpi::setexp(sfpi::vConst1, sfpi::exexp_nodebias(sfpi::vConst1) + k_int);
        sfpi::vFloat result = (two_k - sfpi::vConst1) + two_k * p;
        result = alpha * result;

        v_if(orig_x >= 0.0f) { result = orig_x; }
        v_endif;

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
