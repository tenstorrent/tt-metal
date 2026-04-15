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
// BF16: degree-10 for scale*alpha*(exp(x)-1) on [-5, 0], pre-baked
//       (Sollya remez, err=2.01e-6 in single precision)
// FP32: degree-14 for exp(x)-1 on [-10, 0], runtime scale*alpha multiply
// Saturation clamp via vec_min_max.
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
// BF16: degree-10 minimax for scale*alpha*(exp(x)-1) on [-5, 0]
// Pre-baked scale*alpha (≈1.758) eliminates runtime multiply.
constexpr uint32_t SELU_DEGREE = 10;
constexpr float SELU_COEFFS[] = {
    -9.6848864928e-08f,
    1.7580944300e+00f,
    8.7900894880e-01f,
    2.9288360476e-01f,
    7.3030054569e-02f,
    1.4428040944e-02f,
    2.3012275342e-03f,
    2.9013518360e-04f,
    2.7009362384e-05f,
    1.6228416371e-06f,
    4.6336307236e-08f};
constexpr float SELU_CLAMP_LO = -5.0f;
#endif

// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS>
inline void calculate_selu(uint scale, uint alpha) {
    const sfpi::vFloat scale_val = Converter::as_float(scale);
#ifdef INP_FLOAT32
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);
#endif
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Clamp input to [SELU_CLAMP_LO, +inf) before polynomial eval (branchless)
        sfpi::vFloat clamped = x;
        sfpi::vFloat lo = SELU_CLAMP_LO;
        sfpi::vec_min_max(lo, clamped);  // clamped = max(x, SELU_CLAMP_LO)
#ifdef INP_FLOAT32
        sfpi::vFloat result = scale_alpha * piecewise_poly_eval<SELU_DEGREE>(SELU_COEFFS, clamped);
#else
        // BF16: polynomial already includes scale*alpha factor
        sfpi::vFloat result = piecewise_poly_eval<SELU_DEGREE>(SELU_COEFFS, clamped);
#endif
        v_if(x >= 0.0f) { result = scale_val * x; }
        v_endif;
        v_if(x < SELU_CLAMP_LO) { result = -1.7580993408e+00f; }
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
