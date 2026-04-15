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
// BF16: degree-9 on [-5, 0], inline Horner (Sollya remez, err=1.37e-6)
// FP32: degree-14 on [-10, 0], inline Horner (err < 1e-13)
// Evaluated on x_rescaled = x/alpha.  Saturation clamp via vec_min_max.
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
// BF16: degree-9 minimax coefficients for exp(x)-1 on [-5, 0]
constexpr uint32_t CELU_DEGREE = 9;
constexpr float CELU_COEFFS[] = {
    -4.9142539638e-07f,
    9.9997937679e-01f,
    4.9985671043e-01f,
    1.6627873480e-01f,
    4.1127938777e-02f,
    7.8961579129e-03f,
    1.1664814083e-03f,
    1.2462615268e-04f,
    8.4452276496e-06f,
    2.6786776175e-07f};
constexpr float CELU_CLAMP_LO = -5.0f;
#endif

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_celu(uint32_t param0, uint32_t param1) {
    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x_rescaled = alpha_recip * x;
        // Clamp x_rescaled to [CELU_CLAMP_LO, +inf) before polynomial eval (branchless)
        sfpi::vFloat lo = CELU_CLAMP_LO;
        sfpi::vec_min_max(lo, x_rescaled);  // x_rescaled = max(x_rescaled, CELU_CLAMP_LO)
        sfpi::vFloat result = alpha * piecewise_poly_eval<CELU_DEGREE>(CELU_COEFFS, x_rescaled);
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
