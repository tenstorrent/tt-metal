// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

// Adaptive per-segment degree — reduces Horner steps for low-degree segments
#define HAS_SEGMENT_DEGREES
#ifdef INP_FLOAT32
constexpr uint32_t SEGMENT_DEGREES[] = {5, 9, 1, 1};
#else
constexpr uint32_t SEGMENT_DEGREES[] = {5, 9, 1, 1};
#endif

#include "ckernel_sfpu_piecewise_polynomial.h"


namespace ckernel::sfpu {

// ======================================================================
// LUT-based selu via piecewise polynomial P(x)
//
// BF16: n9/d0, 4 segment(s), range [-10.0, 10.0]
// FP32: n9/d0, 4 segment(s), range [-10.0, 10.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t SELU_NUM_DEGREE = 9;
constexpr uint32_t SELU_NUM_SEGMENTS = 4;
constexpr uint32_t SELU_LUT_SIZE = 45;
constexpr std::array<float, 45> SELU_LUT = {{
    -1.0000000000e+01f, -5.0000000000e+00f, 0.0000000000e+00f, 5.0000000000e+00f, 1.0000000000e+01f,
    -1.2325942399e+00f, 2.9522360124e-01f, 6.7486839242e-02f, 7.8074470499e-03f, 4.5529933620e-04f,
    1.0674313713e-05f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 1.7580974102e+00f, 8.7899565697e-01f, 2.9275542498e-01f, 7.2747394443e-02f,
    1.4138808474e-02f, 2.1381909028e-03f, 2.3641152075e-04f, 1.6692752979e-05f, 5.5284021983e-07f,
    0.0000000000e+00f, 1.0507009874e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 1.0507009874e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f
}};

#else

constexpr uint32_t SELU_NUM_DEGREE = 9;
constexpr uint32_t SELU_NUM_SEGMENTS = 4;
constexpr uint32_t SELU_LUT_SIZE = 45;
constexpr std::array<float, 45> SELU_LUT = {{
    -1.0000000000e+01f, -5.0000000000e+00f, 0.0000000000e+00f, 5.0000000000e+00f, 1.0000000000e+01f,
    -1.2325942399e+00f, 2.9522360124e-01f, 6.7486839242e-02f, 7.8074470499e-03f, 4.5529933620e-04f,
    1.0674313713e-05f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 1.7580974102e+00f, 8.7899565697e-01f, 2.9275542498e-01f, 7.2747394443e-02f,
    1.4138808474e-02f, 2.1381909028e-03f, 2.3641152075e-04f, 1.6692752979e-05f, 5.5284021983e-07f,
    0.0000000000e+00f, 1.0507009874e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 1.0507009874e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f
}};

#endif

// Boundary clamping: selu(x) → -lambda*alpha ≈ -1.758 as x→-∞, selu(x) = lambda*x for x≥0

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS>
inline void calculate_selu(uint scale, uint alpha) {
    sfpi::vFloat scale_val = Converter::as_float(scale);
    sfpi::vFloat alpha_val = Converter::as_float(alpha);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = scale_val * alpha_val * piecewise_polynomial_eval<SELU_NUM_DEGREE, SELU_NUM_SEGMENTS, SELU_LUT_SIZE>(SELU_LUT, x);
        v_if(x >= 0.0f) { result = scale_val * x; }
        v_endif;
        v_if(x < -10.0f) { result = -1.7580993408e+00f; }
        v_endif;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void selu_init() {
}

}  // namespace ckernel::sfpu
