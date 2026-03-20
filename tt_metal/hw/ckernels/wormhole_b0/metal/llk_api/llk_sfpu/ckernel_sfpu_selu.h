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
constexpr uint32_t SEGMENT_DEGREES[] = {14, 1};
#else
constexpr uint32_t SEGMENT_DEGREES[] = {0, 11, 1, 1};
#endif

#include "ckernel_sfpu_piecewise_polynomial.h"


namespace ckernel::sfpu {

// ======================================================================
// LUT-based selu via piecewise polynomial P(x)
//
// BF16: n11/d0, 4 segment(s), range [-10.0, 10.0]
// FP32: n14/d0, 2 segment(s), range [-10.0, 10.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t SELU_NUM_DEGREE = 14;
constexpr uint32_t SELU_NUM_SEGMENTS = 2;
constexpr uint32_t SELU_LUT_SIZE = 33;
constexpr std::array<float, 33> SELU_LUT = {{
    -1.0000000000e+01f, 0.0000000000e+00f, 1.0000000000e+01f, 0.0000000000e+00f, 1.0000000000e+00f,
    4.9999934435e-01f, 1.6666224599e-01f, 4.1655816138e-02f, 8.3194561303e-03f, 1.3780959416e-03f,
    1.9285458256e-04f, 2.2803982574e-05f, 2.2368417376e-06f, 1.7566813426e-07f, 1.0487319457e-08f,
    4.4169295998e-10f, 1.1582759057e-11f, 1.4128406561e-13f, 0.0000000000e+00f, 1.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f
}};

#else

constexpr uint32_t SELU_NUM_DEGREE = 11;
constexpr uint32_t SELU_NUM_SEGMENTS = 4;
constexpr uint32_t SELU_LUT_SIZE = 53;
constexpr std::array<float, 53> SELU_LUT = {{
    -1.0000000000e+01f, -5.0000000000e+00f, 0.0000000000e+00f, 5.0000000000e+00f, 1.0000000000e+01f,
    -9.9660831690e-01f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 1.0000000000e+00f, 4.9999964237e-01f,
    1.6666349769e-01f, 4.1656695306e-02f, 8.3174258471e-03f, 1.3739520218e-03f, 1.8949422520e-04f,
    2.1270112484e-05f, 1.8079401798e-06f, 1.0120993466e-07f, 2.7328819208e-09f, 0.0000000000e+00f,
    1.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 1.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f
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
