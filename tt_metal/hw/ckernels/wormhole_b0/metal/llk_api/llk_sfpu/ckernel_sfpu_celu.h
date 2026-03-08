// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ======================================================================
// LUT-based celu via piecewise rational P(x)/Q(x)
//
// BF16: n11/d0, 4 segment(s), range [-10.0, 10.0]
// FP32: n14/d0, 2 segment(s), range [-10.0, 10.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t CELU_NUM_DEGREE = 14;
constexpr uint32_t CELU_NUM_SEGMENTS = 2;
constexpr uint32_t CELU_LUT_SIZE = 33;
constexpr std::array<float, 33> CELU_LUT = {
    {-1.0000000000e+01f, 0.0000000000e+00f, 1.0000000000e+01f, 0.0000000000e+00f, 1.0000000000e+00f, 4.9999934435e-01f,
     1.6666224599e-01f,  4.1655816138e-02f, 8.3194561303e-03f, 1.3780959416e-03f, 1.9285458256e-04f, 2.2803982574e-05f,
     2.2368417376e-06f,  1.7566813426e-07f, 1.0487319457e-08f, 4.4169295998e-10f, 1.1582759057e-11f, 1.4128406561e-13f,
     0.0000000000e+00f,  1.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
     0.0000000000e+00f,  0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
     0.0000000000e+00f,  0.0000000000e+00f, 0.0000000000e+00f}};

#else

constexpr uint32_t CELU_NUM_DEGREE = 11;
constexpr uint32_t CELU_NUM_SEGMENTS = 4;
constexpr uint32_t CELU_LUT_SIZE = 53;
constexpr std::array<float, 53> CELU_LUT = {
    {-1.0000000000e+01f, -5.0000000000e+00f, 0.0000000000e+00f, 5.0000000000e+00f, 1.0000000000e+01f,
     -9.9660831690e-01f, 0.0000000000e+00f,  0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
     0.0000000000e+00f,  0.0000000000e+00f,  0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
     0.0000000000e+00f,  0.0000000000e+00f,  0.0000000000e+00f, 1.0000000000e+00f, 4.9999964237e-01f,
     1.6666349769e-01f,  4.1656695306e-02f,  8.3174258471e-03f, 1.3739520218e-03f, 1.8949422520e-04f,
     2.1270112484e-05f,  1.8079401798e-06f,  1.0120993466e-07f, 2.7328819208e-09f, 0.0000000000e+00f,
     1.0000000000e+00f,  0.0000000000e+00f,  0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
     0.0000000000e+00f,  0.0000000000e+00f,  0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
     0.0000000000e+00f,  0.0000000000e+00f,  1.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
     0.0000000000e+00f,  0.0000000000e+00f,  0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
     0.0000000000e+00f,  0.0000000000e+00f,  0.0000000000e+00f}};

#endif

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_celu(uint32_t param0, uint32_t param1) {
    vFloat alpha = Converter::as_float(param0);
    vFloat alpha_recip = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x_orig = dst_reg[0];
        vFloat x = alpha_recip * x_orig;  // x/alpha
        vFloat result = piecewise_polynomial_eval<CELU_NUM_DEGREE, CELU_NUM_SEGMENTS, CELU_LUT_SIZE>(CELU_LUT, x);
        v_if(x_orig < 0.0f) { result = alpha * result; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void celu_init() {}

}  // namespace sfpu
}  // namespace ckernel
