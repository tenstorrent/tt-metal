// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_piecewise_rational.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ======================================================================
// LUT-based erfc via piecewise rational P(x)/Q(x)
//
// BF16: n4/d4, 3 segment(s), range [-5.0, 5.0]
// FP32: n4/d4, 3 segment(s), range [-5.0, 5.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t ERFC_NUM_DEGREE = 4;
constexpr uint32_t ERFC_DEN_DEGREE = 4;
constexpr uint32_t ERFC_NUM_SEGMENTS = 3;
constexpr uint32_t ERFC_LUT_SIZE = 34;
constexpr std::array<float, 34> ERFC_LUT = {
    {-5.0000000000e+00f, -1.6666666667e+00f, 1.6666666667e+00f,  5.0000000000e+00f,  8.3347129822e-01f,
     1.9954245090e+00f,  1.8438844681e+00f,  7.7074432373e-01f,  1.2812392414e-01f,  4.1895556450e-01f,
     1.0000000000e+00f,  9.2282271385e-01f,  3.8552197814e-01f,  6.4071461558e-02f,  1.0008066893e+00f,
     -2.0714581013e+00f, 1.6268670559e+00f,  -5.7597428560e-01f, 7.7643394470e-02f,  1.0000000000e+00f,
     -9.3980979919e-01f, 5.7149255276e-01f,  -3.2921802998e-01f, 6.5222814679e-02f,  -9.8275504570e-06f,
     8.1833359218e-06f,  -2.5547028599e-06f, 3.5436926282e-07f,  -1.8428352178e-08f, -5.0887596607e-01f,
     1.0000000000e+00f,  -7.3091888428e-01f, 2.3553080857e-01f,  -2.8261199594e-02f}};

#else

constexpr uint32_t ERFC_NUM_DEGREE = 4;
constexpr uint32_t ERFC_DEN_DEGREE = 4;
constexpr uint32_t ERFC_NUM_SEGMENTS = 3;
constexpr uint32_t ERFC_LUT_SIZE = 34;
constexpr std::array<float, 34> ERFC_LUT = {
    {-5.0000000000e+00f, -1.6666666667e+00f, 1.6666666667e+00f,  5.0000000000e+00f,  8.3347129822e-01f,
     1.9954245090e+00f,  1.8438844681e+00f,  7.7074432373e-01f,  1.2812392414e-01f,  4.1895556450e-01f,
     1.0000000000e+00f,  9.2282271385e-01f,  3.8552197814e-01f,  6.4071461558e-02f,  1.0008066893e+00f,
     -2.0714581013e+00f, 1.6268670559e+00f,  -5.7597428560e-01f, 7.7643394470e-02f,  1.0000000000e+00f,
     -9.3980979919e-01f, 5.7149255276e-01f,  -3.2921802998e-01f, 6.5222814679e-02f,  -9.8275504570e-06f,
     8.1833359218e-06f,  -2.5547028599e-06f, 3.5436926282e-07f,  -1.8428352178e-08f, -5.0887596607e-01f,
     1.0000000000e+00f,  -7.3091888428e-01f, 2.3553080857e-01f,  -2.8261199594e-02f}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_erfc() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        dst_reg[0] =
            piecewise_rational_eval<ERFC_NUM_DEGREE, ERFC_DEN_DEGREE, ERFC_NUM_SEGMENTS, ERFC_LUT_SIZE>(ERFC_LUT, x);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfc_init() {
    sfpu_reciprocal_init();
}

}  // namespace sfpu
}  // namespace ckernel
