// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "ckernel_sfpu_piecewise_rational.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ======================================================================
// LUT-based erf via piecewise rational P(x)/Q(x)
// BF16: n14/d14, 1 seg, range [-10.0, 10.0]
// FP32: n16/d16, 1 seg, range [-10.0, 10.0]
//
// LUT-based erfc via piecewise rational P(x)/Q(x)
// BF16: n4/d4, 3 seg, range [-5.0, 5.0]
// FP32: n4/d4, 3 seg, range [-5.0, 5.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t ERF_NUM_DEGREE = 16;
constexpr uint32_t ERF_DEN_DEGREE = 16;
constexpr uint32_t ERF_NUM_SEGMENTS = 1;
constexpr uint32_t ERF_LUT_SIZE = 36;
constexpr std::array<float, 36> ERF_LUT = {
    {-1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f,  1.1283791065e+00f,  0.0000000000e+00f,
     2.1477432549e-01f,  0.0000000000e+00f, 6.2133435160e-02f,  0.0000000000e+00f,  5.6230435148e-03f,
     0.0000000000e+00f,  6.1307044234e-04f, 0.0000000000e+00f,  1.7678321456e-05f,  0.0000000000e+00f,
     2.7384647439e-08f,  0.0000000000e+00f, -2.8632063387e-10f, 0.0000000000e+00f,  1.0000000000e+00f,
     0.0000000000e+00f,  5.2367275953e-01f, 0.0000000000e+00f,  1.2961706519e-01f,  0.0000000000e+00f,
     1.9642570987e-02f,  0.0000000000e+00f, 1.9545555115e-03f,  0.0000000000e+00f,  1.3179056987e-04f,
     0.0000000000e+00f,  1.3156344494e-06f, 0.0000000000e+00f,  -3.5153888689e-09f, 0.0000000000e+00f,
     -6.7350725691e-12f}};

#else

constexpr uint32_t ERF_NUM_DEGREE = 14;
constexpr uint32_t ERF_DEN_DEGREE = 14;
constexpr uint32_t ERF_NUM_SEGMENTS = 1;
constexpr uint32_t ERF_LUT_SIZE = 32;
constexpr std::array<float, 32> ERF_LUT = {
    {-1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f, 1.1283791065e+00f, 0.0000000000e+00f, 2.2798484564e-01f,
     0.0000000000e+00f,  6.2783762813e-02f, 0.0000000000e+00f, 6.0815168545e-03f, 0.0000000000e+00f, 5.6875223527e-04f,
     0.0000000000e+00f,  1.4256307622e-05f, 0.0000000000e+00f, 4.9761702314e-08f, 0.0000000000e+00f, 1.0000000000e+00f,
     0.0000000000e+00f,  5.3537726402e-01f, 0.0000000000e+00f, 1.3411131501e-01f, 0.0000000000e+00f, 2.0345563069e-02f,
     0.0000000000e+00f,  2.0084537100e-03f, 0.0000000000e+00f, 1.1187216296e-04f, 0.0000000000e+00f, 1.1231509234e-06f,
     0.0000000000e+00f,  9.4874041956e-10f}};

#endif

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

template <bool APPROXIMATION_MODE>
inline void calculate_erf() {
    for (int d = 0; d < 8; d++) {
        vFloat x = dst_reg[0];
        dst_reg[0] =
            piecewise_rational_eval<ERF_NUM_DEGREE, ERF_DEN_DEGREE, ERF_NUM_SEGMENTS, ERF_LUT_SIZE>(ERF_LUT, x);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfc() {
    for (int d = 0; d < 8; d++) {
        vFloat x = dst_reg[0];
        dst_reg[0] =
            piecewise_rational_eval<ERFC_NUM_DEGREE, ERFC_DEN_DEGREE, ERFC_NUM_SEGMENTS, ERFC_LUT_SIZE>(ERFC_LUT, x);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
