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
// LUT-based erf via piecewise rational P(x)/Q(x)
//
// BF16: n14/d14, 1 segment(s), range [-10.0, 10.0]
// FP32: n16/d16, 1 segment(s), range [-10.0, 10.0]
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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_erf() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        dst_reg[0] =
            piecewise_rational_eval<ERF_NUM_DEGREE, ERF_DEN_DEGREE, ERF_NUM_SEGMENTS, ERF_LUT_SIZE, true>(ERF_LUT, x);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erf_init() {
    sfpu_reciprocal_init();
}

}  // namespace sfpu
}  // namespace ckernel
