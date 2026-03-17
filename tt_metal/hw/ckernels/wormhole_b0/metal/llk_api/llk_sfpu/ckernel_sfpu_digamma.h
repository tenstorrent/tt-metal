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
// LUT-based digamma via piecewise rational P(x)/Q(x)
//
// BF16: n5/d4, 1 segment(s), range [0.01, 10.0]
// FP32: n5/d4, 2 segment(s), range [0.01, 10.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t DIGAMMA_NUM_DEGREE = 5;
constexpr uint32_t DIGAMMA_DEN_DEGREE = 4;
constexpr uint32_t DIGAMMA_NUM_SEGMENTS = 2;
constexpr uint32_t DIGAMMA_LUT_SIZE = 25;
constexpr std::array<float, 25> DIGAMMA_LUT = {
    {1.0000000000e-02f,  5.0050000000e+00f, 1.0000000000e+01f, -7.0329284668e-01f, -1.4059219360e+00f,
     2.5364404917e-01f,  5.9145379066e-01f, 8.0310069025e-02f, 4.4234484085e-04f,  6.9550951665e-09f,
     7.0329165459e-01f,  1.0000000000e+00f, 3.2575941086e-01f, 2.1074503660e-02f,  -1.2833554745e+00f,
     -6.1750292778e-01f, 7.6493072510e-01f, 1.6943506896e-01f, 5.2741440013e-03f,  6.0327092797e-06f,
     8.5594326258e-02f,  1.0000000000e+00f, 5.3459835052e-01f, 5.5676151067e-02f,  1.0372793768e-03f}};

#else

constexpr uint32_t DIGAMMA_NUM_DEGREE = 5;
constexpr uint32_t DIGAMMA_DEN_DEGREE = 4;
constexpr uint32_t DIGAMMA_NUM_SEGMENTS = 1;
constexpr uint32_t DIGAMMA_LUT_SIZE = 13;
constexpr std::array<float, 13> DIGAMMA_LUT = {
    {1.0000000000e-02f,
     1.0000000000e+01f,
     -6.7370545864e-01f,
     -1.3893005848e+00f,
     1.9062051177e-01f,
     6.0996395350e-01f,
     8.5371270776e-02f,
     4.2144476902e-04f,
     -2.1869135480e-07f,
     6.7372852564e-01f,
     1.0000000000e+00f,
     3.4258455038e-01f,
     2.2140301764e-02f}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_digamma() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        dst_reg[0] =
            piecewise_rational_eval<DIGAMMA_NUM_DEGREE, DIGAMMA_DEN_DEGREE, DIGAMMA_NUM_SEGMENTS, DIGAMMA_LUT_SIZE>(
                DIGAMMA_LUT, x);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void digamma_init() {
    sfpu_reciprocal_init();
}

}  // namespace sfpu
}  // namespace ckernel
