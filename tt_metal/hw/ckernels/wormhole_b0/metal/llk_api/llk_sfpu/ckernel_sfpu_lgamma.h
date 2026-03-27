// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_piecewise_rational.h"


namespace ckernel::sfpu {

// ======================================================================
// LUT-based lgamma via piecewise rational P(x)/Q(x)
//
// BF16: n11/d10, 1 segment(s), range [0.001, 100.0]
// FP32: n11/d10, 1 segment(s), range [0.001, 100.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t LGAMMA_NUM_DEGREE = 11;
constexpr uint32_t LGAMMA_DEN_DEGREE = 10;
constexpr uint32_t LGAMMA_NUM_SEGMENTS = 1;
constexpr uint32_t LGAMMA_LUT_SIZE = 25;
constexpr std::array<float, 25> LGAMMA_LUT = {
    {1.0000000000e-03f,  1.0000000000e+02f,  6.1552302588e-01f,  3.7864482763e+02f,  2.6323933533e+04f,
     3.2911032348e+05f,  5.2394239199e+05f,  -9.4690992096e+05f, -3.2854427432e+05f, 4.3195436796e+05f,
     -1.5172413760e+04f, -2.0299018931e+04f, -7.8122579297e+02f, -3.4237057964e+00f, 7.2137855261e-02f,
     6.8175694161e+01f,  6.9665720606e+03f,  1.4462326457e+05f,  7.0495521862e+05f,  7.2947280946e+05f,
     1.0000000000e+00f,  -8.9455702065e+04f, -1.0102667064e+04f, -2.0632527757e+02f, -5.3398544548e-01f}};

#else

constexpr uint32_t LGAMMA_NUM_DEGREE = 11;
constexpr uint32_t LGAMMA_DEN_DEGREE = 10;
constexpr uint32_t LGAMMA_NUM_SEGMENTS = 1;
constexpr uint32_t LGAMMA_LUT_SIZE = 25;
constexpr std::array<float, 25> LGAMMA_LUT = {
    {1.0000000000e-03f,  1.0000000000e+02f,  6.1552302588e-01f,  3.7864482763e+02f,  2.6323933533e+04f,
     3.2911032348e+05f,  5.2394239199e+05f,  -9.4690992096e+05f, -3.2854427432e+05f, 4.3195436796e+05f,
     -1.5172413760e+04f, -2.0299018931e+04f, -7.8122579297e+02f, -3.4237057964e+00f, 7.2137855261e-02f,
     6.8175694161e+01f,  6.9665720606e+03f,  1.4462326457e+05f,  7.0495521862e+05f,  7.2947280946e+05f,
     1.0000000000e+00f,  -8.9455702065e+04f, -1.0102667064e+04f, -2.0632527757e+02f, -5.3398544548e-01f}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void lgamma() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = piecewise_rational_eval<LGAMMA_NUM_DEGREE, LGAMMA_DEN_DEGREE, LGAMMA_NUM_SEGMENTS, LGAMMA_LUT_SIZE>(LGAMMA_LUT, x);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void lgamma_init() {
    sfpu_reciprocal_init();
}

}  // namespace ckernel::sfpu
