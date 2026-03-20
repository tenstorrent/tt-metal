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
// LUT-based lgamma via piecewise rational P(x)/Q(x)
//
// BF16: n9/d9, 1 segment(s), range [0.001, 100.0]
// FP32: n9/d9, 1 segment(s), range [0.001, 100.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t LGAMMA_NUM_DEGREE = 9;
constexpr uint32_t LGAMMA_DEN_DEGREE = 9;
constexpr uint32_t LGAMMA_NUM_SEGMENTS = 1;
constexpr uint32_t LGAMMA_LUT_SIZE = 22;
constexpr std::array<float, 22> LGAMMA_LUT = {
    {1.0000000000e-03f,  1.0000000000e+02f,  -1.9806508332e-01f, -4.6424515243e+01f, -8.0140756701e+02f,
     -1.3223679165e+03f, 2.3922525054e+03f,  6.8231552410e+02f,  -9.9722845441e+02f, 5.0856851399e+01f,
     4.1120346978e+01f,  1.0812873344e+00f,  -2.4906110974e-02f, -1.0309750142e+01f, -3.2291669484e+02f,
     -1.7247930100e+03f, -1.7459626192e+03f, 1.0000000000e+00f,  1.8876048732e+02f,  1.8191631958e+01f,
     2.3192116199e-01f,  -1.6444029461e-04f}};

#else

constexpr uint32_t LGAMMA_NUM_DEGREE = 9;
constexpr uint32_t LGAMMA_DEN_DEGREE = 9;
constexpr uint32_t LGAMMA_NUM_SEGMENTS = 1;
constexpr uint32_t LGAMMA_LUT_SIZE = 22;
constexpr std::array<float, 22> LGAMMA_LUT = {
    {1.0000000000e-03f,  1.0000000000e+02f,  -1.9806508332e-01f, -4.6424515243e+01f, -8.0140756701e+02f,
     -1.3223679165e+03f, 2.3922525054e+03f,  6.8231552410e+02f,  -9.9722845441e+02f, 5.0856851399e+01f,
     4.1120346978e+01f,  1.0812873344e+00f,  -2.4906110974e-02f, -1.0309750142e+01f, -3.2291669484e+02f,
     -1.7247930100e+03f, -1.7459626192e+03f, 1.0000000000e+00f,  1.8876048732e+02f,  1.8191631958e+01f,
     2.3192116199e-01f,  -1.6444029461e-04f}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void lgamma() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        dst_reg[0] =
            piecewise_rational_eval<LGAMMA_NUM_DEGREE, LGAMMA_DEN_DEGREE, LGAMMA_NUM_SEGMENTS, LGAMMA_LUT_SIZE>(
                LGAMMA_LUT, x);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void lgamma_init() {
    sfpu_reciprocal_init();
}

}  // namespace sfpu
}  // namespace ckernel
