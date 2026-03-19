// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

// Parity: odd num / even den → x²-Horner (~2x speedup)
#define RATIONAL_NUM_PARITY_ODD
#define RATIONAL_DEN_PARITY_EVEN

#include "ckernel_sfpu_piecewise_rational.h"


namespace ckernel::sfpu {

// ======================================================================
// LUT-based i1 via piecewise rational P(x)/Q(x)
//
// BF16: n5/d4, 1 segment(s), range [-10.0, 10.0]
// FP32: n14/d14, 1 segment(s), range [-10.0, 10.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t I1_NUM_DEGREE = 14;
constexpr uint32_t I1_DEN_DEGREE = 14;
constexpr uint32_t I1_NUM_SEGMENTS = 1;
constexpr uint32_t I1_LUT_SIZE = 32;
constexpr std::array<float, 32> I1_LUT = {{
    -1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f, 5.0000000000e-01f, 0.0000000000e+00f,
    5.6819390506e-02f, 0.0000000000e+00f, 1.9247245509e-03f, 0.0000000000e+00f, 2.8397364076e-05f,
    0.0000000000e+00f, 2.0916867527e-07f, 0.0000000000e+00f, 7.7937084564e-10f, 0.0000000000e+00f,
    1.2293555930e-12f, 0.0000000000e+00f, 1.0000000000e+00f, 0.0000000000e+00f, -1.1361218989e-02f,
    0.0000000000e+00f, 6.1268139689e-05f, 0.0000000000e+00f, -1.9771712800e-07f, 0.0000000000e+00f,
    3.8127551116e-10f, 0.0000000000e+00f, -3.1218170410e-13f, 0.0000000000e+00f, -3.0635529988e-16f,
    0.0000000000e+00f, 7.4301498523e-19f
}};

#else

constexpr uint32_t I1_NUM_DEGREE = 5;
constexpr uint32_t I1_DEN_DEGREE = 4;
constexpr uint32_t I1_NUM_SEGMENTS = 1;
constexpr uint32_t I1_LUT_SIZE = 13;
constexpr std::array<float, 13> I1_LUT = {{
    -1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f, 5.0793987513e-01f, 0.0000000000e+00f,
    4.6934813261e-02f, 0.0000000000e+00f, 2.4524198379e-03f, 1.0000000000e+00f, 0.0000000000e+00f,
    -1.5700012445e-02f, 0.0000000000e+00f, 6.8308552727e-05f
}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i1() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = piecewise_rational_eval<I1_NUM_DEGREE, I1_DEN_DEGREE, I1_NUM_SEGMENTS, I1_LUT_SIZE>(I1_LUT, x);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void i1_init() {
    sfpu_reciprocal_init();
}

}  // namespace ckernel::sfpu
