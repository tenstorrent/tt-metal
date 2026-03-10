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
// ======================================================================

constexpr uint32_t I1_NUM_DEGREE = 5;
constexpr uint32_t I1_DEN_DEGREE = 4;
constexpr uint32_t I1_NUM_SEGMENTS = 1;
constexpr uint32_t I1_LUT_SIZE = 13;
constexpr std::array<float, 13> I1_LUT = {{
    -1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f, 5.0793987513e-01f, 0.0000000000e+00f,
    4.6934813261e-02f, 0.0000000000e+00f, 2.4524198379e-03f, 1.0000000000e+00f, 0.0000000000e+00f,
    -1.5700012445e-02f, 0.0000000000e+00f, 6.8308552727e-05f
}};

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i1() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = piecewise_rational_eval<I1_NUM_DEGREE, I1_DEN_DEGREE, I1_NUM_SEGMENTS, I1_LUT_SIZE>(I1_LUT, x);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void i1_init() {
    sfpu_reciprocal_init();
}

}  // namespace ckernel::sfpu
