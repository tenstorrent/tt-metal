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
constexpr uint32_t SEGMENT_DEGREES[] = {0, 2, 1};
#else
constexpr uint32_t SEGMENT_DEGREES[] = {0, 2, 1};
#endif

#include "ckernel_sfpu_piecewise_polynomial.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ======================================================================
// LUT-based hardmish via piecewise polynomial P(x)
//
// BF16: n2/d0, 3 segment(s), range [-10.0, 10.0]
// FP32: n2/d0, 3 segment(s), range [-10.0, 10.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t HARDMISH_NUM_DEGREE = 2;
constexpr uint32_t HARDMISH_NUM_SEGMENTS = 3;
constexpr uint32_t HARDMISH_LUT_SIZE = 13;
constexpr std::array<float, 13> HARDMISH_LUT = {{
    -1.0000000000e+01f, -2.0000000000e+00f, 0.0000000000e+00f, 1.0000000000e+01f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 1.0000000000e+00f, 5.0000000000e-01f,
    0.0000000000e+00f, 1.0000000000e+00f, 0.0000000000e+00f
}};

#else

constexpr uint32_t HARDMISH_NUM_DEGREE = 2;
constexpr uint32_t HARDMISH_NUM_SEGMENTS = 3;
constexpr uint32_t HARDMISH_LUT_SIZE = 13;
constexpr std::array<float, 13> HARDMISH_LUT = {{
    -1.0000000000e+01f, -2.0000000000e+00f, 0.0000000000e+00f, 1.0000000000e+01f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 1.0000000000e+00f, 5.0000000000e-01f,
    0.0000000000e+00f, 1.0000000000e+00f, 0.0000000000e+00f
}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void hardmish() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        dst_reg[0] = piecewise_polynomial_eval<HARDMISH_NUM_DEGREE, HARDMISH_NUM_SEGMENTS, HARDMISH_LUT_SIZE>(HARDMISH_LUT, x);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void hardmish_init() {
}

}  // namespace sfpu
}  // namespace ckernel
