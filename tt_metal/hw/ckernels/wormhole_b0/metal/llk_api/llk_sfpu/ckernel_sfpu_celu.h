// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

// Adaptive per-segment degree — reduces Horner steps for low-degree segments
#define HAS_SEGMENT_DEGREES
constexpr uint32_t SEGMENT_DEGREES[] = {0, 11, 1, 1};

#include "ckernel_sfpu_piecewise_polynomial.h"


namespace ckernel::sfpu {

// ======================================================================
// LUT-based celu via piecewise polynomial P(x)
//
// BF16: n11/d0, 4 segment(s), range [-10.0, 10.0]
// ======================================================================

constexpr uint32_t CELU_NUM_DEGREE = 11;
constexpr uint32_t CELU_NUM_SEGMENTS = 4;
constexpr uint32_t CELU_LUT_SIZE = 53;
constexpr std::array<float, 53> CELU_LUT = {{
    -1.0000000000e+01f, -5.0000000000e+00f, 0.0000000000e+00f, 5.0000000000e+00f, 1.0000000000e+01f,
    -9.9660831690e-01f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 1.0000000000e+00f, 4.9999964237e-01f,
    1.6666349769e-01f, 4.1656695306e-02f, 8.3174258471e-03f, 1.3739520218e-03f, 1.8949422520e-04f,
    2.1270112484e-05f, 1.8079401798e-06f, 1.0120993466e-07f, 2.7328819208e-09f, 0.0000000000e+00f,
    1.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 1.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
    0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f
}};

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_celu(uint32_t param0, uint32_t param1) {
    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x_orig = sfpi::dst_reg[0];
        sfpi::vFloat result = x_orig;  // positive passthrough
        v_if(x_orig < 0.0f) {
            sfpi::vFloat x = alpha_recip * x_orig;  // x/alpha
            result = alpha * piecewise_polynomial_eval<CELU_NUM_DEGREE, CELU_NUM_SEGMENTS, CELU_LUT_SIZE>(CELU_LUT, x);
        }
        v_endif;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void celu_init() {
}

}  // namespace ckernel::sfpu
