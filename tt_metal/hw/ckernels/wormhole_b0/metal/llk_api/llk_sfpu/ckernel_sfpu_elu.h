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
// LUT-based elu via piecewise polynomial P(x)
//
// BF16: n11/d0, 4 segment(s), range [-10.0, 10.0]
// ======================================================================

constexpr uint32_t ELU_NUM_DEGREE = 11;
constexpr uint32_t ELU_NUM_SEGMENTS = 4;
constexpr uint32_t ELU_LUT_SIZE = 53;
constexpr std::array<float, 53> ELU_LUT = {{
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

// Boundary clamping: elu(x) → -alpha as x→-∞, elu(x) = x for x≥0

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_elu(uint slope) {
    sfpi::vFloat alpha = Converter::as_float(slope);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = alpha * piecewise_polynomial_eval<ELU_NUM_DEGREE, ELU_NUM_SEGMENTS, ELU_LUT_SIZE>(ELU_LUT, x);
        v_if(x >= 0.0f) { result = x; }
        v_endif;
        v_if(x < -10.0f) { result = -alpha; }
        v_endif;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void elu_init() {
}

}  // namespace ckernel::sfpu
