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
// LUT-based softplus via piecewise rational P(x)/Q(x)
//
// BF16: n9/d9, 1 segment(s), range [-10.0, 10.0]
// ======================================================================

constexpr uint32_t SOFTPLUS_NUM_DEGREE = 9;
constexpr uint32_t SOFTPLUS_DEN_DEGREE = 9;
constexpr uint32_t SOFTPLUS_NUM_SEGMENTS = 1;
constexpr uint32_t SOFTPLUS_LUT_SIZE = 22;
constexpr std::array<float, 22> SOFTPLUS_LUT = {{
    -1.0000000000e+01f, 1.0000000000e+01f, 6.9314640760e-01f, 3.0334073305e-01f, 1.0266214609e-01f,
    2.5587543845e-02f, 4.1804499924e-03f, 4.4794997666e-04f, 3.1875460991e-05f, 1.4791692138e-06f,
    4.1209883506e-08f, 5.2948567753e-10f, 1.0000000000e+00f, -2.8372007608e-01f, 1.7242786288e-01f,
    -3.6300994456e-02f, 8.6406944320e-03f, -1.1708125239e-03f, 1.2544600759e-04f, -8.4130733740e-06f,
    3.4111849345e-07f, -6.1077645164e-09f
}};

// Boundary clamping: softplus(x) → 0 as x→-∞, softplus(x) → x as x→+∞

template <bool APPROXIMATION_MODE>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat x = beta * sfpi::dst_reg[0];
    v_if(x < threshold) {
        sfpi::dst_reg[0] = beta_reciprocal * piecewise_rational_eval<SOFTPLUS_NUM_DEGREE, SOFTPLUS_DEN_DEGREE, SOFTPLUS_NUM_SEGMENTS, SOFTPLUS_LUT_SIZE>(SOFTPLUS_LUT, x);
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void softplus_init() {
    sfpu_reciprocal_init();
}

}  // namespace ckernel::sfpu
