// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline sfpi::vFloat silu_sigmoid_piecewise_linear_positive(sfpi::vFloat val) {
    sfpi::vFloat result = 1.0f;
    v_if(val <= 1.0f) {
        result = 0.2415f * val + 0.5f;  // linear appx as y = 0.2415f + 0.5
    }
    v_elseif(val < 7.7f) {
        result = POLYVAL5<sfpi::vFloat>(
            -3.82558889e-04f, 9.22008486e-03f, -8.34694910e-02f, 3.39967832e-01f, 4.66254244e-01f, val);
    }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_silu() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpi::abs(val);
        result = silu_sigmoid_piecewise_linear_positive(result);
        v_if(val < 0.0f) { result = 1.0f - result; }
        v_endif;
        sfpi::dst_reg[0] = val * result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
