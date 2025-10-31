// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void relu_min(uint uint_threshold) {
    sfpi::vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        v_if(a < threshold) { a = threshold; }
        v_endif;
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void relu_max(uint uint_threshold) {
    sfpi::vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        v_if(a > threshold) { a = threshold; }
        v_endif;
        v_if(a < 0.0f) { a = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel
