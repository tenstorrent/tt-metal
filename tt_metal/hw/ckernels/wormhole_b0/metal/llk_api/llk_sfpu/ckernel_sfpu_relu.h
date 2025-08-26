// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE>
inline void relu_min(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a < threshold) { a = threshold; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <ApproximationMode APPROX_MODE>
inline void relu_max(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a > threshold) { a = threshold; }
        v_endif;
        v_if(a < 0.0f) { a = 0.0f; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(const uint slope) {
    _calculate_lrelu_<(APPROX_MODE == ApproximationMode::Fast)>(ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel
