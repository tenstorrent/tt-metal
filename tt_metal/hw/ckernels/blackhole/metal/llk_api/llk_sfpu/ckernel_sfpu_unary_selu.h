// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_selu(uint param0, uint param1) {
    vFloat scale = Converter::as_float(param0);
    vFloat alpha = Converter::as_float(param1);
#pragma GCC unroll 0

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        vFloat result_term2 = (_calculate_exponential_body_<APPROXIMATION_MODE>(v) - 1.0) * alpha;
        v_if(result_term2 > 0.0f) { result_term2 = 0.0; }
        v_endif;
        v_if(v < 0.0f) { v = 0.0; }
        v_endif;
        vFloat sum_max_term2 = v + result_term2;
        dst_reg[0] = sum_max_term2 * scale;
        dst_reg++;
    }
}

// template <bool APPROXIMATION_MODE>
// void selu_init() {}

}  // namespace sfpu
}  // namespace ckernel
