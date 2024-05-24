// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sigmoid.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_silu() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        v_if(val < 0.0f) { val = -val; }
        v_endif;

        vFloat result = sigmoid_piecewise_linear_positive(val);

        val = dst_reg[0];
        v_if(val < 0.0f) { result = 1.0f - result; }
        v_endif;
        result = val * result;
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
