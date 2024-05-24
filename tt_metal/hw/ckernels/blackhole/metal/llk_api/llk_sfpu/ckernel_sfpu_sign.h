// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sign() {
    // All params are in FP16 format
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        vFloat result = vConst1;
        v_if(v < 0.0f) { result = vConstNeg1; }
        v_elseif(v > 0.0f) { result = vConst1; }
        v_else { result = vConst0; }
        v_endif;

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
