// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_eqz() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;
        v_if(v == 0) { val = 1; }
        v_endif;

        dst_reg[0] = val;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_ne_int32(int scalar) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;
        v_if(v != scalar) { val = 1; }
        v_endif;
        v_if(v == -2) { val = 7; }
        v_endif;
        // v_if(v < -5) { val = 8; }
        // v_endif;
        dst_reg[0] = val;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
