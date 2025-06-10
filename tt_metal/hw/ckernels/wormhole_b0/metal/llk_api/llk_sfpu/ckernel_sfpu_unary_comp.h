// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_ne(uint value) {
    // SFPU microcode
    vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v == s) { v = 0.0f; }
        v_else { v = 1.0f; }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_eq(uint value) {
    // SFPU microcode
    vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v == s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_gt(uint value) {
    // SFPU microcode
    vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v > s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_lt(uint value) {
    // SFPU microcode
    vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v < s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
