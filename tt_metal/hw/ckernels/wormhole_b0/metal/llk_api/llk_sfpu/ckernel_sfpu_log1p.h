// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_log1p() {
    vFloat a = vConstFloatPrgm1;
    vFloat b = vConstFloatPrgm2;
    vFloat vConstLn2 = vConstFloatPrgm0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat in = dst_reg[0];
        in = in + 1.0f;
        vFloat x = setexp(in, 127);  // Normalize to [1, 2] range

        // Cheby Approximation using Horner Form Multiplication: 3rd Order
        vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;

        vInt exp = exexp(in);
        v_if(exp < 0) { exp = setsgn(~exp + 1, 1); }
        v_endif;

        vFloat expf = int32_to_float(exp, 0);
        vFloat result = expf * vConstLn2 + series_result;

        v_if(in == 0.0F) { result = -std::numeric_limits<float>::infinity(); }
        v_endif;

        dst_reg[0] = result;
        ++dst_reg;
    }
}

template <bool APPROXIMATION_MODE>
inline void log1p_init() {
    vConstFloatPrgm0 = 0.692871f;  // ln2
    vConstFloatPrgm1 = 0.1058f;
    vConstFloatPrgm2 = -0.7166f;
}

}  // namespace sfpu
}  // namespace ckernel
