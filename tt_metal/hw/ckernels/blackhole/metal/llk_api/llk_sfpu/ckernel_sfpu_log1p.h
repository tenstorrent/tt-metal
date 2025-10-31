// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool FAST_APPROX, int ITERATIONS = 8>
inline void calculate_log1p() {
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        in = in + 1.0f;
        sfpi::vFloat x = sfpi::setexp(in, 127);  // Normalize to [1, 2] range

        // Cheby Approximation using Horner Form Multiplication: 3rd Order
        sfpi::vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;

        sfpi::vInt exp = sfpi::exexp(in);
        v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
        v_endif;

        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
        sfpi::vFloat result = expf * vConstLn2 + series_result;

        v_if(in == 0.0F) { result = -std::numeric_limits<float>::infinity(); }
        v_endif;

        if constexpr (!FAST_APPROX) {
            v_if(in < 0.0F) {
                result = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
            }
            v_endif;
        }

        sfpi::dst_reg[0] = result;
        ++sfpi::dst_reg;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX>
inline void log1p_init() {
    sfpi::vConstFloatPrgm0 = 0.692871f;  // ln2
    sfpi::vConstFloatPrgm1 = 0.1058f;
    sfpi::vConstFloatPrgm2 = -0.7166f;
}

}  // namespace sfpu
}  // namespace ckernel
