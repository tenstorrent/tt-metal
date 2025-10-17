// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline vFloat softplus(vFloat x) {
    vFloat result = x;

    v_if(x < -4.0f) {
        // For very negative values, softplus(x) ≈ exp(x)
        result = _sfpu_exp_21f_(x);
    }
    v_elseif(x < 0.0f) {
        // Coefficients for [-4, 0]
        result =
            ((((((-6.11343628e-05 * x - 9.83003622e-04) * x - 4.84124664e-03) * x + 4.19676832e-03) * x +
               1.30285097e-01) *
                  x +
              5.01969907e-01) *
                 x +
             6.93148958e-01);
    }

    v_elseif(x < 4.0f) {
        // Coefficients for [0, 4] - 3rd degree polynomial (Remez-style Minimax)
        result = (((-1.2162164682e-02 * x + 1.3015606879e-01) * x + 5.0493554482e-01) * x + 6.9174850982e-01);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_softplus_body(uint param0, uint param1, uint param2) {
    // x = beta * input
    vFloat x = Converter::as_float(param0) * dst_reg[0];
    // If beta * input < threshold: output = (1/beta) * softplus(beta * input)
    v_if(x < Converter::as_float(param2)) { dst_reg[0] = Converter::as_float(param1) * softplus(x); }
    v_endif;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE>(param0, param1, param2);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void softplus_init() {}

}  // namespace sfpu
}  // namespace ckernel
