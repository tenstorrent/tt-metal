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

// Reuse the existing softplus implementation
inline vFloat softplus(vFloat x) {
    /*
     This function implements softplus using piecewise polynomial
       approximation. The approximation is done using 4 intervals (each branch
       below) and the coefficients were generated using Remez algorithm:

       > guess = np.polyfit(x, y, degree)
       > result = optimize.least_squares(error, guess, args=(x,))

       The intervals and degrees of freedom for each interval was selected
       based on what gave the best compromise of speed vs. accuracy.
    */
    vFloat result;
    v_if(x < -20.0f) { result = vConst0; }
    v_elseif(x < -5.0f) {
        // Coefficients for [-20, -5]
        result =
            (((((2.01778601e-07 * x + 1.41959790e-05) * x + 3.90682149e-04) * x + 5.25169871e-03) * x +
              3.44602422e-02) *
                 x +
             8.83130932e-02);
    }
    v_elseif(x < 0.0f) {
        // Coefficients for [-5, 0]
        result =
            ((((((-6.11343628e-05 * x - 9.83003622e-04) * x - 4.84124664e-03) * x + 4.19676832e-03) * x +
               1.30285097e-01) *
                  x +
              5.01969907e-01) *
                 x +
             6.93148958e-01);
    }
    v_elseif(x < 5.0f) {
        // Coefficients for [0, 5]
        result =
            ((((((-6.11343628e-05 * x + 9.83003622e-04) * x - 4.84124664e-03) * x - 4.19676832e-03) * x +
               1.30285097e-01) *
                  x +
              4.98030093e-01) *
                 x +
             6.93148958e-01);
    }
    v_else {
        // Coefficients for [5, 20]
        result =
            (((((-2.01778601e-07 * x + 1.41959790e-05) * x - 3.90682149e-04) * x + 5.25169871e-03) * x +
              9.65539758e-01) *
                 x +
             8.83130932e-02);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_logsigmoid_body() {
    // Implementation of logsigmoid(x) = -softplus(-x)
    // This fuses the input negation, softplus call, and output negation

    // logsigmoid(x) = -softplus(-x).

    vFloat neg_x = -dst_reg[0];
    dst_reg[0] = -softplus(dst_reg[0]);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid() {
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_logsigmoid_body<APPROXIMATION_MODE>();
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {}

}  // namespace sfpu
}  // namespace ckernel
