// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

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
inline void calculate_softplus_body(vFloat beta, vFloat beta_reciprocal, vFloat threshold) {
    vFloat x = beta * dst_reg[0];
    v_if(x < threshold) { dst_reg[0] = beta_reciprocal * softplus(x); }
    v_endif;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    vFloat beta = Converter::as_float(param0);
    vFloat beta_reciprocal = Converter::as_float(param1);
    vFloat threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE>(beta, beta_reciprocal, threshold);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void softplus_init() {}

}  // namespace sfpu
}  // namespace ckernel
