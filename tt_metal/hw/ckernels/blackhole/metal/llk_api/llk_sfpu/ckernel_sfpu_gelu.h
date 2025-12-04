// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
void gelu_init() {
    _init_gelu_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    _init_gelu_derivative_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat result;

            // - in == 0: exact zero (polynomial gives small non-zero)
            // - in >= 3: identity (GELU(x) ≈ x for large positive)
            // - -5.5 <= in < 3: Chebyshev polynomial approximation
            // - in < -5.5: zero (GELU(x) ≈ 0 for large negative)
            v_if(in == 0.0f) { result = 0.0f; }
            v_elseif(in >= 3.0f) { result = in; }
            v_elseif(in >= -5.5f) {
                result = PolynomialEvaluator::eval(
                    in,
                    2.98325768482e-05f,   // c0
                    0.499174645395f,      // c1
                    0.398579460781f,      // c2
                    0.00244542739174f,    // c3
                    -0.0656369476513f,    // c4
                    -0.00208478037614f,   // c5
                    0.0092074331601f,     // c6
                    0.000768340223025f,   // c7
                    -0.000862052441087f,  // c8
                    -0.000138391838381f,  // c9
                    4.25116466215e-05f,   // c10
                    1.19076782913e-05f,   // c11
                    -2.29754133825e-07f,  // c12
                    -3.74540617693e-07f,  // c13
                    -4.59055119276e-08f,  // c14
                    -1.81205228163e-09f   // c15
                );
                result = sfpi::setsgn(result, in);
            }
            v_else { result = 0.0f; }
            v_endif;

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative() {
    _calculate_gelu_derivative_<APPROXIMATION_MODE, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
