// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"

namespace ckernel {
namespace sfpu {

// Fast reciprocal sqrt (rsqrt) using fast inverse square root algorithm
// rsqrt(x) = 1 / sqrt(x)
sfpi_inline sfpi::vFloat gelu_rsqrt(sfpi::vFloat val) {
    // Fast inverse square root using magic number approximation
    // See: https://en.wikipedia.org/wiki/Fast_inverse_square_root
    sfpi::vUInt magic = sfpi::reinterpret<sfpi::vUInt>(sfpi::vFloat(sfpi::s2vFloat16b(0x5f37)));
    sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));
    sfpi::vFloat neg_half_val = val * -0.5f;
    // Two Newton-Raphson iterations for accuracy
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    return approx;
}

// GELU approximation using sigmoid: σ(z) ≈ 0.5 + 0.5z / √(1 + z²)
// GELU(x) ≈ x · σ(1.702x)
// This avoids the expensive erf function and has good precision across all inputs
// Note: z / sqrt(1 + z²) = z * rsqrt(1 + z²) to avoid division
inline sfpi::vFloat calculate_gelu_sigmoid_approx(sfpi::vFloat x) {
    constexpr float GELU_SCALE = 1.702f;

    // z = 1.702 * x
    sfpi::vFloat z = x * GELU_SCALE;

    // Compute 1 + z²
    sfpi::vFloat z_sq = z * z;
    sfpi::vFloat one_plus_z_sq = z_sq + 1.0f;

    // Compute rsqrt(1 + z²) = 1 / sqrt(1 + z²)
    sfpi::vFloat rsqrt_val = gelu_rsqrt(one_plus_z_sq);

    // σ(z) = 0.5 + 0.5 * z * rsqrt(1 + z²)
    // Note: z / sqrt(x) = z * rsqrt(x)
    sfpi::vFloat sigma = 0.5f + 0.5f * z * rsqrt_val;

    // GELU(x) = x * σ(z)
    return x * sigma;
}

#define POLYVAL15(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0, x)                         \
    (((((((((((((((c15) * (x) + (c14)) * (x) + (c13)) * (x) + (c12)) * (x) + (c11)) * (x) + (c10)) * (x) + (c9)) * \
                (x) +                                                                                              \
            (c8)) *                                                                                                \
               (x) +                                                                                               \
           (c7)) *                                                                                                 \
              (x) +                                                                                                \
          (c6)) *                                                                                                  \
             (x) +                                                                                                 \
         (c5)) *                                                                                                   \
            (x) +                                                                                                  \
        (c4)) *                                                                                                    \
           (x) +                                                                                                   \
       (c3)) *                                                                                                     \
          (x) +                                                                                                    \
      (c2)) *                                                                                                      \
         (x) +                                                                                                     \
     (c1)) * (x) +                                                                                                 \
        (c0)

inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    v_if(val >= -5.5f) {
        result = POLYVAL15(
            -1.81205228163e-09,
            -4.59055119276e-08,
            -3.74540617693e-07,
            -2.29754133825e-07,
            1.19076782913e-05,
            4.25116466215e-05,
            -0.000138391838381,
            -0.000862052441087,
            0.000768340223025,
            0.0092074331601,
            -0.00208478037614,
            -0.0656369476513,
            0.00244542739174,
            0.398579460781,
            0.499174645395,
            2.98325768482e-05,
            val);

        // Ensure result has the same sign as input using setsgn
        result = setsgn(result, val);
    }
    v_endif;

    return result;
}

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
        sfpi::vFloat result = in;
        v_if(in == 0.0f) { result = 0.0f; }
        v_elseif(sfpi::abs(in) <= 1e-6f) { result = in * 0.5f; }        // Taylor approx for tiny inputs
        v_elseif(in < 3.0f) { result = calculate_gelu_chebyshev(in); }  // Chebyshev for middle range
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
