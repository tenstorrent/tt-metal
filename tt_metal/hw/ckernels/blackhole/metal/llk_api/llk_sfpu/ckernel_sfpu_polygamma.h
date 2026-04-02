// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

/**
 * Fused SFPU kernel for polygamma function: ψ^(n)(x)
 *
 * Computes: ψ^(n)(x) = (-1)^(n+1) * n! * Σ_{k=0}^{∞} 1/(x+k)^(n+1)
 *
 * Uses exact summation for the first NUM_TERMS terms, then adds an
 * Euler-Maclaurin asymptotic tail correction for the remaining infinite sum.
 * This dramatically improves accuracy vs plain truncation (e.g. trigamma
 * max ULP drops from ~108 to ~1).
 *
 * Tail at z = x + NUM_TERMS (Euler-Maclaurin remainder with B₂, B₄, B₆ corrections):
 *   tail = 1/(n·z^n) + 1/(2·z^(n+1)) + B₂·(n+1)/(z^(n+2))
 *          + B₄·(n+1)(n+2)(n+3)/(z^(n+4))
 *          + B₆·(n+1)(n+2)(n+3)(n+4)(n+5)/(z^(n+6))
 * where B₂=1/6, B₄=-1/30, B₆=1/42 are Bernoulli numbers, giving coefficients
 *   (n+1)/12, -(n+1)(n+2)(n+3)/720, (n+1)(n+2)(n+3)(n+4)(n+5)/30240
 *
 * Parameters are passed as bit-cast uint32_t values:
 *   n_packed:     order n (as float bits)
 *   scale_packed: precomputed (-1)^(n+1) * n! (as float bits)
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_polygamma(uint32_t n_packed, uint32_t scale_packed) {
    constexpr int NUM_TERMS = 11;  // Exact terms (k=0..10)

    // Unpack parameters using Converter (union-based type punning supported by SFPU compiler)
    float n_float = Converter::as_float(n_packed);
    int n = static_cast<int>(n_float);
    int n_plus_1 = n + 1;
    float scale = Converter::as_float(scale_packed);

    // Precompute Bernoulli-related coefficients for asymptotic tail
    // B_2 = 1/6 → coeff = (n+1)/12  (from B_2/(2!) * s*(s-1)... but simplified)
    // B_4 = -1/30 → coeff = -(n+1)(n+2)(n+3)/720
    // B_6 = 1/42  → coeff = (n+1)(n+2)(n+3)(n+4)(n+5)/30240
    float n1 = static_cast<float>(n + 1);
    float n2 = static_cast<float>(n + 2);
    float n3 = static_cast<float>(n + 3);
    float n4 = static_cast<float>(n + 4);
    float n5 = static_cast<float>(n + 5);
    float nf = static_cast<float>(n);
    float c_b2 = n1 / 12.0f;                           // B_2 term coefficient
    float c_b4 = -(n1 * n2 * n3) / 720.0f;             // B_4 term coefficient
    float c_b6 = (n1 * n2 * n3 * n4 * n5) / 30240.0f;  // B_6 term coefficient

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat sum = sfpi::vFloat(0.0f);

        // Part 1: Exact summation of first NUM_TERMS terms
        // Σ_{k=0}^{NUM_TERMS-1} 1/(x+k)^(n+1)
        for (int k = 0; k < NUM_TERMS; k++) {
            sfpi::vFloat xi = x + static_cast<float>(k);

            // Compute xi^(n+1) by repeated multiplication
            sfpi::vFloat power = xi;
            for (int j = 1; j < n_plus_1; j++) {
                power = power * xi;
            }

            // 1 / xi^(n+1)
            sum = sum + _sfpu_reciprocal_<2>(power);
        }

        // Part 2: Euler-Maclaurin asymptotic tail correction
        // For the remaining sum Σ_{k=NUM_TERMS}^{∞} 1/(x+k)^(n+1)
        // at z = x + NUM_TERMS:
        sfpi::vFloat z = x + static_cast<float>(NUM_TERMS);
        sfpi::vFloat inv_z = _sfpu_reciprocal_<2>(z);
        sfpi::vFloat inv_z2 = inv_z * inv_z;

        // z^n = z^(n+1) / z, but we compute inv_z^n = inv_z^(n+1) * z
        // Compute inv_z^n and inv_z^(n+1) by repeated multiplication
        sfpi::vFloat inv_z_n = inv_z;  // inv_z^1
        for (int j = 1; j < n; j++) {
            inv_z_n = inv_z_n * inv_z;  // inv_z^n
        }
        sfpi::vFloat inv_z_np1 = inv_z_n * inv_z;  // inv_z^(n+1)

        // Integral term: 1/(n * z^n)
        sfpi::vFloat tail = inv_z_n * (1.0f / nf);

        // Half-endpoint: 1/(2 * z^(n+1))
        tail = tail + inv_z_np1 * 0.5f;

        // B_2 Bernoulli correction: (n+1)/(12 * z^(n+2))
        sfpi::vFloat inv_z_np2 = inv_z_np1 * inv_z;
        tail = tail + inv_z_np2 * c_b2;

        // B_4 Bernoulli correction: -(n+1)(n+2)(n+3)/(720 * z^(n+4))
        sfpi::vFloat inv_z_np4 = inv_z_np2 * inv_z2;
        tail = tail + inv_z_np4 * c_b4;

        // B_6 Bernoulli correction: (n+1)(n+2)(n+3)(n+4)(n+5)/(30240 * z^(n+6))
        sfpi::vFloat inv_z_np6 = inv_z_np4 * inv_z2;
        tail = tail + inv_z_np6 * c_b6;

        sum = sum + tail;

        // Apply scale: (-1)^(n+1) * n!
        sfpi::vFloat result = sum * scale;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void polygamma_init() {
    // On BH, _init_reciprocal_ initializes the SFPLOADMACRO-based fast reciprocal path,
    // but our kernel uses _sfpu_reciprocal_<2>() which is Newton-Raphson-based and requires
    // vConstFloatPrgm0 = 2.0f. Use _init_sfpu_reciprocal_ which sets that correctly.
    _init_sfpu_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
