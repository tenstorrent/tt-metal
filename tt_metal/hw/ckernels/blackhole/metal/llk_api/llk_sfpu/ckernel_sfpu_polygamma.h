// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_recip.h"

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
    int n = int(Converter::as_float(n_packed));
    float scale = Converter::as_float(scale_packed);

    // Precompute Bernoulli-related coefficients for asymptotic tail
    // B_2 = 1/6 → coeff = (n+1)/12  (from B_2/(2!) * s*(s-1)... but simplified)
    // B_4 = -1/30 → coeff = -(n+1)(n+2)(n+3)/720
    // B_6 = 1/42  → coeff = (n+1)(n+2)(n+3)(n+4)(n+5)/30240
    float n1 = n + 1;
    float n2 = n + 2;
    float n3 = n + 3;
    float n4 = n + 4;
    float n5 = n + 5;
    float nf = n;
    float inv_nf = 1.0f / nf;
    float c_b2 = n1 / 12.0f;                           // B_2 term coefficient
    float c_b4 = -(n1 * n2 * n3) / 720.0f;             // B_4 term coefficient
    float c_b6 = (n1 * n2 * n3 * n4 * n5) / 30240.0f;  // B_6 term coefficient

    constexpr auto RECIP = APPROXIMATION_MODE ? 0 : is_fp32_dest_acc_en ? 2 : 1;

    auto power = [] __attribute__((always_inline)) (sfpi::vFloat x, int pwr, sfpi::vFloat val = 1.0f) {
        for (;;) {
            if (pwr & 1) {
                val *= x;
            }
            pwr >>= 1;
            if (!pwr) {
                break;
            }
            x *= x;
        }
        return val;
    };

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat sum = 0.0f;

        // Part 1: Exact summation of first NUM_TERMS terms
        // Σ_{k=0}^{NUM_TERMS-1} 1/(x+k)^(n+1)
        for (int k = 0; k < NUM_TERMS; k++) {
            sfpi::vFloat xi = x + float(k);

            // Compute reciprocal first, then raise to power (avoids overflow of large intermediates)
            sfpi::vFloat inv_xi = sfpu_reciprocal_iter<RECIP>(xi);
            sfpi::vFloat inv_power = power(inv_xi, n, inv_xi);

            sum += inv_power;
        }

        // Part 2: Euler-Maclaurin asymptotic tail correction
        // For the remaining sum Σ_{k=NUM_TERMS}^{∞} 1/(x+k)^(n+1)
        // at z = x + NUM_TERMS:
        sfpi::vFloat z = x + float(NUM_TERMS);
        sfpi::vFloat inv_z = sfpu_reciprocal_iter<RECIP>(z);
        sfpi::vFloat inv_z2 = inv_z * inv_z;

        // Use PolynomialEvaluator for the Bernoulli polynomial in the tail:
        // E = inv_nf + c_b2*inv_z2 + c_b4*inv_z2^2 + c_b6*inv_z2^3
        sfpi::vFloat E = PolynomialEvaluator::eval(inv_z2, inv_nf, c_b2, c_b4, c_b6);
        sfpi::vFloat tail = E + 0.5f * inv_z;

        // Scale by inv_z^n, taking advantage of inv_z^2's
        // computation above
        int pwr = n;
        if (pwr & 1) {
            tail *= inv_z;
        }
        pwr >>= 1;
        if (pwr) {
            // x^2n == (x^2)^n
            tail = power(inv_z2, pwr, tail);
        }

        sum += tail;

        // Apply scale: (-1)^(n+1) * n!
        sfpi::vFloat result = sum * scale;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void polygamma_init() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
