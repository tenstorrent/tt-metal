// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

/**
 * Largest polygamma order the kernel evaluates. ttnn::polygamma rejects orders outside [1, 11] (11 is
 * reached through polygamma_bw, which evaluates order n + 1 for n <= 10). Orders outside the range are
 * implementation-defined at the compute API and are clamped into it here.
 */
constexpr int POLYGAMMA_MAX_ORDER = 11;

/**
 * Euler-Maclaurin tail coefficients of calculate_polygamma for one order n:
 *   inv_n = 1/n
 *   c_b2  = B₂ · (n+1)                          = (n+1)/12
 *   c_b4  = B₄ · (n+1)(n+2)(n+3) / 4!           = -(n+1)(n+2)(n+3)/720
 *   c_b6  = B₆ · (n+1)(n+2)(n+3)(n+4)(n+5) / 6! = (n+1)(n+2)(n+3)(n+4)(n+5)/30240
 * with the Bernoulli numbers B₂ = 1/6, B₄ = -1/30, B₆ = 1/42.
 */
struct PolygammaTailCoefficients {
    float inv_n;
    float c_b2;
    float c_b4;
    float c_b6;
};

/**
 * Tail coefficients for order n, evaluated at compile time.
 *
 * They used to be computed from the runtime n on the TRISC, which has no FPU: 14 __mulsf3 + 2 __divsf3 +
 * 28 __floatsisf soft-float library calls per kernel invocation (both dest formats). The compiler evaluated
 * the divisions by a constant as multiplications by the rounded reciprocal (-freciprocal-math), and the
 * expressions below are written in exactly that form, so the table is bit-identical to the values the
 * kernel computed before. (Four of the 44 entries differ in the last bit from the correctly rounded
 * quotient; keeping them preserves the kernel's outputs bit for bit.)
 */
constexpr PolygammaTailCoefficients polygamma_tail_coefficients(const int n) {
    const float n1 = static_cast<float>(n + 1);
    const float n2 = static_cast<float>(n + 2);
    const float n3 = static_cast<float>(n + 3);
    const float n4 = static_cast<float>(n + 4);
    const float n5 = static_cast<float>(n + 5);
    return {
        1.0f / static_cast<float>(n),
        n1 * (1.0f / 12.0f),
        -(n1 * n2 * n3) * (1.0f / 720.0f),
        (n1 * n2 * n3 * n4 * n5) * (1.0f / 30240.0f)};
}

inline constexpr PolygammaTailCoefficients POLYGAMMA_TAIL_COEFFICIENTS[POLYGAMMA_MAX_ORDER] = {
    polygamma_tail_coefficients(1),
    polygamma_tail_coefficients(2),
    polygamma_tail_coefficients(3),
    polygamma_tail_coefficients(4),
    polygamma_tail_coefficients(5),
    polygamma_tail_coefficients(6),
    polygamma_tail_coefficients(7),
    polygamma_tail_coefficients(8),
    polygamma_tail_coefficients(9),
    polygamma_tail_coefficients(10),
    polygamma_tail_coefficients(11),
};

/**
 * Decode the order from its float bit pattern without the __fixsfsi soft-float call.
 *
 * n_packed is the bit pattern of a small integer-valued float. For 1.0f <= n < 12.0f the exponent field is
 * 127..130, so shifting the explicit mantissa right by (150 - exponent) is the truncating float->int
 * conversion the kernel used to perform with int(float). Anything below 1.0f (including 0 and every
 * negative pattern, which reads as a huge unsigned value) or at/above 12.0f is clamped, matching the
 * [1, POLYGAMMA_MAX_ORDER] contract of polygamma_tile.
 */
inline int polygamma_decode_order(const std::uint32_t n_packed) {
    constexpr std::uint32_t ONE_BITS = 0x3f800000u;     // 1.0f
    constexpr std::uint32_t TWELVE_BITS = 0x41400000u;  // 12.0f = POLYGAMMA_MAX_ORDER + 1
    if (n_packed < ONE_BITS) {
        return 1;
    }
    if (n_packed >= TWELVE_BITS) {
        return POLYGAMMA_MAX_ORDER;
    }
    const std::uint32_t mantissa = (n_packed & 0x7fffffu) | 0x800000u;
    return static_cast<int>(mantissa >> (150u - (n_packed >> 23)));
}

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
 * The n-dependent coefficients come from POLYGAMMA_TAIL_COEFFICIENTS.
 *
 * Domain: x >= 0.5 (the compute API documents positive real x). Every reciprocal in the kernel is of a
 * positive finite argument there, so the NaN/inf guard of sfpu_reciprocal_iter is not needed; the +inf
 * rows, which the guarded reciprocal used to map to 0, are restored explicitly (see the loop body).
 * Outside the domain the poles x ∈ {0, -1, ..., -6} now produce NaN where the guarded reciprocal
 * produced ±inf; x = -inf produces NaN instead of +0.
 *
 * Parameters are passed as bit-cast uint32_t values:
 *   n_packed:     order n (as float bits), 1 <= n <= POLYGAMMA_MAX_ORDER
 *   scale_packed: precomputed (-1)^(n+1) * n! (as float bits)
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_polygamma(std::uint32_t n_packed, std::uint32_t scale_packed) {
    // Exact terms (k=0..NUM_TERMS-1). The Euler-Maclaurin tail (with B2,B4,B6 corrections)
    // is applied at z = x + NUM_TERMS. For the supported domain (x >= 0.5) this puts
    // z >= 6.5, where the asymptotic remainder is far below bfloat16 precision, so 6 exact
    // terms are sufficient. Reduced from 11 to save ~5 reciprocals (+power chains) per element.
    constexpr int NUM_TERMS = 6;

    const int n = polygamma_decode_order(n_packed);
    const float scale = Converter::as_float(scale_packed);

    // Tail coefficients: four loads from the table instead of soft-float arithmetic on the TRISC.
    const PolygammaTailCoefficients& tail_coefficients = POLYGAMMA_TAIL_COEFFICIENTS[n - 1];
    const float inv_nf = tail_coefficients.inv_n;
    const float c_b2 = tail_coefficients.c_b2;
    const float c_b4 = tail_coefficients.c_b4;
    const float c_b6 = tail_coefficients.c_b6;

    constexpr int RECIP = APPROXIMATION_MODE ? 0 : is_fp32_dest_acc_en ? 2 : 1;

    // Newton refinement of an approx_recip seed for a positive finite argument: the same steps as
    // sfpu_reciprocal_iter<RECIP> (t = x*y - 2, y = y*(-t), one SFPMAD each), without the predicate that
    // skips the step when t is NaN. That predicate costs SFPGT + SFPENCC per reciprocal and only matters
    // for x ∈ {0, inf}. (sfpu_reciprocal_iter spells the step y * -t - 0.0f; the -0.0f addend changes no
    // value and ICEs sfpi 7.83.0's combiner when the statement is not inside a v_if.)
    auto refine_reciprocal = [] __attribute__((always_inline)) (const sfpi::vFloat x, sfpi::vFloat y) {
        if constexpr (RECIP > 0) {
            sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;
            y = y * -t;
            if constexpr (RECIP > 1) {
                t = x * y - sfpi::vConstFloatPrgm0;
                y = y * -t;
            }
        }
        return y;
    };

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
        // Fully unrolled so that k is a constant: x + k is one SFPADDI with an immediate instead of a
        // runtime int->float conversion (__floatsisf) on the TRISC per term.
#pragma GCC unroll 6
        for (int k = 0; k < NUM_TERMS; k++) {
            const sfpi::vFloat xi = (k == 0) ? x : x + float(k);

            // Compute reciprocal first, then raise to power (avoids overflow of large intermediates)
            const sfpi::vFloat inv_xi = refine_reciprocal(xi, sfpi::approx_recip(xi));
            const sfpi::vFloat inv_power = power(inv_xi, n, inv_xi);

            sum += inv_power;
        }

        // Part 2: Euler-Maclaurin asymptotic tail correction
        // For the remaining sum Σ_{k=NUM_TERMS}^{∞} 1/(x+k)^(n+1)
        // at z = x + NUM_TERMS:
        const sfpi::vFloat z = x + float(NUM_TERMS);
        const sfpi::vFloat seed_z = sfpi::approx_recip(z);
        const sfpi::vFloat inv_z = refine_reciprocal(z, seed_z);
        const sfpi::vFloat inv_z2 = inv_z * inv_z;

        // Use PolynomialEvaluator for the Bernoulli polynomial in the tail:
        // E = inv_nf + c_b2*inv_z2 + c_b4*inv_z2^2 + c_b6*inv_z2^3
        const sfpi::vFloat E = PolynomialEvaluator::eval(inv_z2, inv_nf, c_b2, c_b4, c_b6);
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

        // x = +inf: approx_recip seeds every reciprocal with +0 and the guarded reciprocal used to leave it
        // there, giving ψ^(n)(+inf) = 0. The unguarded Newton step turns inf * 0 into NaN instead, so the
        // row is recognised by its tail seed (+0 only for z = +inf or z >= 2^126, where the sum is 0
        // anyway) and the 0 restored. Without a Newton step (APPROXIMATION_MODE) the seed is the result
        // and nothing needs restoring.
        if constexpr (RECIP > 0) {
            v_if(seed_z == 0.0f) { sum = 0.0f; }
            v_endif;
        }

        // Apply scale: (-1)^(n+1) * n!
        sfpi::vFloat result = sum * scale;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void polygamma_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
