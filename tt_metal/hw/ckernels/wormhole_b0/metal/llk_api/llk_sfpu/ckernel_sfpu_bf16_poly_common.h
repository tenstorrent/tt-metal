// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"

// Shared evaluator helpers for BF16-certified SFPU activation kernels.
// Each activation header supplies only a Config struct of coefficient
// constants plus a thin wrapper; the arithmetic lives here once per route
// family so every sibling kernel is reviewed against the same evaluator.
//
// These paths serve the BF16 destination-register case: they assume fp32
// SFPU arithmetic on values that arrived from a BF16 tensor and round
// results back to the BF16 grid (round-to-nearest-even) before the store.

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------
// log1p on an anchored binade split: ln(1 + v) for v in (-1, 0].
//
// Config must provide the correction coefficients
//     static constexpr float log1p_c0, log1p_c1, log1p_c2;
// for log1p(r) ~= r + r^2 * ((c2*r + c1)*r + c0) on r in [-0.25, 0.5).
// ---------------------------------------------------------------------------
template <typename Config>
sfpi_inline sfpi::vFloat log1p_anchored_bf16(sfpi::vFloat v) {
    // u = 1 + v, used only to pick the binade split k below. The reduced
    // argument itself is rebuilt from v so it carries a single rounding.
    sfpi::vFloat u = v + 1.0f;

    // Choose k such that u * 2^-k lands in [0.75, 1.5): subtracting the fp32
    // bit pattern of 0.75 and flooring to a multiple of 2^23 leaves exactly
    // k << 23 (two's complement handles negative k).
    sfpi::vInt anchor_delta = sfpi::as<sfpi::vInt>(u) - sfpi::vInt(0x3F400000);
    sfpi::vInt k_shl23 = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(anchor_delta), 0));

    // 2^-k and v * 2^-k exactly, via exponent arithmetic.
    sfpi::vFloat pow2_neg_k = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x3F800000) - k_shl23);
    sfpi::vFloat v_scaled = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(v) - k_shl23);

    // r = (1 + v) * 2^-k - 1 in [-0.25, 0.5), with one final rounding.
    sfpi::vFloat r = v_scaled + (pow2_neg_k - 1.0f);

    sfpi::vFloat correction = Config::log1p_c2 * r + Config::log1p_c1;
    correction = correction * r + Config::log1p_c0;
    sfpi::vFloat r_sq = r * r;
    sfpi::vFloat log1p_r = r_sq * correction + r;

    // log1p(v) = k * ln2 + log1p(r). k << 23 converts exactly to k * 2^23;
    // the ln2 constant is pre-scaled by 2^-23.
    constexpr float ln2_pow2_m23 = 0x1.62e430p-24f;
    sfpi::vFloat k_pow23 = sfpi::convert<sfpi::vFloat>(k_shl23, sfpi::RoundMode::Nearest);
    return k_pow23 * ln2_pow2_m23 + log1p_r;
}

// ---------------------------------------------------------------------------
// Exact halving of a doubled core value, reproducing BF16 round-to-nearest-
// even at the subnormal boundary.
// ---------------------------------------------------------------------------
sfpi_inline sfpi::vFloat halve_exact_bf16(sfpi::vFloat doubled) {
    sfpi::vFloat half = doubled * 0.5f;

    // If halving a biased-exponent-1 value flushes to zero, restore the one
    // boundary cell that BF16 RNE rounds up to the minimum normal.
    v_if(sfpi::exexp_nodebias(doubled) == 1) {
        sfpi::vFloat norm_mag = sfpi::setexp(sfpi::abs(doubled), 127);
        v_if(norm_mag >= 1.9921875f) { half = sfpi::setman(doubled, 0); }
        v_endif;
    }
    v_endif;
    return half;
}

// ---------------------------------------------------------------------------
// Route family "log_square_factorized_odd":
//
//     t = ln(1 - x^2);   f(x) ~= x * P(|t|)   on |x| < 1
//
// with the inverse-error-function special-value shape:
//     x = +/-1 -> +/-Inf;
//     |x| > 1, +/-Inf, NaN -> quiet NaN before BF16 conversion;
//     x = +/-0 and BF16 subnormals -> zero.
//
// Config must provide:
//     static constexpr int poly_degree;
//     static constexpr float poly[poly_degree + 1];
// ---------------------------------------------------------------------------
template <typename Config>
sfpi_inline sfpi::vFloat log_square_factorized_odd_body(sfpi::vFloat x) {
    static_assert(Config::poly_degree == 3 || Config::poly_degree == 4);

    sfpi::vFloat x_sq = x * x;
    sfpi::vFloat neg_x_sq = -x_sq;
    sfpi::vFloat t = log1p_anchored_bf16<Config>(neg_x_sq);

    sfpi::vFloat t_abs = sfpi::abs(t);
    sfpi::vFloat q = Config::poly[Config::poly_degree];
#pragma GCC unroll 8
    for (int i = Config::poly_degree - 1; i >= 0; i--) {
        q = q * t_abs + Config::poly[i];
    }

    sfpi::vFloat two_x = x + x;
    sfpi::vFloat core = halve_exact_bf16(two_x * q);

    sfpi::vFloat abs_x = sfpi::abs(x);
    sfpi::vFloat result = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x7FC00000));
    v_if(abs_x < 1.0f) { result = core; }
    v_elseif(abs_x == 1.0f) { result = sfpi::setexp(x, 255); }
    v_endif;
    return result;
}

template <typename Config, int ITERATIONS>
inline void calculate_log_square_factorized_odd_bf16() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = log_square_factorized_odd_body<Config>(in);
        sfpi::vFloat16b rounded = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = rounded;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
