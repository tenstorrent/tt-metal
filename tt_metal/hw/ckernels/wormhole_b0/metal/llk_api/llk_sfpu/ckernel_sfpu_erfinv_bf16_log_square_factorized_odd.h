// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

// y = x * P(|log(1 - x*x)|) for |x| < 1; the including configuration holds the coefficients and terminals.

namespace ckernel::sfpu::bf16::erfinv {

constexpr float fp32_from_bits(std::uint32_t bits) { return __builtin_bit_cast(float, bits); }

// The assertions bind the configuration's class table to the partition below.
template <typename Config, std::uint32_t Index>
sfpi_inline sfpi::vFloat log_coefficient() {
    static_assert(Index < 3u);
    return fp32_from_bits(Config::kLogCoefficientBits[Index]);
}

template <typename Config, std::uint32_t Index>
sfpi_inline sfpi::vFloat outer_coefficient() {
    static_assert(Index <= Config::kOuterDegree);
    return fp32_from_bits(Config::kOuterCoefficientBits[Index]);
}

template <typename Config>
sfpi_inline sfpi::vFloat anchored_log1p(sfpi::vFloat v, sfpi::vFloat ln2_times_2_neg23) {
    // k reduces 1 + v to [0.75, 1.5); the correction is r + r^2 * P2(r), coefficients ascending.
    constexpr std::uint32_t kBits0p75 = 0x3f400000u;
    constexpr std::uint32_t kBits1p0 = 0x3f800000u;
    sfpi::vFloat u = v + 1.0f;
    sfpi::vInt anchor_delta = sfpi::as<sfpi::vInt>(u) - sfpi::vInt(kBits0p75);
    sfpi::vInt k_shifted = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(anchor_delta), 0));
    sfpi::vFloat pow2_neg_k = sfpi::as<sfpi::vFloat>(sfpi::vInt(kBits1p0) - k_shifted);
    sfpi::vFloat v_scaled = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(v) - k_shifted);
    sfpi::vFloat r = v_scaled + (pow2_neg_k - 1.0f);
    sfpi::vFloat correction = log_coefficient<Config, 2>() * r + log_coefficient<Config, 1>();
    correction = correction * r + log_coefficient<Config, 0>();
    sfpi::vFloat log1p_r = (r * r) * correction + r;
    sfpi::vFloat k_times_2_23 = sfpi::convert<sfpi::vFloat>(k_shifted, sfpi::RoundMode::Nearest);
    return k_times_2_23 * ln2_times_2_neg23 + log1p_r;
}

sfpi_inline sfpi::vFloat halve_to_bf16_boundary(sfpi::vFloat doubled) {
    sfpi::vFloat half = doubled * 0.5f;
    v_if(sfpi::exexp(doubled, sfpi::ExponentMode::Biased) == 1) {
        sfpi::vFloat normalized = sfpi::setexp(sfpi::abs(doubled), 127);
        v_if(normalized >= 1.9921875f) { half = sfpi::setman(doubled, 0); }
        v_endif;
    }
    v_endif;
    return half;
}

template <typename Config>
sfpi_inline sfpi::vFloat evaluate_open_unit_core(
    sfpi::vFloat x, sfpi::vFloat outer_top, sfpi::vFloat outer_next, sfpi::vFloat ln2_times_2_neg23) {
    static_assert(Config::kOuterDegree == 3);
    sfpi::vFloat t = anchored_log1p<Config>(-(x * x), ln2_times_2_neg23);
    sfpi::vFloat coordinate = sfpi::abs(t);
    sfpi::vFloat q = outer_top * coordinate + outer_next;
    q = q * coordinate + outer_coefficient<Config, 1>();
    q = q * coordinate + outer_coefficient<Config, 0>();
    return halve_to_bf16_boundary((x + x) * q);
}

template <typename Config>
sfpi_inline sfpi::vFloat install_post_round_decoded_terminals(sfpi::vFloat x, sfpi::vFloat narrowed_core) {
    sfpi::vFloat result;
    // Exponent 255 makes x = +/-1 its own +/-Inf (|x| < 1 is overwritten below); any other x stores
    // the NaN class, +Inf in BF16.
    result = fp32_from_bits(0x7fc00000u);
    v_if(sfpi::abs(x) == 1.0f) { result = sfpi::setexp(x, 255); }
    v_endif;
    v_if(sfpi::exexp(x, sfpi::ExponentMode::Biased) < 127) { result = narrowed_core; }
    v_endif;
    return result;
}

template <typename Config>
sfpi_inline sfpi::vFloat evaluate(
    sfpi::vFloat x, sfpi::vFloat outer_top, sfpi::vFloat outer_next, sfpi::vFloat ln2_times_2_neg23) {
    sfpi::vFloat result = 0.0f;
    v_if(sfpi::exexp(x, sfpi::ExponentMode::Biased) < 127) {
        result = evaluate_open_unit_core<Config>(x, outer_top, outer_next, ln2_times_2_neg23);
    }
    v_endif;
    result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    // Reload the row for its sign once the arithmetic is done, rather than keep x live across it.
    sfpi::vFloat terminal_x = sfpi::dst_reg[0];
    return install_post_round_decoded_terminals<Config>(terminal_x, result);
}

// The two leading outer coefficients and ln(2) * 2^-23 stay in LREGs: the row leaves three free.
template <typename Config, int Iterations = 8>
inline void calculate() {
    const sfpi::vFloat outer_top = outer_coefficient<Config, Config::kOuterDegree>();
    const sfpi::vFloat outer_next = outer_coefficient<Config, Config::kOuterDegree - 1>();
    const sfpi::vFloat ln2_times_2_neg23 = 0x1.62e430p-24f;
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = evaluate<Config>(x, outer_top, outer_next, ln2_times_2_neg23);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu::bf16::erfinv
