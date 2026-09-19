// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

// Shared S60 runtime for the structural family
//
//     t = log(1 - x*x); y = x * P(abs(t)), |x| < 1.
//
// The including generated configuration owns every coefficient and terminal
// class.  This file owns only the family algorithm.  It must therefore remain
// free of activation names and coefficient values.

namespace ckernel::sfpu::ttpoly {

constexpr float fp32_from_bits(std::uint32_t bits) { return __builtin_bit_cast(float, bits); }

// This runtime implements the decoded TTNN class partition.  The assertions
// bind that implementation to S60's complete raw-class table; a Torch config
// (which requires a NaN BF16 terminal) cannot compile through this runtime.
template <typename Config>
constexpr bool supported_contract() {
    return Config::kReplayBodySlots == 31u && Config::kExhaustivePopulation == 65536u &&
           (Config::kOuterDegree == 3u || Config::kOuterDegree == 4u) && Config::kInputDaz && Config::kOutputFtz &&
           Config::kRawClassOutput[0] == 3u &&  // +0 -> +0
           Config::kRawClassOutput[1] == 3u &&  // -0 -> +0
           Config::kRawClassOutput[2] == 3u &&  // +subnormal -> +0
           Config::kRawClassOutput[3] == 3u &&  // -subnormal -> +0
           Config::kRawClassOutput[4] == 5u &&  // finite -> evaluated
           Config::kRawClassOutput[5] == 1u &&  // +Inf -> +Inf
           Config::kRawClassOutput[6] == 2u &&  // -Inf -> -Inf
           Config::kRawClassOutput[7] == 1u &&  // +NaN -> +Inf
           Config::kRawClassOutput[8] == 2u &&  // -NaN -> -Inf
           // This family runtime implements exactly the four open/closed
           // boundaries at +/-1 emitted by S60.  Refuse any other terminal
           // program instead of silently hardcoding its behavior.
           (sizeof(Config::kTerminalActions) / sizeof(Config::kTerminalActions[0])) == 4u &&
           Config::kTerminalActions[0].direction == 0u && Config::kTerminalActions[0].bound_bits == 0xbf800000u &&
           Config::kTerminalActions[0].inclusive == 0u && Config::kTerminalActions[0].return_class == 2u &&
           Config::kTerminalActions[1].direction == 1u && Config::kTerminalActions[1].bound_bits == 0x3f800000u &&
           Config::kTerminalActions[1].inclusive == 0u && Config::kTerminalActions[1].return_class == 1u &&
           Config::kTerminalActions[2].direction == 0u && Config::kTerminalActions[2].bound_bits == 0xbf800000u &&
           Config::kTerminalActions[2].inclusive == 1u && Config::kTerminalActions[2].return_class == 2u &&
           Config::kTerminalActions[3].direction == 1u && Config::kTerminalActions[3].bound_bits == 0x3f800000u &&
           Config::kTerminalActions[3].inclusive == 1u && Config::kTerminalActions[3].return_class == 1u;
}

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
sfpi_inline sfpi::vFloat anchored_log1p(sfpi::vFloat v) {
    // k chooses a reduced 1+v in [0.75, 1.5).  S60 supplies the minimax
    // correction in ascending order for r + r^2 * P2(r).
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
    constexpr float kLn2Times2Neg23 = 0x1.62e430p-24f;
    sfpi::vFloat k_times_2_23 = sfpi::convert<sfpi::vFloat>(k_shifted, sfpi::RoundMode::Nearest);
    return k_times_2_23 * kLn2Times2Neg23 + log1p_r;
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
sfpi_inline sfpi::vFloat evaluate_open_unit_core(sfpi::vFloat x) {
    sfpi::vFloat t = anchored_log1p<Config>(-(x * x));
    sfpi::vFloat coordinate = sfpi::abs(t);
    sfpi::vFloat q = outer_coefficient<Config, Config::kOuterDegree>();
    if constexpr (Config::kOuterDegree == 4u) {
        q = q * coordinate + outer_coefficient<Config, 3>();
    }
    q = q * coordinate + outer_coefficient<Config, 2>();
    q = q * coordinate + outer_coefficient<Config, 1>();
    q = q * coordinate + outer_coefficient<Config, 0>();
    return halve_to_bf16_boundary((x + x) * q);
}

sfpi_inline sfpi::vFloat install_post_round_decoded_terminals(sfpi::vFloat x, sfpi::vFloat narrowed_core) {
    sfpi::vFloat positive_infinity = sfpi::setexp(sfpi::vFloat(0.0f), 255);
    sfpi::vFloat result = sfpi::copysgn(positive_infinity, x);
    sfpi::vInt input_exponent = sfpi::exexp(x, sfpi::ExponentMode::Biased);
    // Exact class terminals are installed after the sole BF16 narrowing.
    // Narrowing a synthesized nonfinite value would hand its sign/class back
    // to the target conversion instead of implementing the typed terminal
    // program.  Keep the signed-nonfinite value as the default so an unordered
    // input cannot fall through to an invented finite result.  For the
    // statically asserted +/-1 boundaries, biased exponent < 127 is exactly
    // the open unit interval and does not use unordered float comparison.
    v_if(input_exponent < 127) { result = narrowed_core; }
    v_endif;
    // TTNN's BF16 ingress DAZ profile maps both zero signs and all BF16
    // subnormals to positive zero.  Install it after narrowing as an exact
    // terminal rather than relying on the egress conversion.
    v_if(input_exponent == 0) { result = 0.0f; }
    v_endif;
    return result;
}

template <typename Config>
sfpi_inline sfpi::vFloat evaluate(sfpi::vFloat x) {
    static_assert(supported_contract<Config>());
    sfpi::vFloat result = 0.0f;
    // For the statically asserted +/-1 domain, a biased exponent below the
    // IEEE-754 bias is exactly |x| < 1.  Unlike an SFPU float comparison this
    // partition cannot admit an unordered NaN lane.
    v_if(sfpi::exexp(x, sfpi::ExponentMode::Biased) < 127) { result = evaluate_open_unit_core<Config>(x); }
    v_endif;
    result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    // Reload the same physical row after the arithmetic is dead.  The target
    // load retains the signed-nonfinite carrier needed by copysgn, without
    // keeping the original coordinate live across log reduction and Horner.
    sfpi::vFloat terminal_x = sfpi::dst_reg[0];
    return install_post_round_decoded_terminals(terminal_x, result);
}

template <typename Config, int Iterations = 8>
inline void calculate() {
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = evaluate<Config>(x);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu::ttpoly

#define TT_POLY_LLK_DECODED_BF16_SIGNED_CLASS_V1 1
#define TT_POLY_LLK_POST_ROUND_TERMINAL_STORE_V1 1
#define TT_POLY_LLK_SIGNED_NONFINITE_V1 1
