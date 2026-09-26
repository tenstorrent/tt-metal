// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

// Shared S60 runtime for the structural family
//
//     y = copysign(min(P(abs(x)), clamp_max), x).
//
// The generated Config owns the polynomial degree, every fitted coefficient,
// the clamp, and the complete typed class contract.  This runtime owns only
// the family algorithm.  It intentionally has no operation-name dispatch.

namespace ckernel::sfpu::ttpoly {

constexpr float signed_abs_fp32_from_bits(std::uint32_t bits) { return __builtin_bit_cast(float, bits); }

template <typename Config>
constexpr bool signed_abs_supported_contract() {
    return Config::kExhaustivePopulation == 65536u && Config::kDegree >= 2u && Config::kDegree <= 8u &&
           Config::kInputDaz && Config::kOutputFtz && Config::kRawClassOutput[0] == 3u &&  // +0 -> +0
           Config::kRawClassOutput[1] == 3u &&                                             // -0 -> +0
           Config::kRawClassOutput[2] == 3u &&                                             // +subnormal -> +0
           Config::kRawClassOutput[3] == 3u &&                                             // -subnormal -> +0
           Config::kRawClassOutput[4] == 5u &&                                             // finite -> evaluated
           Config::kRawClassOutput[5] == 5u &&                                             // +Inf -> finite saturation
           Config::kRawClassOutput[6] == 5u &&                                             // -Inf -> finite saturation
           Config::kRawClassOutput[7] == 5u &&       // +NaN -> decoded finite saturation
           Config::kRawClassOutput[8] == 5u &&       // -NaN -> decoded finite saturation
           Config::kRawToEffectiveClass[0] == 3u &&  // +0 -> +0
           Config::kRawToEffectiveClass[1] == 3u &&  // -0 -> +0
           Config::kRawToEffectiveClass[2] == 3u &&  // +subnormal -> +0
           Config::kRawToEffectiveClass[3] == 3u &&  // -subnormal -> +0
           Config::kRawToEffectiveClass[4] == 5u &&  // finite remains finite
           Config::kRawToEffectiveClass[5] == 1u &&  // +Inf remains +Inf
           Config::kRawToEffectiveClass[6] == 2u &&  // -Inf remains -Inf
           Config::kRawToEffectiveClass[7] == 1u &&  // +NaN decodes as +Inf
           Config::kRawToEffectiveClass[8] == 2u;    // -NaN decodes as -Inf
}

template <typename Config, std::uint32_t Index>
sfpi_inline sfpi::vFloat signed_abs_coefficient() {
    static_assert(Index <= Config::kDegree);
    return signed_abs_fp32_from_bits(Config::kCoefficientBits[Index]);
}

template <typename Config, std::uint32_t Index>
sfpi_inline sfpi::vFloat signed_abs_horner(sfpi::vFloat coordinate) {
    if constexpr (Index == Config::kDegree) {
        return signed_abs_coefficient<Config, Index>();
    } else {
        sfpi::vFloat accumulator = signed_abs_horner<Config, Index + 1u>(coordinate);
        sfpi::vFloat addend = signed_abs_coefficient<Config, Index>();
        // Preserve the existing polynomial evaluator's single-rounding MAD.
        return __builtin_rvtt_sfpmad(accumulator.get(), coordinate.get(), addend.get(), SFPMAD_MOD1_OFFSET_NONE);
    }
}

template <typename Config>
sfpi_inline sfpi::vFloat signed_abs_evaluate(sfpi::vFloat x) {
    static_assert(signed_abs_supported_contract<Config>());
    sfpi::vInt input_exponent = sfpi::exexp(x, sfpi::ExponentMode::Biased);
    sfpi::vFloat magnitude = signed_abs_horner<Config, 0u>(sfpi::abs(x));
    sfpi::vFloat clamp_max = signed_abs_fp32_from_bits(Config::kClampMaxBits);
    // Preserve the existing SFPSWAP clamp, including overflow ordering.
    auto clamped = sfpi::min_max(magnitude, clamp_max);
    magnitude = clamped.first;
    clamp_max = clamped.second;
    sfpi::vFloat result = sfpi::copysgn(magnitude, x);
    // Decoded BF16 ingress retains nonfinite sign on both targets. Saturate
    // explicitly: Horner overflow and unordered min cannot own this terminal.
    v_if(input_exponent == 255) {
        sfpi::vFloat terminal = signed_abs_fp32_from_bits(Config::kClampMaxBits);
        result = sfpi::copysgn(terminal, x);
    }
    v_endif;
    // TTNN input DAZ maps both zero signs and every BF16 subnormal to +0.
    v_if(input_exponent == 0) { result = 0.0f; }
    v_endif;
    return sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
}

template <typename Config, int Iterations = 8>
inline void calculate_signed_abs() {
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = signed_abs_evaluate<Config>(x);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu::ttpoly

#define TT_POLY_LLK_DECODED_BF16_SATURATING_SIGNED_ABS_V2 1
