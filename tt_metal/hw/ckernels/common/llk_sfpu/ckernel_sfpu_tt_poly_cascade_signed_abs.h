// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_tt_poly_horner.h"

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

template <typename Config, std::uint32_t Offset>
struct signed_abs_coefficients {
    constexpr float operator[](std::uint32_t index) const {
        return signed_abs_fp32_from_bits(Config::kCoefficientBits[Offset + index]);
    }
};

template <typename Config, std::uint32_t Index>
sfpi_inline sfpi::vFloat signed_abs_horner(sfpi::vFloat coordinate) {
    static_assert(Index <= Config::kDegree);
    return sfpi::eval_polynomial<Config::kDegree - Index>(signed_abs_coefficients<Config, Index>{}, coordinate);
}

template <typename Config, bool SymmetricTerminal>
constexpr bool signed_abs_explicit_terminal() {
    if constexpr (SymmetricTerminal) {
        return !Config::kIntrinsicTerminal;
    }
    return false;
}

template <typename Config, bool SymmetricTerminal = false>
sfpi_inline sfpi::vFloat signed_abs_finish(sfpi::vFloat x, sfpi::vFloat magnitude) {
    static_assert(signed_abs_supported_contract<Config>());
    sfpi::vInt input_exponent = sfpi::exexp(x, sfpi::ExponentMode::Biased);
    sfpi::vFloat clamp_max = signed_abs_fp32_from_bits(Config::kClampMaxBits);
    // Preserve the existing SFPSWAP clamp, including overflow ordering.
    auto clamped = sfpi::min_max(magnitude, clamp_max);
    magnitude = clamped.first;
    clamp_max = clamped.second;
    sfpi::vFloat result = sfpi::copysgn(magnitude, x);
    if constexpr (signed_abs_explicit_terminal<Config, SymmetricTerminal>()) {
        // The admitted action pair compares the decoded raw coordinate. One
        // predicate owns both finite tails and signed exponent-FF terminals.
        sfpi::vFloat bound = signed_abs_fp32_from_bits(Config::kTerminalBoundBits);
        sfpi::vFloat terminal = signed_abs_fp32_from_bits(Config::kClampMaxBits);
        if constexpr (Config::kTerminalInclusive) {
            v_if(input_exponent == 255 || sfpi::abs(x) >= bound) { result = sfpi::copysgn(terminal, x); }
            v_endif;
        } else {
            v_if(input_exponent == 255 || sfpi::abs(x) > bound) { result = sfpi::copysgn(terminal, x); }
            v_endif;
        }
    } else if constexpr (!Config::kIntrinsicExceptional) {
        // Decoded BF16 ingress retains nonfinite sign on both targets. Saturate
        // explicitly: Horner overflow and unordered min cannot own this terminal.
        v_if(input_exponent == 255) {
            sfpi::vFloat terminal = signed_abs_fp32_from_bits(Config::kClampMaxBits);
            result = sfpi::copysgn(terminal, x);
        }
        v_endif;
    }
    // TTNN input DAZ maps both zero signs and every BF16 subnormal to +0.
    if constexpr (!Config::kIntrinsicExceptional) {
        v_if(input_exponent == 0) { result = 0.0f; }
        v_endif;
    }
    return sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
}

template <typename Config, bool SymmetricTerminal = false>
sfpi_inline sfpi::vFloat signed_abs_evaluate(sfpi::vFloat x) {
    return signed_abs_finish<Config, SymmetricTerminal>(x, signed_abs_horner<Config, 0u>(sfpi::abs(x)));
}

template <typename Config>
inline void init_signed_abs() {
    static_assert(signed_abs_supported_contract<Config>());
    sfpi::vConstFloatPrgm0 = signed_abs_fp32_from_bits(Config::kCoefficientBits[Config::kDegree]);
    sfpi::vConstFloatPrgm1 = signed_abs_fp32_from_bits(Config::kCoefficientBits[Config::kDegree - 1u]);
    sfpi::vConstFloatPrgm2 = signed_abs_fp32_from_bits(Config::kCoefficientBits[Config::kDegree - 2u]);
}

template <typename Config, std::uint32_t Index>
sfpi_inline sfpi::vFloat signed_abs_pair_coefficient() {
    if constexpr (Index == Config::kDegree - 1u) {
        return sfpi::vConstFloatPrgm1;
    } else if constexpr (Index == Config::kDegree - 2u) {
        return sfpi::vConstFloatPrgm2;
    } else {
        return signed_abs_coefficient<Config, Index>();
    }
}

template <typename Config, std::uint32_t Index>
sfpi_inline void signed_abs_horner_pair(sfpi::vFloat& r0, sfpi::vFloat& r1, sfpi::vFloat a0, sfpi::vFloat a1) {
    // Each coefficient is loaded once for two rows. Interleaving independent
    // MADs fills the WH dependency slot without changing either row's order.
    sfpi::vFloat coefficient = signed_abs_pair_coefficient<Config, Index>();
    r0 = __builtin_rvtt_sfpmad(r0.get(), a0.get(), coefficient.get(), SFPMAD_MOD1_OFFSET_NONE);
    r1 = __builtin_rvtt_sfpmad(r1.get(), a1.get(), coefficient.get(), SFPMAD_MOD1_OFFSET_NONE);
    if constexpr (Index > 0u) {
        signed_abs_horner_pair<Config, Index - 1u>(r0, r1, a0, a1);
    }
}

template <typename Config, bool SymmetricTerminal, int Iterations>
inline void calculate_signed_abs_pairs() {
#pragma GCC unroll 4
    for (int d = 0; d < Iterations / 2; ++d) {
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::vFloat r0 = sfpi::vConstFloatPrgm0;
        sfpi::vFloat r1 = sfpi::vConstFloatPrgm0;
        signed_abs_horner_pair<Config, Config::kDegree - 1u>(r0, r1, sfpi::abs(x0), sfpi::abs(x1));
        r0 = signed_abs_finish<Config, SymmetricTerminal>(x0, r0);
        r1 = signed_abs_finish<Config, SymmetricTerminal>(x1, r1);
        sfpi::dst_reg[0] = r0;
        sfpi::dst_reg[1] = r1;
        sfpi::dst_reg += 2;
    }
    if constexpr (Iterations % 2 != 0) {
        sfpi::vFloat result = signed_abs_evaluate<Config, SymmetricTerminal>(sfpi::dst_reg[0]);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <typename Config, int Iterations = 8>
inline void calculate_signed_abs() {
    calculate_signed_abs_pairs<Config, false, Iterations>();
}

template <typename Config, int Iterations = 8>
inline void calculate_signed_abs_terminal() {
    static_assert(Config::kTerminalBoundBits > 0u && Config::kTerminalBoundBits < 0x7f800000u);
#if defined(ARCH_BLACKHOLE)
    static_assert(Config::kIntrinsicTerminal);
#elif defined(ARCH_WORMHOLE)
    static_assert(!Config::kIntrinsicTerminal);
#else
    static_assert(sizeof(Config) == 0, "signed-abs terminal requires BH or WH");
#endif
    calculate_signed_abs_pairs<Config, true, Iterations>();
}

}  // namespace ckernel::sfpu::ttpoly

#define TT_POLY_LLK_DECODED_BF16_PAIRED_SATURATING_SIGNED_ABS_V2 1
#define TT_POLY_LLK_DECODED_BF16_PAIRED_SYMMETRIC_SIGNED_ABS_V2 1
