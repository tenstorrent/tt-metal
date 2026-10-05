// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"

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

namespace ckernel::sfpu::bf16 {

constexpr float signed_abs_fp32_from_bits(std::uint32_t bits) { return __builtin_bit_cast(float, bits); }

// A zero stays +0, which the constant register holds: the BF16 narrowing drops a zero's sign.
constexpr float signed_abs_negated(std::uint32_t bits) {
    return (bits & 0x7fffffffu) == 0u ? 0.0f : -signed_abs_fp32_from_bits(bits);
}

template <typename Config, std::uint32_t Index>
sfpi_inline sfpi::vFloat signed_abs_coefficient() {
    static_assert(Index <= Config::kDegree);
    return signed_abs_fp32_from_bits(Config::kCoefficientBits[Index]);
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

template <typename Config>
inline void init_signed_abs() {
    sfpi::vConstFloatPrgm0 = signed_abs_fp32_from_bits(Config::kCoefficientBits[Config::kDegree]);
    sfpi::vConstFloatPrgm1 = signed_abs_fp32_from_bits(Config::kCoefficientBits[Config::kDegree - 1u]);
    sfpi::vConstFloatPrgm2 = signed_abs_fp32_from_bits(Config::kCoefficientBits[Config::kDegree - 2u]);
}

// SFPLOADIs a coefficient costs at each use: none for +0 and +-1, which constant
// registers hold; one for a value exact in BF16 or in normal FP16; two otherwise.
constexpr std::uint32_t signed_abs_load_slots(std::uint32_t bits) {
    if (bits == 0u || (bits & 0x7fffffffu) == 0x3f800000u) {
        return 0u;
    }
    if ((bits & 0xffffu) == 0u) {
        return 1u;
    }
    std::uint32_t exponent = (bits >> 23) & 0xffu;
    return (bits & 0x1fffu) == 0u && exponent >= 127u - 14u && exponent <= 127u + 15u ? 1u : 2u;
}

// The immediate coefficient (below the three in programmable registers) that the
// pair loop holds in a register for the whole call, or kDegree for none: of those
// that cost the most SFPLOADIs, the one of highest degree, for which sfpi needs no
// register copy before the clamp.
template <typename Config>
constexpr std::uint32_t signed_abs_held_index() {
    std::uint32_t held = Config::kDegree, slots = 0u;
    for (std::uint32_t index = 0u; index + 2u < Config::kDegree; ++index) {
        std::uint32_t cost = signed_abs_load_slots(Config::kCoefficientBits[index]);
        if (cost != 0u && cost >= slots) {
            held = index;
            slots = cost;
        }
    }
    return held;
}

template <typename Config, std::uint32_t Index, bool Negated = false>
sfpi_inline sfpi::vFloat signed_abs_pair_coefficient() {
    if constexpr (Index == Config::kDegree - 1u) {
        return sfpi::vConstFloatPrgm1;
    } else if constexpr (Index == Config::kDegree - 2u) {
        return sfpi::vConstFloatPrgm2;
    } else if constexpr (Negated) {
        return signed_abs_negated(Config::kCoefficientBits[Index]);
    } else {
        return signed_abs_coefficient<Config, Index>();
    }
}

template <typename Config, std::uint32_t Index, bool Negated, bool Hold>
sfpi_inline void signed_abs_horner_pair(
    sfpi::vFloat& r0, sfpi::vFloat& r1, sfpi::vFloat a0, sfpi::vFloat a1, sfpi::vFloat held) {
    // Each coefficient is loaded once for two rows. Interleaving independent
    // MADs fills the WH dependency slot without changing either row's order.
    sfpi::vFloat coefficient;
    if constexpr (Hold && Index == signed_abs_held_index<Config>()) {
        coefficient = held;
    } else {
        coefficient = signed_abs_pair_coefficient<Config, Index, Negated>();
    }
    r0 = __builtin_rvtt_sfpmad(r0.get(), a0.get(), coefficient.get(), SFPMAD_MOD1_OFFSET_NONE);
    r1 = __builtin_rvtt_sfpmad(r1.get(), a1.get(), coefficient.get(), SFPMAD_MOD1_OFFSET_NONE);
    if constexpr (Index > 0u) {
        signed_abs_horner_pair<Config, Index - 1u, Negated, Hold>(r0, r1, a0, a1, held);
    }
}

template <typename Config, std::uint32_t Index, bool Negated = false>
sfpi_inline void signed_abs_horner_pair(sfpi::vFloat& r0, sfpi::vFloat& r1, sfpi::vFloat a0, sfpi::vFloat a1) {
    signed_abs_horner_pair<Config, Index, Negated, false>(r0, r1, a0, a1, sfpi::vFloat(0.0f));
}

// Whether the pair loop holds a coefficient. Only the finish with no
// exceptional or terminal code has a free register for it while the data are live.
template <typename Config, bool SymmetricTerminal>
constexpr bool signed_abs_holds_coefficient() {
    return signed_abs_held_index<Config>() < Config::kDegree && Config::kIntrinsicExceptional &&
           !signed_abs_explicit_terminal<Config, SymmetricTerminal>();
}

template <typename Config, bool SymmetricTerminal, int Iterations>
inline void calculate_signed_abs_pairs() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    constexpr bool hold = signed_abs_holds_coefficient<Config, SymmetricTerminal>();
    // Loaded once here instead of at every row pair.
    sfpi::vFloat held = 0.0f;
    if constexpr (hold) {
        held = signed_abs_fp32_from_bits(Config::kCoefficientBits[signed_abs_held_index<Config>()]);
    }
#pragma GCC unroll 4
    for (int d = 0; d < Iterations / 2; ++d) {
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::vFloat r0 = sfpi::vConstFloatPrgm0;
        sfpi::vFloat r1 = sfpi::vConstFloatPrgm0;
        if constexpr (hold) {
            signed_abs_horner_pair<Config, Config::kDegree - 1u, false, true>(
                r0, r1, sfpi::abs(x0), sfpi::abs(x1), held);
        } else {
            signed_abs_horner_pair<Config, Config::kDegree - 1u>(r0, r1, sfpi::abs(x0), sfpi::abs(x1));
        }
        r0 = signed_abs_finish<Config, SymmetricTerminal>(x0, r0);
        r1 = signed_abs_finish<Config, SymmetricTerminal>(x1, r1);
        sfpi::dst_reg[0] = r0;
        sfpi::dst_reg[1] = r1;
        sfpi::dst_reg += 2;
    }
}

template <typename Config, int Iterations = 8>
inline void calculate_signed_abs_terminal() {
    static_assert(Config::kTerminalBoundBits > 0u && Config::kTerminalBoundBits < 0x7f800000u);
    calculate_signed_abs_pairs<Config, true, Iterations>();
}

}  // namespace ckernel::sfpu::bf16
