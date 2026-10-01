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

template <typename Config, std::uint32_t Index, bool Negated = false>
sfpi_inline void signed_abs_horner_pair(sfpi::vFloat& r0, sfpi::vFloat& r1, sfpi::vFloat a0, sfpi::vFloat a1) {
    // Each coefficient is loaded once for two rows. Interleaving independent
    // MADs fills the WH dependency slot without changing either row's order.
    sfpi::vFloat coefficient = signed_abs_pair_coefficient<Config, Index, Negated>();
    r0 = __builtin_rvtt_sfpmad(r0.get(), a0.get(), coefficient.get(), SFPMAD_MOD1_OFFSET_NONE);
    r1 = __builtin_rvtt_sfpmad(r1.get(), a1.get(), coefficient.get(), SFPMAD_MOD1_OFFSET_NONE);
    if constexpr (Index > 0u) {
        signed_abs_horner_pair<Config, Index - 1u, Negated>(r0, r1, a0, a1);
    }
}

// The NaN form returns +NaN for a NaN of either sign and the saturating form's
// result for every other input.
template <typename Config>
inline void init_signed_abs_nan() {
    // Horner runs on -P. BH SFPMAD returns +NaN for a NaN operand, which survives
    // the max that bounds -P below; the min that bounds P would drop it.
    sfpi::vConstFloatPrgm0 = signed_abs_negated(Config::kCoefficientBits[Config::kDegree]);
    sfpi::vConstFloatPrgm1 = signed_abs_negated(Config::kCoefficientBits[Config::kDegree - 1u]);
    sfpi::vConstFloatPrgm2 = signed_abs_negated(Config::kCoefficientBits[Config::kDegree - 2u]);
}

template <typename Config>
sfpi_inline sfpi::vFloat signed_abs_nan_finish(sfpi::vFloat x, sfpi::vFloat negated) {
    sfpi::vFloat bound = -signed_abs_fp32_from_bits(Config::kClampMaxBits);
    sfpi::vFloat bounded = sfpi::min_max(bound, negated).second;
    // bounded <= -0 except for the NaN, so the AND keeps x's sign or clears it.
    sfpi::vFloat sign = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(x) & sfpi::as<sfpi::vInt>(bounded));
    return sfpi::convert<sfpi::vFloat16b>(sfpi::copysgn(bounded, sign), sfpi::RoundMode::Nearest);
}

template <typename Config, int Iterations = 8>
inline void calculate_signed_abs_nan() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
#pragma GCC unroll 4
    for (int d = 0; d < Iterations / 2; ++d) {
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::vFloat r0 = sfpi::vConstFloatPrgm0;
        sfpi::vFloat r1 = sfpi::vConstFloatPrgm0;
        signed_abs_horner_pair<Config, Config::kDegree - 1u, true>(r0, r1, sfpi::abs(x0), sfpi::abs(x1));
        r0 = signed_abs_nan_finish<Config>(x0, r0);
        r1 = signed_abs_nan_finish<Config>(x1, r1);
        sfpi::dst_reg[0] = r0;
        sfpi::dst_reg[1] = r1;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu::bf16
