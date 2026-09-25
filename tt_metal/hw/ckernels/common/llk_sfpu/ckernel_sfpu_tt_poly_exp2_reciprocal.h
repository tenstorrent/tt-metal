// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu::ttpoly {

constexpr float exp2_reciprocal_float(std::uint32_t bits) { return __builtin_bit_cast(float, bits); }

sfpi_inline sfpi::vFloat exp2_reciprocal_mad(sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), SFPMAD_MOD1_OFFSET_NONE);
}

template <typename Config>
constexpr bool exp2_reciprocal_contract() {
#if defined(ARCH_BLACKHOLE)
    constexpr bool target_terminal = Config::kPositiveNanZero;
#elif defined(ARCH_WORMHOLE)
    constexpr bool target_terminal = !Config::kPositiveNanZero;
#else
    constexpr bool target_terminal = false;
#endif
    return target_terminal && Config::kDegree == 2u && Config::kBias == 127u && Config::kFinitePopulation == 65280u &&
           Config::kInputDaz && Config::kOutputFtz && Config::kLowerClampBits == 0u &&
           Config::kUpperClampBits == 0x437f0000u && Config::kRawClassOutput[0] == 5u &&
           Config::kRawClassOutput[1] == 5u && Config::kRawClassOutput[2] == 5u && Config::kRawClassOutput[3] == 5u &&
           Config::kRawClassOutput[4] == 5u && Config::kRawClassOutput[5] == 5u && Config::kRawClassOutput[6] == 3u &&
           Config::kRawClassOutput[7] == (Config::kPositiveNanZero ? 3u : 5u) && Config::kRawClassOutput[8] == 3u;
}

template <typename Config>
inline void init_exp2_reciprocal() {
    static_assert(exp2_reciprocal_contract<Config>());
    // The target header owns ARECIP/Newton on BH and quadratic-seed/Newton
    // on WH. Its program registers must survive until the numerical call.
    ckernel::sfpu::sfpu_reciprocal_init<false>();
}

template <typename Config>
sfpi_inline sfpi::vFloat exp2_reciprocal_evaluate(sfpi::vFloat x) {
    static_assert(exp2_reciprocal_contract<Config>());
    // BH uses the upstream negated-product form; its compiler folds -x into
    // SFPMUL modifier 1. WH retains its two-instruction signed-coefficient
    // reduction. The final raw-class contract admits each target explicitly.
    sfpi::vFloat transformed;
#if defined(ARCH_BLACKHOLE)
    sfpi::vFloat magnitude = exp2_reciprocal_float(Config::kMultiplierBits & 0x7fffffffu);
    transformed = (-x) * magnitude;
#else
    sfpi::vFloat multiplier = exp2_reciprocal_float(Config::kMultiplierBits);
    transformed = __builtin_rvtt_sfpmul(x.get(), multiplier.get(), SFPMAD_MOD1_OFFSET_NONE);
#endif
    // A separate immediate add preserves the declared split rounding.
    transformed = transformed + float(Config::kBias);
    // Fill the ADDI dependency slot with the independent clamp constant.
    sfpi::vFloat upper_bound = exp2_reciprocal_float(Config::kUpperClampBits);
    auto lower = sfpi::min_max(sfpi::vFloat(exp2_reciprocal_float(Config::kLowerClampBits)), transformed);
    transformed = lower.second;
    auto upper = sfpi::min_max(transformed, upper_bound);
    transformed = upper.first;

    sfpi::vInt exponent = sfpi::exexp(transformed);
    sfpi::vInt mantissa = sfpi::exman(transformed, sfpi::MantissaMode::ImplicitOne);
    mantissa = sfpi::shft(mantissa, exponent, sfpi::ShiftMode::Logical);
    sfpi::vFloat encoded = sfpi::as<sfpi::vFloat>(mantissa);
    sfpi::vInt biased_exponent = sfpi::exexp(encoded, sfpi::ExponentMode::Biased);
    sfpi::vMag fraction_bits = sfpi::exman(encoded);
    sfpi::vFloat fraction = sfpi::convert<sfpi::vFloat>(fraction_bits, sfpi::RoundMode::Nearest);
    // Same power-of-two coefficient scaling as the existing fused Horner.
    constexpr float scaled_c2 = exp2_reciprocal_float(Config::kCoefficientBits[2]) * 0x1p-46f;
    constexpr float scaled_c1 = exp2_reciprocal_float(Config::kCoefficientBits[1]) * 0x1p-23f;
    sfpi::vFloat c2 = scaled_c2;
    sfpi::vFloat c1 = scaled_c1;
    sfpi::vFloat polynomial = exp2_reciprocal_mad(c2, fraction, c1);
    // Load the final coefficient while the first MAD result is pending.
    sfpi::vFloat c0 = exp2_reciprocal_float(Config::kCoefficientBits[0]);
    polynomial = exp2_reciprocal_mad(polynomial, fraction, c0);
    sfpi::vFloat exponential = sfpi::setexp(polynomial, biased_exponent);
    sfpi::vFloat result = ckernel::sfpu::sfpu_reciprocal_iter<1>(1.0f + exponential);
    result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

    // The clamp owns the decoded zero/infinity terminals. NaN results also
    // depend on the target multiply primitive above, so the final raw contract
    // must be qualified on the complete physical BF16 population.
    return result;
}

template <typename Config, int Iterations = 8>
inline void calculate_exp2_reciprocal() {
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = exp2_reciprocal_evaluate<Config>(x);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_EXP2_RECIPROCAL_BF16_V2 1
