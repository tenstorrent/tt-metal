// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu::bf16 {

constexpr float exp2_reciprocal_float(std::uint32_t bits) { return __builtin_bit_cast(float, bits); }

sfpi_inline sfpi::vFloat exp2_reciprocal_mad(sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), SFPMAD_MOD1_OFFSET_NONE);
}

// Same power-of-two coefficient scaling as the existing fused Horner.
template <typename Config>
constexpr float exp2_reciprocal_c2 = exp2_reciprocal_float(Config::kCoefficientBits[2]) * 0x1p-46f;
template <typename Config>
constexpr float exp2_reciprocal_c1 = exp2_reciprocal_float(Config::kCoefficientBits[1]) * 0x1p-23f;

template <typename Config>
inline void init_exp2_reciprocal() {
    // The target header owns ARECIP/Newton on BH and quadratic-seed/Newton
    // on WH. Its program registers must survive until the numerical call.
    ckernel::sfpu::sfpu_reciprocal_init<false>();
    // BH's reciprocal reads only Prgm0, which leaves Prgm1/Prgm2 to the
    // leading Horner coefficients.
    sfpi::vConstFloatPrgm1 = exp2_reciprocal_c2<Config>;
    sfpi::vConstFloatPrgm2 = exp2_reciprocal_c1<Config>;
}

template <typename Config>
sfpi_inline sfpi::vFloat exp2_reciprocal_evaluate(
    sfpi::vFloat x, sfpi::vFloat multiplier, sfpi::vFloat c2, sfpi::vFloat c1, sfpi::vFloat c0) {
    // BH uses the upstream negated-product form; its compiler folds -x into
    // SFPMUL modifier 1. The final raw-class contract admits each target explicitly.
    sfpi::vFloat transformed;
    transformed = (-x) * multiplier;
    transformed = transformed + float(Config::kBias);
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
    sfpi::vFloat polynomial = exp2_reciprocal_mad(c2, fraction, c1);
    polynomial = exp2_reciprocal_mad(polynomial, fraction, c0);
    sfpi::vFloat exponential = sfpi::setexp(polynomial, biased_exponent);
    sfpi::vFloat result = ckernel::sfpu::sfpu_reciprocal_iter<1>(1.0f + exponential);

    // The clamp owns the decoded zero/infinity terminals. NaN results also
    // depend on the target multiply primitive above, so the final raw contract
    // must be qualified on the complete physical BF16 population.
    sfpi::vFloat output = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    if constexpr (Config::kNanTerminal) {
        // A NaN of either sign is stored as +Inf, its BF16 class.
        v_if(sfpi::as<sfpi::vInt>(sfpi::setsgn(x, 0)) > sfpi::vInt(0x7f800000)) {
            output = std::numeric_limits<float>::infinity();
        }
        v_endif;
    }
    return output;
}

template <typename Config, int Iterations = 8>
inline void calculate_exp2_reciprocal() {
    // Loop-invariant operands stay in registers across the tile instead of
    // being reloaded for every row. WH's reciprocal owns all three program
    // registers, so its two Horner coefficients take general registers too.
    const sfpi::vFloat multiplier = exp2_reciprocal_float(Config::kMultiplierBits & 0x7fffffffu);
    const sfpi::vFloat c2 = sfpi::vConstFloatPrgm1;
    const sfpi::vFloat c1 = sfpi::vConstFloatPrgm2;
    const sfpi::vFloat c0 = exp2_reciprocal_float(Config::kCoefficientBits[0]);
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = exp2_reciprocal_evaluate<Config>(x, multiplier, c2, c1, c0);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu::bf16
