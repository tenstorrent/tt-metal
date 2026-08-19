// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the unarypower op (storm contract:
// fresh_cpp/README.md).  Migrated verbatim from fresh_cpp_operations.h
// (Lane BR batch 2); byte-stable algorithm, only the file moved.

#include <cstdint>
#include <limits>

namespace ckernel::sfpu
{

// Unary power, bf16 exp_21f contract (production: metal
// _sfpu_unary_power_21f_ with 1/ln2, the -127 clamp, and NaN parked in
// vConstFloatPrgm0/1/2, the addexp(z,23) strength reduction, and integer
// mantissa-offset spellings of the exp_21f quadratic).  Same two-step
// algorithm — log2(base) by the rminimax cubic, then 2^z by the exp_21f
// exponent/mantissa recombination already established in
// calculate_exp_fresh_cpp — with the production's special-value contract
// (0^negative = NaN, negative base: sign by integer-power parity, NaN for
// non-integer powers) and the bf16 RNE store.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_unary_power_fresh_cpp(const std::uint32_t exponent)
{
    // rminimax cubic over [1, 2) for ln(m) (production constants).
    constexpr float P3      = 0x2.44734p-4f;
    constexpr float P2      = -0xd.e712ap-4f;
    constexpr float P1      = 0x2.4f5388p+0f;
    constexpr float P0      = -0x1.952992p+0f;
    constexpr float ONE_LN2 = 1.4426950408889634f;
    // exp_21f fractional refinement (the calculate_exp_fresh_cpp constants).
    constexpr float C0 = 1.0017248f;
    constexpr float C1 = 7.839635491371155e-08f;
    constexpr float C2 = 4.791750143340323e-15f;

    const float pow_scalar = Converter::as_float(exponent);
    const sfpi::vFloat pow = pow_scalar;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat base = sfpi::dst_reg[0];

        // Step 1: log2(|base|) = poly(mantissa)/ln2 + exponent.
        const sfpi::vFloat abs_base    = sfpi::abs(base);
        const sfpi::vFloat m           = sfpi::setexp(abs_base, 127);
        const sfpi::vFloat series      = m * (m * (m * P3 + P2) + P1) + P0;
        const sfpi::vFloat exp_f32     = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(sfpi::exexp(base)), sfpi::RoundMode::Nearest);
        const sfpi::vFloat log2_result = exp_f32 + series * ONE_LN2;

        // Step 2: 2^z by exponent/mantissa recombination (exp_21f).
        sfpi::vFloat zlog2   = pow * log2_result + 127.0f; // biased result exponent
        zlog2                = sfpi::min(zlog2, 255.0f);
        sfpi::vInt zi        = sfpi::shft(sfpi::exman(zlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(zlog2), sfpi::ShiftMode::Logical);
        const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(zi);

        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
        frac              = (C2 * frac + C1) * frac + C0;

        sfpi::vFloat zc = z;
        v_if (zlog2 <= 0.0f)
        {
            zc = 0.0f;
        }
        v_endif;
        sfpi::vFloat y = sfpi::setexp(frac, sfpi::exexp(zc, sfpi::ExponentMode::Biased));

        // Special values (production contract): 0^negative = NaN.
        if (pow_scalar < 0.0f)
        {
            v_if (abs_base == 0.0f)
            {
                y = std::numeric_limits<float>::quiet_NaN();
            }
            v_endif;
        }
        // Negative base: sign from integer-power parity; NaN for non-integer powers.
        v_if (base < 0.0f)
        {
            const auto pow_int             = sfpi::convert<sfpi::vSMag16>(pow, sfpi::RoundMode::Nearest);
            const sfpi::vFloat pow_rounded = sfpi::convert<sfpi::vFloat>(pow_int, sfpi::RoundMode::Nearest);
            y                              = sfpi::setsgn2(y, pow_int);
            v_if (pow_rounded != pow)
            {
                y = std::numeric_limits<float>::quiet_NaN();
            }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
