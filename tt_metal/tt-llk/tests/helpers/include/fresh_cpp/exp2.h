// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// exp2 — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// 2**x by exponent/mantissa recombination: xlog2 = x + 127 is already the
// BIASED exponent of the result, so no argument scaling is needed at all —
// the fixed-point encoding (mantissa-with-implicit-one << unbiased exponent)
// splits integer and fractional parts and the fractional part is refined by
// the exp_21f quadratic (Moroz et al. 2022, the same refinement the fresh
// exp body states for e**x).  Range handling: overflow saturates the biased
// exponent at 255; underflow (xlog2 <= 0) zeroes the exponent source so the
// recombination flushes to zero.  Golden: torch.exp2
// (golden_generators._exp2), bf16 corr contract (Float16_b sweep row).
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_exp2_fresh_cpp()
{
    constexpr float C0 = 1.0017248f;
    constexpr float C1 = 7.839635491371155e-08f;
    constexpr float C2 = 4.791750143340323e-15f;
    for (int row = 0; row < ITERATIONS; ++row)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat xlog2   = x + 127.0f;
        xlog2                = sfpi::min(xlog2, 255.0f);

        // Fixed-point encoding of xlog2: mantissa (implicit one) shifted
        // left by the unbiased exponent.
        const sfpi::vInt iexp = sfpi::exexp(xlog2);
        sfpi::vInt zi         = sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne);
        zi                    = sfpi::shft(zi, iexp, sfpi::ShiftMode::Logical);
        const sfpi::vFloat z  = sfpi::as<sfpi::vFloat>(zi);

        // Quadratic refinement of 2**xf on [0, 1) from the encoding's
        // mantissa field.
        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
        frac              = (C2 * frac + C1) * frac + C0;

        // Underflow: zero the exponent source where xlog2 is not positive.
        sfpi::vFloat zc = z;
        v_if (xlog2 <= 0.0f)
        {
            zc = 0.0f;
        }
        v_endif;

        sfpi::vFloat y = sfpi::setexp(frac, sfpi::exexp(zc, sfpi::ExponentMode::Biased));

        // bf16 destination: round to nearest-even before the store truncates.
        y                = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
