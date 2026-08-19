// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// elu — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// elu(x) = x for x > 0, alpha*(exp(x) - 1) for x <= 0 with the dispatch
// constant alpha = 1 (PyTorch F.elu reference semantics; golden_generators
// ._elu).  exp on the negative branch by the same exponent/mantissa
// recombination the fresh exp body states (exp_21f quadratic refinement,
// Moroz et al. 2022): x <= 0 keeps the biased exponent below 127 so no
// overflow branch is needed; the underflow mask (x below ~-88, outside the
// [-5, 5] stimulus domain but stated for totality) flushes exp to 0 and the
// result to the exact limit -1.  bf16 corr contract (Float16_b sweep row).
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_elu_fresh_cpp()
{
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    constexpr float C0      = 1.0017248f;
    constexpr float C1      = 7.839635491371155e-08f;
    constexpr float C2      = 4.791750143340323e-15f;
    for (int row = 0; row < ITERATIONS; ++row)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat r       = x;
        v_if (x <= 0.0f)
        {
            const sfpi::vFloat xlog2 = x * ONE_LN2 + 127.0f;

            const sfpi::vInt iexp = sfpi::exexp(xlog2);
            sfpi::vInt zi         = sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne);
            zi                    = sfpi::shft(zi, iexp, sfpi::ShiftMode::Logical);
            const sfpi::vFloat z  = sfpi::as<sfpi::vFloat>(zi);

            sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            frac              = (C2 * frac + C1) * frac + C0;

            sfpi::vFloat zc = z;
            v_if (xlog2 <= 0.0f)
            {
                zc = 0.0f;
            }
            v_endif;

            const sfpi::vFloat e = sfpi::setexp(frac, sfpi::exexp(zc, sfpi::ExponentMode::Biased));
            r                    = e - 1.0f;
        }
        v_endif;
        // bf16 destination: round to nearest-even before the store truncates.
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
