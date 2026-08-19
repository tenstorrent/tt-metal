// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// gelu — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Migrated verbatim from ../fresh_cpp_operations.h (Lane BR causal-tier lift);
// self-contained (no shared-helper dependency).
#include <cstdint>

namespace ckernel::sfpu
{

// GELU, bf16 non-approx contract (production: calculate_gelu_piecewise —
// progressive v_and CC-narrowing inside one predicate block).  The same
// four-region piecewise CDF (identical constants, including the 2^-25
// ROUND_TO_GRID staircase snap, which is golden math reproducing torch's
// float32 erfc tail) stated as independent typed regions.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_gelu_fresh_cpp()
{
    constexpr float GELU_SAT         = -5.54259443f;
    constexpr float NEG_HALF_ONE_LN2 = -0.72134752044f; // -0.5 / ln(2)
    constexpr float HC0              = 3.0369991064e-01f;
    constexpr float HC1              = 9.5413386822e-02f;
    constexpr float HC2              = 1.3809983619e-02f;
    constexpr float HC3              = 7.5950479368e-04f;
    constexpr float ROUND_TO_GRID    = 0.375f;
    constexpr float E0               = 1.0017248f;
    constexpr float E1               = 7.839635491371155e-08f;
    constexpr float E2               = 4.791750143340323e-15f;
    constexpr float P0               = 5.000000000e-01f;
    constexpr float P1               = 3.9894227818e-01f;
    constexpr float P3               = -6.6361041488e-02f;
    constexpr float P5               = 9.7720050615e-03f;
    constexpr float P7               = -1.0717806322e-03f;
    constexpr float P9               = 8.1812159812e-05f;
    constexpr float P11              = -3.8082057209e-06f;
    constexpr float P13              = 7.9821413868e-08f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        // x^2 feeds two of the three computed regions; state it once at the
        // top (all lanes) so neither predicate arm recomputes it.
        const sfpi::vFloat x2 = x * x;
        // Identity region (x >= 2.78125) as the all-lane default.
        sfpi::vFloat r = x;
        v_if (x <= GELU_SAT)
        {
            r = 0.0f;
        }
        v_elseif (x < -3.125f)
        {
            // H = exp(-x^2/2) * corr_H(x), snapped to the 2^-25 grid.
            const sfpi::vFloat xlog2   = x2 * NEG_HALF_ONE_LN2 + 127.0f;
            const sfpi::vInt zi        = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
            const sfpi::vFloat z       = sfpi::as<sfpi::vFloat>(zi);
            sfpi::vFloat frac          = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            frac                       = (E2 * frac + E1) * frac + E0;
            const sfpi::vFloat exp_val = sfpi::setexp(frac, sfpi::exexp(z, sfpi::ExponentMode::Biased));

            const sfpi::vFloat H  = exp_val * (((HC3 * x + HC2) * x + HC1) * x + HC0);
            const sfpi::vFloat Hs = (H + ROUND_TO_GRID) - ROUND_TO_GRID;
            r                     = x * Hs;
        }
        v_elseif (x < 2.78125f)
        {
            sfpi::vFloat odd = P13;
            odd                   = odd * x2 + P11;
            odd                   = odd * x2 + P9;
            odd                   = odd * x2 + P7;
            odd                   = odd * x2 + P5;
            odd                   = odd * x2 + P3;
            odd                   = odd * x2 + P1;
            r                     = x * (P0 + x * odd);
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
