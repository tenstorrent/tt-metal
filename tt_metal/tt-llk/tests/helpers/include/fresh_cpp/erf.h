// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// erf — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// erf is odd and saturates to +/-1, so it is stated as x * P(x^2) with P a
// degree-7 least-squares fit of erf(x)/x on Chebyshev nodes over the sweep
// stimulus domain [0, 3] (fit derivation + tolerance validation:
// laneS2-evidence-20260819/fit_s2.py — max abs error 6.2e-4 against
// torch.erf, two orders under the suite's atol/rtol 0.05 gate), clamped to
// [-1, 1] so the tails stay monotone-saturated.  Golden: torch.erf
// (golden_generators._erf), Float32 corr contract.
#include <cstdint>

namespace ckernel::sfpu
{

// Shared core: erf(x) = clamp(x * P(x^2), -1, 1) on |x| <= 3 (fit domain).
// erfc.h states its complement through this same core.
sfpi_inline sfpi::vFloat fresh_erf_core(const sfpi::vFloat x)
{
    constexpr float E7 = -5.511776635e-07f;
    constexpr float E6 = 2.186222991e-05f;
    constexpr float E5 = -3.751075710e-04f;
    constexpr float E4 = 3.713592421e-03f;
    constexpr float E3 = -2.405889891e-02f;
    constexpr float E2 = 1.101540402e-01f;
    constexpr float E1 = -3.751232028e-01f;
    constexpr float E0 = 1.128316879e+00f;

    const sfpi::vFloat u = x * x;
    sfpi::vFloat p       = E7;
    p                    = p * u + E6;
    p                    = p * u + E5;
    p                    = p * u + E4;
    p                    = p * u + E3;
    p                    = p * u + E2;
    p                    = p * u + E1;
    p                    = p * u + E0;
    sfpi::vFloat r       = x * p;
    r                    = sfpi::min(r, 1.0f);
    r                    = sfpi::max(r, -1.0f);
    return r;
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_erf_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::dst_reg[0] = fresh_erf_core(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
