// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// digamma — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// psi(x) on the sweep stimulus domain [0.1, 50] by the standard recurrence +
// asymptotic-series composition (Abramowitz & Stegun 6.3.5 / 6.3.18):
//   psi(x) = psi(x + n) - sum_{k=0..n-1} 1/(x + k)   (shift until x + n >= 6)
//   psi(y) ~= ln(y) - 1/(2y) - 1/(12 y^2) + 1/(120 y^4) - 1/(252 y^6)
// Six predicated shift steps take every lane from >= 0.1 to >= 6, where the
// truncated series error is < 1e-6.  Reciprocal and ln are stated as typed
// helpers below.  Derivation + tolerance validation:
// laneS2-evidence-20260819/fit_s2.py — max abs error 6.9e-5 against
// torch.digamma over the domain (suite gate atol/rtol 0.05).  Golden:
// torch.digamma (golden_generators._digamma), Float32 corr contract.
#include <cstdint>

namespace ckernel::sfpu
{

// 1/x for positive normal x: magic-constant seed (the classic division-free
// Newton reciprocal seed, 0x7EF311C3 — see e.g. Blinn, "Floating-point
// tricks", IEEE CG&A 1997) refined by three Newton steps y <- y*(2 - x*y);
// relative error ~1 ulp fp32 after refinement.  Header of record for the
// helper (erfinv.h reuses it and fresh_ln_positive via this include).
sfpi_inline sfpi::vFloat fresh_recip_positive(const sfpi::vFloat x)
{
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x7EF311C3) - sfpi::as<sfpi::vInt>(x));
    y              = y * (2.0f - x * y);
    y              = y * (2.0f - x * y);
    y              = y * (2.0f - x * y);
    return y;
}

// ln(x) for positive normal x: ln(x) = e*ln2 + P(m) with m the mantissa in
// [1, 2) and P a degree-4 least-squares fit on Chebyshev nodes (derivation +
// validation: laneS2-evidence-20260819/fit_s2.py — max abs error 6.9e-5).
sfpi_inline sfpi::vFloat fresh_ln_positive(const sfpi::vFloat x)
{
    constexpr float L4  = -5.545931309e-02f;
    constexpr float L3  = 4.405027330e-01f;
    constexpr float L2  = -1.455194831e+00f;
    constexpr float L1  = 2.806980610e+00f;
    constexpr float L0  = -1.736759782e+00f;
    constexpr float LN2 = 0.6931471805599453f;

    const sfpi::vFloat m = sfpi::setexp(x, 127); // mantissa into [1, 2)
    sfpi::vFloat p       = L4;
    p                    = p * m + L3;
    p                    = p * m + L2;
    p                    = p * m + L1;
    p                    = p * m + L0;

    const sfpi::vFloat e = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(sfpi::exexp(x)), sfpi::RoundMode::Nearest);
    return e * LN2 + p;
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_digamma_fresh_cpp()
{
    constexpr float A1 = -1.0f / 12.0f;
    constexpr float A2 = 1.0f / 120.0f;
    constexpr float A3 = -1.0f / 252.0f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat x   = sfpi::dst_reg[0];
        sfpi::vFloat acc = 0.0f;
        // Recurrence: fold 1/x and shift until every lane reaches x >= 6
        // (0.1 + 6 = 6.1; the stimulus domain floor is 0.1).
        for (int step = 0; step < 6; ++step)
        {
            v_if (x < 6.0f)
            {
                acc = acc + fresh_recip_positive(x);
                x   = x + 1.0f;
            }
            v_endif;
        }
        const sfpi::vFloat r  = fresh_recip_positive(x);
        const sfpi::vFloat r2 = r * r;
        sfpi::vFloat tail     = A3;
        tail                  = tail * r2 + A2;
        tail                  = tail * r2 + A1;
        tail                  = tail * r2;
        sfpi::dst_reg[0]      = fresh_ln_positive(x) - 0.5f * r + tail - acc;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
