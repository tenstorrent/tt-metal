// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// erfinv — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Central-branch polynomial of Giles, "Approximating the erfinv function",
// GPU Computing Gems Jade Edition (2011), ch. 10 (single-precision erfinv):
//   w = -ln(1 - x^2),  erfinv(x) = x * P(w - 2.5)
// The sweep stimulus domain is |x| <= 0.99, where w <= 3.93 < 5 — entirely
// inside the central branch, so the tail branch is never needed.  1 - x^2 is
// computed as (1-x)(1+x) for accuracy near the endpoints; ln comes from the
// typed fresh_ln_positive helper (fresh_cpp/digamma.h, the header of
// record).  Tolerance validation: laneS2-evidence-20260819/fit_s2.py — max
// abs error 1.6e-5 against torch.erfinv over the domain (suite gate
// atol/rtol 0.05).  Golden: torch.erfinv (golden_generators._erfinv),
// Float32 corr contract.
#include <cstdint>

#include "digamma.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_erfinv_fresh_cpp()
{
    // Giles (2011) single-precision central-branch coefficients.
    constexpr float G8 = 2.81022636e-08f;
    constexpr float G7 = 3.43273939e-07f;
    constexpr float G6 = -3.5233877e-06f;
    constexpr float G5 = -4.39150654e-06f;
    constexpr float G4 = 0.00021858087f;
    constexpr float G3 = -0.00125372503f;
    constexpr float G2 = -0.00417768164f;
    constexpr float G1 = 0.246640727f;
    constexpr float G0 = 1.50140941f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x     = sfpi::dst_reg[0];
        const sfpi::vFloat one_m = (1.0f - x) * (1.0f + x);
        const sfpi::vFloat w     = -fresh_ln_positive(one_m) - 2.5f;
        sfpi::vFloat p           = G8;
        p                        = p * w + G7;
        p                        = p * w + G6;
        p                        = p * w + G5;
        p                        = p * w + G4;
        p                        = p * w + G3;
        p                        = p * w + G2;
        p                        = p * w + G1;
        p                        = p * w + G0;
        sfpi::dst_reg[0]         = p * x;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
