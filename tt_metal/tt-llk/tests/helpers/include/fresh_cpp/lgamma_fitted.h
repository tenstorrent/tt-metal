// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// lgamma on the production kernel's stage contract (eltwise_unary/lgamma.h):
// the result is lgamma(z) with z = (x < 0.5) ? 1 - x : x, the reflection term
// being applied by a later stage.  Every z the stage sees is >= 0.5, where
//   lgamma(z) = (z - 1/2) (ln z - 1) + R(z),
//   R(z)      = mu(z) + (ln(2 pi) - 1) / 2,
// mu being Binet's remainder of Stirling's series (mu(z) ~ 1/(12 z)).
//   - ln z = e ln2 + ln m, z = m 2^e with m in [1, 2); ln m = t + t^2 q(t),
//     t = m - 1, q of degree 3 (max abs error of ln m 1.9e-5).
//   - R is a [2/2] rational in z fitted on [0.5, 4096] (max abs error 2.2e-5),
//     interpolating exactly at z = 1 and z = 2.  Above 4096 its argument is
//     clamped: mu(z) < 2.1e-5 there against lgamma(z) > 3e4.
// N0 is tuned so the fp32 evaluation gives exactly zero at the roots z = 1
// and z = 2.  +inf (either sign of x) gives +inf; NaN propagates.
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_lgamma_fitted_cpp()
{
    constexpr float LN2 = 0.6931471805599453f;
    // q(t), ascending; Q1..Q3 are exact fp16 values.
    constexpr float Q0 = -0.496166766f;
    constexpr float Q1 = 0.301269531f;
    constexpr float Q2 = -0.149291992f;
    constexpr float Q3 = 0.0373535156f;
    // R(z) = (N0 + N1 z + N2 z^2) / (D0 + D1 z + z^2); N1, N2, D1 are exact
    // fp16 values.
    constexpr float N0      = 0.0816658735f;
    constexpr float N1      = 0.368896484f;
    constexpr float N2      = 0.418945312f;
    constexpr float D0      = 0.0568864942f;
    constexpr float D1      = 0.682128906f;
    constexpr float R_Z_MAX = 4096.0f;

    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat z       = x;
        v_if (x < 0.5f)
        {
            z = 1.0f - x;
        }
        v_endif;

        const sfpi::vFloat t = sfpi::setexp(z, 127) - 1.0f;
        sfpi::vFloat q       = Q3;
        q                    = q * t + Q2;
        q                    = q * t + Q1;
        q                    = q * t + Q0;
        const sfpi::vFloat e = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(sfpi::exexp(z)), sfpi::RoundMode::Nearest);
        const sfpi::vFloat ln_z_minus_1 = e * LN2 + (t * (q * t + 1.0f) - 1.0f);

        const sfpi::vFloat zr  = sfpi::min(z, R_Z_MAX);
        const sfpi::vFloat num = (N2 * zr + N1) * zr + N0;
        const sfpi::vFloat den = (zr + D1) * zr + D0;
        const sfpi::vFloat r   = num * fresh_recip_hwseed(den);

        const sfpi::vFloat y = (z - 0.5f) * ln_z_minus_1 + r;
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
