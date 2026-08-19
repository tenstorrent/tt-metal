// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the sqrt op (storm contract: fresh_cpp/README.md).
// The RECIPROCAL=true instantiation also serves the rsqrt row (one shared
// SQRT_23 body, the production kernels share theirs the same way).  Migrated
// verbatim from fresh_cpp_operations.h (Lane BR batch 2); byte-stable
// algorithm, only the file moved.

#include <cstdint>
#include <limits>

namespace ckernel::sfpu
{

// Sqrt / Rsqrt, non-approx bf16 contract (production: shared
// _calculate_sqrt_body_ with the Kokosinski/Moroz integer seed and both
// refinement coefficients parked in vConstIntPrgm0/vConstFloatPrgm1/2).
// Identical SQRT_23-bits algorithm, seed and coefficients local.
template <bool RECIPROCAL, int ITERATIONS>
__attribute__((noinline)) void calculate_sqrt_rsqrt_fresh_cpp()
{
    constexpr int SEED = 0x5f1110a0; // Kokosinski/Moroz SQRT_23 seed (cited paper)
    constexpr float K1 = 2.2825186f;
    constexpr float K2 = 2.2533049f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y       = sfpi::as<sfpi::vFloat>(sfpi::vInt(SEED) - sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1));

        sfpi::vFloat xy                  = x * y;
        const sfpi::vFloat c             = -y * xy;
        const sfpi::vFloat infinity      = std::numeric_limits<float>::infinity();
        const sfpi::vInt infinity_bits   = sfpi::as<sfpi::vInt>(infinity);
        y                                = y * (K1 + c * (K2 + c));
        xy                               = x * y;
        const sfpi::vFloat one_minus_xyy = 1.0f + (-y * xy);

        if constexpr (RECIPROCAL)
        {
            const sfpi::vFloat half_y              = sfpi::addexp(y, -1);
            const sfpi::vInt infinity_minus_x_bits = infinity_bits - sfpi::as<sfpi::vInt>(x);
            v_if (infinity_minus_x_bits != 0 && sfpi::as<sfpi::vInt>(x) != 0)
            {
                y = one_minus_xyy * half_y + y;
            }
            v_else
            {
                // x = 0 -> inf; x = inf -> 0.
                y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits);
            }
            v_endif;
        }
        else
        {
            const sfpi::vFloat half_xy = 0.5f * xy;
            v_if (sfpi::as<sfpi::vInt>(x) < infinity_bits)
            {
                y = one_minus_xyy * half_xy + xy;
            }
            v_endif;
        }

        v_if (x < 0.0f)
        {
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
