// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `addrsqrt-fresh` coverage row (metal
// experimental ckernel_sfpu_add_rsqrt.h calculate_add_rsqrt, corpus manifest
// class D-ABSENT — zero dispatch anywhere).  Mathematical definition
// (RMSNorm epsilon idiom): y = 1 / sqrt(x + eps), eps a compile-time fp32
// scalar.  The rsqrt core restates the established SQRT_23-bits fresh
// algorithm (fresh_cpp/rsqrt.h RECIPROCAL arm: Kokosinski/Moroz integer seed
// + quadratic refinement + one Newton half-step), applied to the shifted
// argument; bf16 RNE store per the fresh float-body convention.
#include <cstdint>
#include <limits>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_add_rsqrt_fresh_cpp(const std::uint32_t eps_bits)
{
    constexpr int SEED = 0x5f1110a0; // Kokosinski/Moroz SQRT_23 seed (cited paper)
    constexpr float K1 = 2.2825186f;
    constexpr float K2 = 2.2533049f;

    const sfpi::vFloat eps = sfpi::as<sfpi::vFloat>(sfpi::vInt(static_cast<int>(eps_bits)));
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0] + eps;
        sfpi::vFloat y       = sfpi::as<sfpi::vFloat>(sfpi::vInt(SEED) - sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1));

        sfpi::vFloat xy                  = x * y;
        const sfpi::vFloat c             = -y * xy;
        const sfpi::vFloat infinity      = std::numeric_limits<float>::infinity();
        const sfpi::vInt infinity_bits   = sfpi::as<sfpi::vInt>(infinity);
        y                                = y * (K1 + c * (K2 + c));
        xy                               = x * y;
        const sfpi::vFloat one_minus_xyy = 1.0f + (-y * xy);

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
