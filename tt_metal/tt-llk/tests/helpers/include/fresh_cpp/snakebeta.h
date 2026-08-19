// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the snakebeta op (storm contract,
// fresh_cpp/README.md).  Production: metal ckernel_sfpu_snake_beta.h — the
// sin minimax and range reduction are typed, but the reciprocal is
// sfpu_reciprocal_iter reading its Newton constant 2.0 from vConstFloatPrgm0
// (programmed by snake_beta_init), the polynomial goes through the
// PolynomialEvaluator template, and the loop carries an unroll-8 pin.
// Semantic statement of the golden (x + sin(alpha*x)^2 / beta), bf16-dest
// contract (the measured row's variant, is_fp32_dest_acc_en = false):
// single-stage range reduction a = (alpha*x/pi - round(alpha*x/pi)) * pi
// via the typed vSMag16 round (the production's own idiom, valid for
// |alpha*x| < 32767*pi), the production's OWN bf16-arm sin cubic-in-s
// coefficients (the golden tolerance is fitted to them), sin^2 by squaring
// (even, so no quadrant sign fix), 1/beta by the shared literal-constant
// reciprocal (fresh_cpp/helpers.h fresh_recip<1>, the production's
// RECIP_ITER for 16-bit dest), and the bf16 RNE store.
#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_snake_beta_fresh_cpp(
    const std::uint32_t dst_index_x, const std::uint32_t dst_index_alpha, const std::uint32_t dst_index_beta, const std::uint32_t dst_index_out)
{
    constexpr std::uint32_t tile_rows = 32;
    constexpr float ONE_OVER_PI       = 0.318309886183791f;
    constexpr float PI_F              = 3.141592653589793f;
    // Production bf16-arm sin minimax (cubic in s = a*a).
    constexpr float C2 = -0x1.8b10a4p-13f; // -1.883816730696708e-04
    constexpr float C1 = 0x1.10c2a2p-7f;   //  8.323983289301395e-03
    constexpr float C0 = -0x1.5554a4p-3f;  // -1.6666534543037415e-01

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x     = sfpi::dst_reg[dst_index_x * tile_rows];
        const sfpi::vFloat alpha = sfpi::dst_reg[dst_index_alpha * tile_rows];
        const sfpi::vFloat beta  = sfpi::dst_reg[dst_index_beta * tile_rows];

        // Range-reduce alpha*x into (-pi/2, pi/2] around the nearest
        // multiple of pi; sin^2 is pi-periodic and even, so no sign fix.
        const sfpi::vFloat ax         = alpha * x;
        const sfpi::vFloat ax_over_pi = ax * ONE_OVER_PI;
        const sfpi::vSMag16 k         = sfpi::convert<sfpi::vSMag16>(ax_over_pi, sfpi::RoundMode::Nearest);
        const sfpi::vFloat k_f        = sfpi::convert<sfpi::vFloat>(k, sfpi::RoundMode::Nearest);
        const sfpi::vFloat a          = (ax_over_pi - k_f) * PI_F;

        const sfpi::vFloat s = a * a;
        const sfpi::vFloat r = (a * s) * ((C2 * s + C1) * s + C0) + a;

        const sfpi::vFloat inv_beta              = fresh_recip<1>(beta);
        sfpi::vFloat result                      = (r * r) * inv_beta + x;
        result                                   = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[dst_index_out * tile_rows] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
