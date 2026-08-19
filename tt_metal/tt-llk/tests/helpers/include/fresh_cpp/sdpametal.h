// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the sdpametal op's ExpPoly vehicle (storm
// contract, fresh_cpp/README.md).  Production: metal experimental
// ckernel_sfpu_sdpa.h calculate_exponential_polynomial — an entirely raw
// TTI stream over hard-pinned LREG0..7 (coefficient preload via SFPLOADI
// pairs, per-iteration SFPMAD chains, an addr_mod_t MMIO write inside the
// measured zone).  Semantic statement of the same golden math
// (exp(scale*x) = 2**n * e**r with n = RNE-int8(scale*x/ln2) and
// r = scale*x - n*ln2, the production's OWN minimax coefficients — the
// golden tolerance is fitted to them), one datum at a time in plain typed
// C++.  The int8/uint8 round steps are the production's SFP_STOCH_RND ops
// in round-nearest-even mode (rnd_mode 0 == SFPSTOCHRND_RND_EVEN; INT8 is
// the SMAG8 encoding), stated through the typed converts.  Column walk and
// footprint mirror the typed sibling calculate_recip_first_column (4
// half-face iterations, dst_reg += 2, VectorMode::C) which is proven
// against the same golden and TRANSFORMED_COLS footprint.
#include <cstdint>

namespace ckernel::sfpu
{

template <bool IS_FP32_DEST_ACC_EN, std::uint16_t SCALE_BF16>
__attribute__((noinline)) void calculate_sdpa_exp_poly_fresh_cpp()
{
    constexpr float LN2_RECIP = 1.44269504088896340736f;
    constexpr float M_LN2     = -0.69314718055994530942f;

    // Production minimax e**r coefficients: degree 4 on the fp32-dest arm,
    // degree 2 on the bf16-dest arm (the same arm split the production body
    // selects through DST_ACCUM_MODE).
    constexpr float C0 = IS_FP32_DEST_ACC_EN ? 1.0000001510806179002040134468008959160576106495165f : 0.999848792924395313327307061545061386175496934006f;
    constexpr float C1 = IS_FP32_DEST_ACC_EN ? 0.99996228117047652035114096488703457970402030983204f : 1.01508760098521056684783640695492761469306929535975f;
    constexpr float C2 = IS_FP32_DEST_ACC_EN ? 0.49998365704615426417337683145647067790385638465486f : 0.50628367056745568861842335616023694454759126020461f;
    constexpr float C3 = 0.16792157982882225102649214918047336097544632172075f;
    constexpr float C4 = 4.1959439860014343843000081999668024587178974865521e-2f;

    const sfpi::vFloat scale = Converter::as_float(static_cast<std::uint32_t>(SCALE_BF16) << 16);

    constexpr int ITERATIONS_HALF_FACE = 4;
    for (int d = 0; d < ITERATIONS_HALF_FACE; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0] * scale;

        // n = round-to-nearest-even int8 of x/ln2 (the production
        // SFP_STOCH_RND FP32->INT8/SMAG8 step), back to float by the RNE
        // cast; r = x - n*ln2 is the poly's reduced argument.
        const sfpi::vFloat nf  = x * LN2_RECIP;
        const auto n           = sfpi::convert<sfpi::vSMag8>(nf, sfpi::RoundMode::Nearest);
        const sfpi::vFloat n_f = sfpi::convert<sfpi::vFloat>(n, sfpi::RoundMode::Nearest);
        const sfpi::vFloat r   = n_f * M_LN2 + x;

        sfpi::vFloat y;
        if constexpr (IS_FP32_DEST_ACC_EN)
        {
            y = (((C4 * r + C3) * r + C2) * r + C1) * r + C0;
        }
        else
        {
            y = (C2 * r + C1) * r + C0;
        }

        // 2**n by the biased exponent: n + 127 rounded to uint8 (the
        // production SFP_STOCH_RND FP32->UINT8 step) into setexp; lanes
        // whose biased exponent is not positive underflow to zero.
        const sfpi::vFloat np = n_f + 127.0f;
        const auto e          = sfpi::convert<sfpi::vUInt8>(np, sfpi::RoundMode::Nearest);
        y                     = y * sfpi::setexp(sfpi::vFloat(1.0f), sfpi::as<sfpi::vInt>(e));
        v_if (np <= 0.0f)
        {
            y = 0.0f;
        }
        v_endif;

        if constexpr (!IS_FP32_DEST_ACC_EN)
        {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg += 2;
    }
}

} // namespace ckernel::sfpu
