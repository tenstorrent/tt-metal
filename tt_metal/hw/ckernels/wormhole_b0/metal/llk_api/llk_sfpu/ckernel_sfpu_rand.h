// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(std::uint32_t seed) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    init_prng_seed(seed);
}

template <bool APPROXIMATION_MODE>
inline void rand(std::uint32_t from, std::uint32_t scale) {
    // The hardware PRNG exposes 32 lane streams that are shifted views of the
    // same LFSR. Salt each lane before applying a nonlinear integer finalizer
    // so adjacent output elements do not retain that correlation.
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG3, 0);

    // Load scale param to lreg1
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);

    TTI_SFPMOV(0, 9, p_sfpu::LREG5, 8);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate a random word, then avalanche its bits together with the
        // lane salt. This is Thomas Wang's 32-bit integer mix expressed with
        // shifts, additions, and XORs so it is efficient on both architectures.
        TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPSHFT(15, 0, p_sfpu::LREG0, 1);
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSHFT2((-12) & 0xFFF, 0, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
        TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPSHFT(2, 0, p_sfpu::LREG0, 1);
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPSHFT((-4) & 0xFFF, 0, p_sfpu::LREG0, 1);
        TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
        TTI_SFPSHFT(3, 0, p_sfpu::LREG0, 1);
        TTI_SFPSHFT(11, 0, p_sfpu::LREG5, 1);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSHFT2((-16) & 0xFFF, 0, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
        TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);

        // Combine the mixed mantissa with the sign and exponent of 1.0f.
        TTI_SFPSETMAN(0, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);

        // -1 to ensure the float is within the range [0, 1).
        // lreg0 = lreg0 - 1
        TTI_SFPADDI(0xbf80 /*-1*/, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, 9, p_sfpu::LREG5, 8);

        // Scale the float from [0, 1) to [from, from + scale)
        // lreg0 = lreg0 * scale + from
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);

        TTI_SFPSTORE(0, InstrModLoadStore::FP32, 3, 0);
        dst_reg++;
    }
}
}  // namespace ckernel::sfpu
