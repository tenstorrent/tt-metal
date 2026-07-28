// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace sfpi;

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(std::uint32_t seed) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    init_prng_seed(seed);
}

template <std::uint32_t VALUE, std::uint32_t NEXT>
inline void rand_row() {
    TTI_SFPSHFT((-12) & 0xFFF, VALUE, p_sfpu::LREG4, 5);
    TTI_SFPXOR(0, p_sfpu::LREG4, VALUE, 0);
    TTI_SFPMUL24(VALUE, p_sfpu::LREG6, p_sfpu::LCONST_0, VALUE, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPMOV(0, 9, NEXT, 8);
    TTI_SFPSHFT((-11) & 0xFFF, VALUE, p_sfpu::LREG4, 5);
    TTI_SFPXOR(0, p_sfpu::LREG4, VALUE, 0);
    TTI_SFPMUL24(VALUE, p_sfpu::LREG7, p_sfpu::LCONST_0, VALUE, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPIADD(0, p_sfpu::LREG3, NEXT, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT((-12) & 0xFFF, VALUE, p_sfpu::LREG4, 5);
    TTI_SFPXOR(0, p_sfpu::LREG4, VALUE, 0);

    TTI_SFPSETMAN(0, p_sfpu::LCONST_1, VALUE, 0);
    TTI_SFPADDI(0xbf80 /* -1.0f */, VALUE, 0);
    TTI_SFPMAD(VALUE, p_sfpu::LREG1, p_sfpu::LREG2, VALUE, 0);
    TTI_SFPSTORE(VALUE, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    dst_reg++;
}

template <bool APPROXIMATION_MODE>
inline void rand(std::uint32_t from, std::uint32_t scale) {
    // The hardware PRNG exposes 32 lane streams that are shifted views of the
    // same LFSR. Salt each lane before applying a nonlinear integer finalizer
    // so adjacent output elements do not retain that correlation.
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG3, 0);

    // Truncate established 32-bit avalanche multipliers to the 23-bit
    // mantissa domain. Both remain odd, so multiplication modulo 2^23 is
    // bijective.
    constexpr std::uint32_t mix_constant_0 = 0x5BD1E995 & 0x7FFFFF;
    constexpr std::uint32_t mix_constant_1 = 0x27D4EB2D & 0x7FFFFF;
    TT_SFPLOADI(p_sfpu::LREG6, 10, mix_constant_0 & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG6, 8, mix_constant_0 >> 16);
    TT_SFPLOADI(p_sfpu::LREG7, 10, mix_constant_1 & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG7, 8, mix_constant_1 >> 16);

    // Load scale param to lreg1
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);

    TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);

#pragma GCC unroll 0
    for (int d = 0; d < 4; d++) {
        // Each multiply's independent latency slot prepares the next row.
        rand_row<p_sfpu::LREG0, p_sfpu::LREG5>();
        rand_row<p_sfpu::LREG5, p_sfpu::LREG0>();
    }
}
}  // namespace ckernel::sfpu
