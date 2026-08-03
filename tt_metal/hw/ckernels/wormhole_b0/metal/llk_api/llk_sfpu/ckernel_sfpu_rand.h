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

constexpr std::uint32_t sfpshft_mod1_arg_imm = 1;

template <std::uint32_t DEST>
inline void rand_prng() {
    constexpr std::uint32_t prng_source = 9;
    constexpr std::uint32_t sfpmov_mod1_from_special = 8;
    TTI_SFPMOV(0, prng_source, DEST, sfpmov_mod1_from_special);
}

template <bool APPROXIMATION_MODE>
inline void rand_init(std::uint32_t seed) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // The all-ones state is the lock-up state of the hardware XNOR LFSR.
    if (seed == 0xFFFFFFFF) {
        seed = 0xFFFFFFFE;
    }
    init_prng_seed(seed);
}

template <std::uint32_t VALUE, std::uint32_t EXPONENT>
inline void uint32_to_dense_uniform_fp32() {
    // A uniform word has a geometric leading-zero count. Use it as the
    // binade index and the low 23 bits as the FP32 mantissa. The zero word
    // maps to the 2^-33 binade, deliberately avoiding subnormals and zero.
    TTI_SFPLZ(0, VALUE, EXPONENT, 0);
    TTI_SFPIADD(0, p_sfpu::LCONST_0, EXPONENT, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(126, EXPONENT, EXPONENT, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETMAN(0, p_sfpu::LCONST_1, VALUE, 0);
    TTI_SFPSETEXP(0, VALUE, EXPONENT, 0);
}

template <bool APPROXIMATION_MODE>
inline void rand(std::uint32_t from, std::uint32_t scale) {
    // The hardware PRNG exposes 32 lane streams that are shifted views of the
    // same LFSR. Salt each lane before applying a nonlinear integer finalizer
    // so adjacent output elements do not retain that correlation.
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG3, 0);

    // Load scale param to lreg1
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, from >> 16);
    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 1);

    rand_prng<p_sfpu::LREG5>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate a random word, then avalanche its bits together with the
        // lane salt. This is Thomas Wang's 32-bit integer mix expressed with
        // shifts, additions, and XORs so it is efficient on both architectures.
        TTI_SFPNOT(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
        TTI_SFPSHFT(15, 0, p_sfpu::LREG5, sfpshft_mod1_arg_imm);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_CC_NONE);
        // In immediate mode, SFPSHFT2 selects its source with Imm12 & 0xf.
        // The low nibble of -12 is 4, so this computes LREG5 = LREG4 >> 12.
        TTI_SFPSHFT2((-12) & 0xFFF, 0, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
        TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPSHFT(2, 0, p_sfpu::LREG0, sfpshft_mod1_arg_imm);
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPSHFT((-4) & 0xFFF, 0, p_sfpu::LREG0, sfpshft_mod1_arg_imm);
        TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
        TTI_SFPSHFT(3, 0, p_sfpu::LREG0, sfpshft_mod1_arg_imm);
        TTI_SFPSHFT(11, 0, p_sfpu::LREG5, sfpshft_mod1_arg_imm);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        // The low nibble of -16 is 0, so this computes LREG4 = LREG0 >> 16.
        TTI_SFPSHFT2((-16) & 0xFFF, 0, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
        TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);

        rand_prng<p_sfpu::LREG5>();
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);
        uint32_to_dense_uniform_fp32<p_sfpu::LREG0, p_sfpu::LREG6>();
        // Add a rounding bit. Carry out of an all-ones mantissa naturally
        // reaches the adjacent binade, giving boundary values half the
        // rounding basin of interior values.
        TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG4, 0);
        TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_CC_NONE);

        // Scale the base sample into the requested range.
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0);

        TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}
}  // namespace ckernel::sfpu
