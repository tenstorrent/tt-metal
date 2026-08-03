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

inline void make_lane_salt() {
    // Reconstruct the salt for every tile so rand_tile remains valid after
    // arbitrary SFPU operations have clobbered the mutable LREGs. This is the
    // conventional xorshift32 transformation.
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG3, 0);

    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(13, 0, p_sfpu::LREG0, sfpshft_mod1_arg_imm);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);

    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    TTI_SFPSHFT((-17) & 0xFFF, 0, p_sfpu::LREG0, sfpshft_mod1_arg_imm);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);

    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(5, 0, p_sfpu::LREG0, sfpshft_mod1_arg_imm);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
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

inline void mix_uint32_fast() {
    // The same bijective ARX permutation used on Blackhole, scheduled around
    // Wormhole's in-place SFPSHFT and SFPSHFT2 immediate-source semantics.
    // LREG1 holds x on entry and LREG0 holds the result on exit.
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    // (-16 & 15) == 0: LREG4 = LREG0 >> 16.
    TTI_SFPSHFT2((-16) & 0xFFF, 0, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);

    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(3, 0, p_sfpu::LREG0, sfpshft_mod1_arg_imm);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPSHFT((-4) & 0xFFF, 0, p_sfpu::LREG4, sfpshft_mod1_arg_imm);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);

    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG1, 0);
    TTI_SFPSHFT(10, 0, p_sfpu::LREG1, sfpshft_mod1_arg_imm);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);

    // (-15 & 15) == 1: LREG0 = LREG1 >> 15.
    TTI_SFPSHFT2((-15) & 0xFFF, 0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
}

inline void rand_row() {
    mix_uint32_fast();
    // LREG1 becomes the following row's input. Its high bit also provides the
    // current row's FP32 rounding bit.
    rand_prng<p_sfpu::LREG1>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
    uint32_to_dense_uniform_fp32<p_sfpu::LREG0, p_sfpu::LREG6>();
    // (-31 & 15) == 1: LREG4 = LREG1 >> 31.
    TTI_SFPSHFT2((-31) & 0xFFF, 0, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG6, 0);
    // Wormhole requires an explicit independent cycle after SFPMAD. Advance
    // the destination counter in that slot and compensate in the store.
    dst_reg++;
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_3, (-2) & 0x3FF);
}

template <bool APPROXIMATION_MODE>
inline void rand(std::uint32_t from, std::uint32_t scale) {
    // Load scale param to lreg5
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, from >> 16);

    make_lane_salt();
    rand_prng<p_sfpu::LREG1>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);

    constexpr std::uint32_t row_instruction_count = 26;
    TTI_REPLAY(0, row_instruction_count, 1, 1);
    rand_row();
#pragma GCC unroll 7
    for (int d = 1; d < 8; d++) {
        TTI_REPLAY(0, row_instruction_count, 0, 0);
    }
}
}  // namespace ckernel::sfpu
