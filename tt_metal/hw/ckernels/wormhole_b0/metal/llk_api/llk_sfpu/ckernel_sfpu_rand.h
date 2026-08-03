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
constexpr std::uint32_t sfpsetsgn_mod1_arg_imm = 1;
constexpr std::uint32_t one_over_2_pow_31_bf16 = 0x3000;

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
    // SFPCAST converts the low 31 bits as a sign-magnitude integer and rounds
    // directly to FP32. Clear its arbitrary sign, then normalise by 2^-31.
    // This gives a correctly rounded 31-bit uniform grid, including both
    // mantissa parities and the half-width upper-bound rounding basin.
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG6, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPSETSGN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpsetsgn_mod1_arg_imm);
    TTI_SFPMULI(one_over_2_pow_31_bf16, p_sfpu::LREG6, 0);
    // Advance the PRNG while SFPMULI's result becomes available.
    rand_prng<p_sfpu::LREG1>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG6, 0);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_3, 0);
    dst_reg++;
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

    constexpr std::uint32_t row_instruction_count = 23;
    TTI_REPLAY(0, row_instruction_count, 1, 1);
    rand_row();
#pragma GCC unroll 7
    for (int d = 1; d < 8; d++) {
        TTI_REPLAY(0, row_instruction_count, 0, 0);
    }
}
}  // namespace ckernel::sfpu
