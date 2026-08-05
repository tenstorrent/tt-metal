// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel::sfpu {

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
    // arbitrary SFPU operations have clobbered the mutable LREGs. LTILEID is
    // the ISA-defined LREG15, whose lane i contains 2*i. Offset LTILEID first
    // so lane zero also receives a nonzero salt. The immediate shift encodings
    // select LREG4 and LREG7 as their respective sources.
    TTI_SFPIADD(
        (-1800) & 0xFFF, p_sfpu::LTILEID, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT2(20, 0, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG7, 0);
    TTI_SFPSHFT2((-9) & 0xFFF, 0, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG3, 0);
}

inline void begin_mix_uint32_fast() {
    // A bijective ARX permutation scheduled around SFPSHFT2's independent
    // source and destination registers. LREG4, LREG7, LREG12, and LREG13 hold
    // -14, 8, -5, and 11, respectively. LREG1 holds x on entry and LREG0 holds
    // the result on exit.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
}

inline void finish_mix_uint32_fast() {
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
}

inline void rand_row() {
    finish_mix_uint32_fast();
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
    // Prime the following row's mixer in SFPMAD's dependency slot. This reads
    // LREG1 and writes LREG0, independently of SFPMAD's LREG6 result. The
    // speculative prime after the final row is harmless.
    begin_mix_uint32_fast();
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_3, 0);
    dst_reg++;
}

template <bool APPROXIMATION_MODE>
inline void rand(std::uint32_t from, std::uint32_t scale) {
    make_lane_salt();
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_SHORT, 8);

    // Load scale param to lreg5
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, from >> 16);

    // Keep the remaining shift counts outside the replayed row body.
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_SHORT, (-14) & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_SHORT, (-5) & 0xFFFF);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_SHORT, 11);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);

    rand_prng<p_sfpu::LREG1>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
    begin_mix_uint32_fast();

    constexpr std::uint32_t row_instruction_count = 18;
    TTI_REPLAY(0, row_instruction_count, 1, 1);
    rand_row();
#pragma GCC unroll 7
    for (int d = 1; d < 8; d++) {
        TTI_REPLAY(0, row_instruction_count, 0, 0);
    }
}
}  // namespace ckernel::sfpu
