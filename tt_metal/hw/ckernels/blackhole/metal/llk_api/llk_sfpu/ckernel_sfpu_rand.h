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

constexpr std::uint32_t sfpshft_mod1_arg_imm = 1;
constexpr std::uint32_t sfpshft_mod1_arg_imm_use_vc = sfpshft_mod1_arg_imm | 4;
constexpr std::uint32_t dense_uniform_exponent_bias = 126;
constexpr std::uint32_t dense_uniform_exponent_bias_reg = p_sfpu::LREG7;

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
    // rand_tile is a shared API and other SFPU operations may clobber every
    // mutable LREG between calls. Reconstruct a lane-specific salt from the
    // read-only lane ID at the start of each tile instead of carrying it in an
    // LREG. This is the conventional xorshift32 transformation.
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(13, p_sfpu::LREG3, p_sfpu::LREG0, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPSHFT((-17) & 0xFFF, p_sfpu::LREG3, p_sfpu::LREG0, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(5, p_sfpu::LREG3, p_sfpu::LREG0, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
}

template <std::uint32_t VALUE, std::uint32_t EXPONENT>
inline void uint32_to_dense_uniform_fp32() {
    // A uniform word has a geometric leading-zero count. Use it as the
    // binade index and the low 23 bits as the FP32 mantissa. The zero word
    // maps to the 2^-33 binade, deliberately avoiding subnormals and zero.
    TTI_SFPLZ(0, VALUE, EXPONENT, 0);
    TTI_SFPIADD(
        0,
        dense_uniform_exponent_bias_reg,
        EXPONENT,
        sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    // The positive bias also supplies a zero sign bit; its temporary exponent
    // is overwritten by SFPSETEXP.
    TTI_SFPSETMAN(0, dense_uniform_exponent_bias_reg, VALUE, 0);
    TTI_SFPSETEXP(0, VALUE, EXPONENT, 0);
}

inline void mix_uint32_fast() {
    // A shorter bijective ARX permutation. The two modular additions provide
    // the nonlinearity that a pure xorshift lacks, while alternating right
    // and left shifts diffuses both the exponent and mantissa source bits.
    TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG5, p_sfpu::LREG0, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);
    TTI_SFPSHFT(3, p_sfpu::LREG5, p_sfpu::LREG0, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT((-4) & 0xFFF, p_sfpu::LREG5, p_sfpu::LREG0, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);
    TTI_SFPSHFT(10, p_sfpu::LREG5, p_sfpu::LREG0, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT((-15) & 0xFFF, p_sfpu::LREG5, p_sfpu::LREG0, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
}

inline void rand_row() {
    mix_uint32_fast();
    rand_prng<p_sfpu::LREG5>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);
    uint32_to_dense_uniform_fp32<p_sfpu::LREG0, p_sfpu::LREG6>();
    // Add a rounding bit. Carry out of an all-ones mantissa naturally
    // reaches the adjacent binade, giving boundary values half the
    // rounding basin of interior values.
    TTI_SFPSHFT((-31) & 0xFFF, p_sfpu::LREG5, p_sfpu::LREG4, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0);
    // SFPMAD has two-cycle latency. Advance the destination counter in its
    // dependency slot, then compensate for that early increment in the store.
    dst_reg++;
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_7, (-2) & 0x3FF);
}

template <bool APPROXIMATION_MODE>
inline void rand(std::uint32_t from, std::uint32_t scale) {
    // Load scale param to lreg1
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, from >> 16);
    TT_SFPLOADI(dense_uniform_exponent_bias_reg, sfpi::SFPLOADI_MOD0_USHORT, dense_uniform_exponent_bias);

    make_lane_salt();
    rand_prng<p_sfpu::LREG5>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_CC_NONE);

    // One row fits in the 32-entry replay buffer. Record and execute it once,
    // then replay it for the remaining rows without scalar loop-control gaps.
    constexpr std::uint32_t row_instruction_count = 21;
    TTI_REPLAY(0, row_instruction_count, 1, 1);
    rand_row();
#pragma GCC unroll 7
    for (int d = 1; d < 8; d++) {
        TTI_REPLAY(0, row_instruction_count, 0, 0);
    }
}
}  // namespace ckernel::sfpu
