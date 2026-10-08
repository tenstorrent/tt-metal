// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Index 9 is the SFPMOV mod 0x8 *read* map (PRNG), not the SFPCONFIG write map in assembly.yaml.
// sfpi defines SFPCONFIG_SRC_RAND for Quasar; test_rand_quasar's distinct checks catch a wrong index.
constexpr std::uint32_t PRNG_RS_INDEX = sfpi::SFPCONFIG_SRC_RAND;
constexpr std::uint32_t SFPMOV_MOD1_FROM_RS = sfpi::SFPMOV_MOD1_CONFIG;
constexpr std::uint32_t LANE_ID_LREG = sfpi::CREG_IDX_TILEID;
constexpr std::uint32_t MIX_MULTIPLIER_LREG = sfpi::CREG_IDX_0P837300003;  // 0x3F56594B, low 23 bits odd

// sfpi's SFPSHFT_MOD1_SHIFT_IMM (0) decodes as "shift by lreg_c", so spell out shift-by-imm12.
constexpr std::uint32_t SFPSHFT_MOD1_IMM_SRC_C = 0b101;
constexpr std::uint32_t SFPIADD_MOD1_IMM_NO_CC = sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE;
constexpr std::uint32_t SFPIADD_MOD1_REG_NO_CC = sfpi::SFPIADD_MOD1_CC_NONE;
constexpr std::uint32_t SFPSETSGN_MOD1_SIGN_FROM_IMM = 1;

constexpr std::uint32_t FP16B_TWO_POW_NEG_31 = 0x3000;

constexpr std::uint32_t LANE_SALT_OFFSET = 407;  // gives lane 0 a nonzero salt
constexpr std::uint32_t LANE_SALT_SHIFT_A = 14;
constexpr std::uint32_t LANE_SALT_SHIFT_B = 6;

// 12-bit two's-complement shift immediates; negative = right shift.
constexpr std::uint32_t MIX_SHIFT_R8 = (-8) & 0xFFF;
constexpr std::uint32_t MIX_SHIFT_R16 = (-16) & 0xFFF;
constexpr std::uint32_t MIX_SHIFT_L8 = 8;
constexpr std::uint32_t MIX_SHIFT_R14 = (-14) & 0xFFF;

constexpr std::uint32_t FP32_EXP_SHIFT = 23;
constexpr std::uint32_t FP32_EXP_MASK = 0xFF;
constexpr std::uint32_t NORMALIZATION_EXPONENT = 31;

constexpr std::uint32_t FP32_HALF_BITS = 16;
constexpr std::uint32_t FP32_HALF_MASK = 0xFFFF;

// All-ones is the XNOR LFSR lock-up state.
constexpr std::uint32_t PRNG_LFSR_LOCKUP_SEED = 0xFFFFFFFF;
constexpr std::uint32_t PRNG_LFSR_LOCKUP_REPAIR = 0xFFFFFFFE;
// The seed bus is not ordered with the instruction FIFO and has no busy flag; Quasar seeder RTL review
// gives >= 1600 (Blackhole uses 600). test_rand_seed_quasar fails if this is too short.
constexpr std::uint32_t PRNG_SEED_SETTLE_NOPS = 1600;

// Body = head (finish_mix + SFPCAST + SFPSETSGN) + tail (SFPIADD, SFPMAD, SFPSHFT, SFPSTORE); the
// per-row-normalize path issues SFPMULI between the two replays.
constexpr std::uint32_t RAND_SHIFT_XOR_LEN = 2;
constexpr std::uint32_t RAND_FINISH_MIX_LEN = 1 + RAND_SHIFT_XOR_LEN + 3 + 2 * RAND_SHIFT_XOR_LEN;
constexpr std::uint32_t RAND_TO_UNIT_LEN = 2;
constexpr std::uint32_t RAND_HEAD_LEN = RAND_FINISH_MIX_LEN + RAND_TO_UNIT_LEN;
constexpr std::uint32_t RAND_TAIL_LEN = 4;
constexpr std::uint32_t RAND_REPLAY_SLOT = 0;
constexpr std::uint32_t RAND_TAIL_SLOT = RAND_REPLAY_SLOT + RAND_HEAD_LEN;
constexpr std::uint32_t RAND_BODY_LEN = RAND_HEAD_LEN + RAND_TAIL_LEN;
constexpr std::uint32_t RAND_REPLAY_DEPTH = 32;
static_assert(RAND_REPLAY_SLOT + RAND_BODY_LEN <= RAND_REPLAY_DEPTH, "the recorded body must fit the replay buffer");

// dst = (src << SHIFT) ^ src
template <std::uint32_t SHIFT, std::uint32_t SRC, std::uint32_t DST>
inline void _rand_shift_xor_() {
    TTI_SFPSHFT(SHIFT, SRC, DST, SFPSHFT_MOD1_IMM_SRC_C);
    TTI_SFPXOR(SRC, DST);
}

// x ^= x << SHIFT, through TMP
template <std::uint32_t SHIFT, std::uint32_t X, std::uint32_t TMP>
inline void _rand_xor_shift_in_place_() {
    TTI_SFPSHFT(SHIFT, X, TMP, SFPSHFT_MOD1_IMM_SRC_C);
    TTI_SFPXOR(TMP, X);
}

template <std::uint32_t LREG>
inline void _rand_load_fp32_(const std::uint32_t bits) {
    TT_SFPLOADI(LREG, sfpi::SFPLOADI_MOD0_LOWER, bits & FP32_HALF_MASK /* imm16 */);
    TT_SFPLOADI(LREG, sfpi::SFPLOADI_MOD0_UPPER, bits >> FP32_HALF_BITS /* imm16 */);
}

// Bijective finalizer. Entry: LREG0 = x, LREG5 = x >> 8. Exit: LREG4 = mixed, LREG0 = next PRNG word.
inline void _rand_finish_mix_() {
    TTI_SFPXOR(p_sfpu::LREG5, p_sfpu::LREG0);
    _rand_shift_xor_<MIX_SHIFT_R16, p_sfpu::LREG0, p_sfpu::LREG5>();
    TTI_SFPMUL24(p_sfpu::LREG5, MIX_MULTIPLIER_LREG, p_sfpu::LREG4, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPMOV(PRNG_RS_INDEX, p_sfpu::LREG0, SFPMOV_MOD1_FROM_RS);  // fills the MUL24 latency shadow
    TTI_SFPSETMAN(0 /* imm12_math */, p_sfpu::LREG5, p_sfpu::LREG4, 0 /* instr_mod1 */);  // restore x[31:23]
    _rand_xor_shift_in_place_<MIX_SHIFT_L8, p_sfpu::LREG4, p_sfpu::LREG5>();
    _rand_xor_shift_in_place_<MIX_SHIFT_R14, p_sfpu::LREG4, p_sfpu::LREG5>();
}

inline void _rand_row_head_() {
    _rand_finish_mix_();
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);  // bit 31 lands in the sign
    TTI_SFPSETSGN(0 /* imm12_math: positive */, p_sfpu::LREG6, p_sfpu::LREG6, SFPSETSGN_MOD1_SIGN_FROM_IMM);
}

// SFPIADD and SFPSHFT start the next row's mix inside the MAD's latency shadow.
inline void _rand_row_tail_() {
    TTI_SFPIADD(0 /* imm12_math */, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_REG_NO_CC);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0 /* instr_mod1 */);
    TTI_SFPSHFT(MIX_SHIFT_R8, p_sfpu::LREG0, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);
    TTI_SFPSTORE(p_sfpu::LREG6, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /* done */, 0 /* dest_reg_addr */);
}

// Reseeds every lane's PRNG. Calling it again restarts the stream, so call it once per stream.
template <bool APPROXIMATION_MODE /*unused*/>
inline void init_rand(std::uint32_t seed) {
    if (seed == PRNG_LFSR_LOCKUP_SEED) {
        seed = PRNG_LFSR_LOCKUP_REPAIR;
    }
    volatile std::uint32_t* cfg = (volatile std::uint32_t*)TENSIX_CFG_BASE;
    cfg[PRNG_SEED_Seed_Val_ADDR32] = seed;
    for (std::uint32_t i = 0; i < PRNG_SEED_SETTLE_NOPS; i++) {
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    }
}

// Per-lane salt in LREG3 (clobbers LREG4/5); rebuilt every face since another op may hold LREG3.
inline void _rand_make_lane_salt_() {
    TTI_SFPIADD(LANE_SALT_OFFSET, LANE_ID_LREG, p_sfpu::LREG4, SFPIADD_MOD1_IMM_NO_CC);
    _rand_shift_xor_<LANE_SALT_SHIFT_A, p_sfpu::LREG4, p_sfpu::LREG5>();
    _rand_shift_xor_<LANE_SALT_SHIFT_B, p_sfpu::LREG5, p_sfpu::LREG3>();
}

/**
 * @brief Overwrite one face of Dest with uniform draws from [from, from + scale] (FP32 bit patterns).
 * @note Programs ADDR_MOD_6 and re-records replay slots 0-15 every call (record-only, TEN-4690), so
 *       other ops may use either in between. Clobbers LREG0-LREG6.
 */
template <bool APPROXIMATION_MODE /*unused*/, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_rand(const std::uint32_t from, std::uint32_t scale) {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);

    // Fold 2^-31 into scale unless that would underflow its exponent or it is Inf/NaN.
    const std::uint32_t exp = (scale >> FP32_EXP_SHIFT) & FP32_EXP_MASK;
    const bool normalize_per_row = (exp <= NORMALIZATION_EXPONENT) || (exp == FP32_EXP_MASK);
    if (!normalize_per_row) {
        scale -= NORMALIZATION_EXPONENT << FP32_EXP_SHIFT;
    }

    _rand_load_fp32_<p_sfpu::LREG1>(scale);
    _rand_load_fp32_<p_sfpu::LREG2>(from);

    _rand_make_lane_salt_();

    load_replay_buf<RAND_REPLAY_SLOT, RAND_BODY_LEN, false /* exec_while_loading */>([] {
        _rand_row_head_();
        _rand_row_tail_();
    });

    // Prime row 0.
    TTI_SFPMOV(PRNG_RS_INDEX, p_sfpu::LREG0, SFPMOV_MOD1_FROM_RS);
    TTI_SFPIADD(0 /* imm12_math */, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_REG_NO_CC);
    TTI_SFPSHFT(MIX_SHIFT_R8, p_sfpu::LREG0, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);

    // Unrolled so loop control does not open a gap between replays.
    if (normalize_per_row) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_REPLAY(RAND_REPLAY_SLOT, RAND_HEAD_LEN, 0 /* last */, 0 /* set_mutex */, 0 /* exec */, 0 /* load */);
            TTI_SFPMULI(FP16B_TWO_POW_NEG_31, p_sfpu::LREG6, 0 /* instr_mod1 */);
            TTI_REPLAY(RAND_TAIL_SLOT, RAND_TAIL_LEN, 0 /* last */, 0 /* set_mutex */, 0 /* exec */, 0 /* load */);
        }
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_REPLAY(RAND_REPLAY_SLOT, RAND_BODY_LEN, 0 /* last */, 0 /* set_mutex */, 0 /* exec */, 0 /* load */);
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
