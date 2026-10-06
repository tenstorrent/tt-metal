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

// Register-view indices read by the row body.
constexpr std::uint32_t PRNG_RS_INDEX = sfpi::SFPCONFIG_SRC_RAND;          // 9: SFPU Status view PRNG counter
constexpr std::uint32_t SFPMOV_MOD1_FROM_RS = sfpi::SFPMOV_MOD1_CONFIG;    // 8: SFPMOV reads RS[lreg_c]
constexpr std::uint32_t LANE_ID_LREG = sfpi::CREG_IDX_TILEID;              // 15: {col[3:0], row[1:0]} lane ID
constexpr std::uint32_t MIX_MULTIPLIER_LREG = sfpi::CREG_IDX_0P837300003;  // 8: 0x3F56594B, low 23 bits odd

// Instruction modifiers. The SFPSHFT mod is spelled out rather than taken from sfpi, whose
// SFPSHFT_MOD1_SHIFT_IMM is 0 - the encoding the hardware decodes as "shift by lreg_c".
constexpr std::uint32_t SFPSHFT_MOD1_IMM_SRC_C = 0b101;  // shift by imm12, source = lreg_c, logical
constexpr std::uint32_t SFPIADD_MOD1_IMM_NO_CC =
    sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE;                  // VD = VC + imm
constexpr std::uint32_t SFPIADD_MOD1_REG_NO_CC = sfpi::SFPIADD_MOD1_CC_NONE;  // VD = VC + VD
constexpr std::uint32_t SFPSETSGN_MOD1_SIGN_FROM_IMM = 1;                     // sign = imm12[0]

// bf16 2^-31: maps the [0, 2^31] integer grid onto [0, 1].
constexpr std::uint32_t FP16B_TWO_POW_NEG_31 = 0x3000;

// Per-lane salt s = h(lane_id + LANE_SALT_OFFSET), h(x) = y ^ (y << 6), y = x ^ (x << 14).
constexpr std::uint32_t LANE_SALT_OFFSET = 407;  // gives lane 0 a nonzero salt
constexpr std::uint32_t LANE_SALT_SHIFT_A = 14;
constexpr std::uint32_t LANE_SALT_SHIFT_B = 6;

// Bijective finalizer shifts (12-bit two's-complement imm; negative = right shift).
constexpr std::uint32_t MIX_SHIFT_R8 = (-8) & 0xFFF;
constexpr std::uint32_t MIX_SHIFT_R16 = (-16) & 0xFFF;
constexpr std::uint32_t MIX_SHIFT_L8 = 8;
constexpr std::uint32_t MIX_SHIFT_R14 = (-14) & 0xFFF;

// FP32 exponent field, used to fold the 2^-31 normalization into scale.
constexpr std::uint32_t FP32_EXP_SHIFT = 23;
constexpr std::uint32_t FP32_EXP_MASK = 0xFF;
constexpr std::uint32_t NORMALIZATION_EXPONENT = 31;

// All-ones is the XNOR LFSR lock-up state.
constexpr std::uint32_t PRNG_LFSR_LOCKUP_SEED = 0xFFFFFFFF;
constexpr std::uint32_t PRNG_LFSR_LOCKUP_REPAIR = 0xFFFFFFFE;
constexpr std::uint32_t PRNG_SEED_SETTLE_NOPS = 1024;  // no seeder busy flag to poll

constexpr std::uint32_t RAND_REPLAY_SLOT = 0;

// Length of the recorded row-pair body. It is one instruction longer when the 2^-31 normalization
// could not fold into scale and has to run per row as an SFPMULI.
constexpr std::uint32_t RAND_ROW_LEN_FOLDED = 16;
constexpr std::uint32_t RAND_ROW_LEN_PER_ROW_NORM = 17;
constexpr std::uint32_t RAND_REPLAY_DEPTH = 32;
static_assert(RAND_ROW_LEN_PER_ROW_NORM <= RAND_REPLAY_DEPTH, "the recorded body must fit the replay buffer");

/**
 * @brief Seed every lane's hardware PRNG and program the Dest advance the row body rides on.
 *
 * The seed reaches the PRNG through a RISC MMIO store to its config register. The seeder exposes no
 * busy flag, so PRNG_SEED_SETTLE_NOPS SFPNOPs stand in for polling one.
 *
 * @tparam APPROXIMATION_MODE: Unused; kept for parity with the Compute-API template tuple.
 * @param seed: 32-bit LFSR seed. All-ones is the XNOR lock-up state and is repaired before use.
 * @note Call this before @ref calculate_rand, and again before resuming rand after another SFPU op
 *       has run on this thread - that op's own init is what overwrites ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE /*unused*/>
inline void init_rand(std::uint32_t seed) {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);

    if (seed == PRNG_LFSR_LOCKUP_SEED) {
        seed = PRNG_LFSR_LOCKUP_REPAIR;
    }
    volatile std::uint32_t* cfg = (volatile std::uint32_t*)TENSIX_CFG_BASE;
    cfg[PRNG_SEED_Seed_Val_ADDR32] = seed;  // reseeds every lane PRNG
    for (std::uint32_t i = 0; i < PRNG_SEED_SETTLE_NOPS; i++) {
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    }
}

/**
 * @brief Derive this lane's salt into LREG3 from the read-only lane-ID constant register.
 *
 * Two xorshift rounds over lane_id + LANE_SALT_OFFSET. Both are bijective, so lanes stay distinct
 * from each other however the hardware seeder happened to seed them.
 *
 * @note Clobbers LREG4 and LREG5. Rebuild the salt once per face - another SFPU op may hold LREG3.
 */
inline void _rand_make_lane_salt_() {
    TTI_SFPIADD(LANE_SALT_OFFSET, LANE_ID_LREG, p_sfpu::LREG4, SFPIADD_MOD1_IMM_NO_CC);    // LREG4 = lane_id + 407
    TTI_SFPSHFT(LANE_SALT_SHIFT_A, p_sfpu::LREG4, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);  // LREG5 = LREG4 << 14
    TTI_SFPXOR(p_sfpu::LREG4, p_sfpu::LREG5);                                              // LREG5 ^= LREG4
    TTI_SFPSHFT(LANE_SALT_SHIFT_B, p_sfpu::LREG5, p_sfpu::LREG3, SFPSHFT_MOD1_IMM_SRC_C);  // LREG3 = LREG5 << 6
    TTI_SFPXOR(p_sfpu::LREG5, p_sfpu::LREG3);                                              // LREG3 ^= LREG5
}

/**
 * @brief Finish the bijective finalizer for this row into LREG4 and fetch the next row's PRNG word.
 *
 * Completes the x ^= x >> 8 round whose shift the caller already issued, multiplies by the odd
 * constant in MIX_MULTIPLIER_LREG, grafts the top bits SFPMUL24 does not produce back on, then two
 * more xorshift rounds. The SFPMOV that steps the PRNG fills the SFPMUL24 latency shadow.
 *
 * @note Entry: LREG0 = x, LREG5 = x >> 8. Exit: LREG4 = finalized word, LREG0 = next PRNG word.
 * @note Clobbers LREG5.
 */
inline void _rand_finish_mix_() {
    TTI_SFPXOR(p_sfpu::LREG5, p_sfpu::LREG0);                                          // x ^= x >> 8
    TTI_SFPSHFT(MIX_SHIFT_R16, p_sfpu::LREG0, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);  // LREG5 = x >> 16
    TTI_SFPXOR(p_sfpu::LREG0, p_sfpu::LREG5);                                          // LREG5 = x ^ (x >> 16)
    TTI_SFPMUL24(
        p_sfpu::LREG5, MIX_MULTIPLIER_LREG, p_sfpu::LREG4, sfpi::SFPMUL24_MOD1_LOWER);  // low 23 bits of x * 0x56594B
    TTI_SFPMOV(PRNG_RS_INDEX, p_sfpu::LREG0, SFPMOV_MOD1_FROM_RS);  // next row's PRNG word; fills the MUL24 shadow
    TTI_SFPSETMAN(0 /* imm12_math */, p_sfpu::LREG5, p_sfpu::LREG4, 0 /* instr_mod1 */);  // restore x[31:23]
    TTI_SFPSHFT(MIX_SHIFT_L8, p_sfpu::LREG4, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);      // LREG5 = y << 8
    TTI_SFPXOR(p_sfpu::LREG5, p_sfpu::LREG4);                                             // y ^= y << 8
    TTI_SFPSHFT(MIX_SHIFT_R14, p_sfpu::LREG4, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);     // LREG5 = y >> 14
    TTI_SFPXOR(p_sfpu::LREG5, p_sfpu::LREG4);                                             // y ^= y >> 14
}

/**
 * @brief One recorded row-pair body: finish the mix, convert to a uniform float, map it, store it.
 *
 * SFPCAST rounds the finalized word's low 31 bits to an FP32 in [0, 2^31], leaving bit 31 in the
 * sign, which SFPSETSGN then forces positive. The two instructions that start the next row's mix,
 * SFPIADD and SFPSHFT, are interleaved here so they fall in the latency shadows of the arithmetic
 * around them rather than costing cycles of their own.
 *
 * @tparam NORMALIZE_PER_ROW: Apply the 2^-31 normalization here rather than folded into scale.
 * @note Entry and exit state match, advanced by one row pair: LREG0 = x, LREG5 = x >> 8,
 *       LREG1 = scale, LREG2 = from, LREG3 = lane salt, and ADDR_MOD_6 has stepped Dest by 2 rows.
 */
template <bool NORMALIZE_PER_ROW>
inline void _calculate_rand_sfp_rows_() {
    _rand_finish_mix_();
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);  // fp32(y[30:0]), arbitrary sign
    TTI_SFPSETSGN(
        0 /* imm12_math: positive */, p_sfpu::LREG6, p_sfpu::LREG6, SFPSETSGN_MOD1_SIGN_FROM_IMM);  // [0, 2^31]
    if constexpr (NORMALIZE_PER_ROW) {
        TTI_SFPMULI(FP16B_TWO_POW_NEG_31, p_sfpu::LREG6, 0 /* instr_mod1 */);  // [0, 1]
    }
    TTI_SFPIADD(0 /* imm12_math */, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_REG_NO_CC);       // next x = salt + prng
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0 /* instr_mod1 */);  // u * scale + from
    TTI_SFPSHFT(
        MIX_SHIFT_R8, p_sfpu::LREG0, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);  // begin next mix; fills the MAD shadow
    TTI_SFPSTORE(
        p_sfpu::LREG6, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /* done */, 0 /* dest_reg_addr */);  // store, Dest += 2
}

/**
 * @brief Record the row-pair body into the replay buffer, then issue it once per row pair.
 *
 * Record-only, not record-with-execute: executing while loading hangs multi-tile runs on Quasar. The
 * caller primes row 0 instead, so every row pair - including the first - comes out of a REPLAY.
 *
 * @tparam NORMALIZE_PER_ROW: Selects the body, and with it the recorded length.
 * @tparam ITERATIONS: Row pairs per face.
 */
template <bool NORMALIZE_PER_ROW, int ITERATIONS>
inline void _rand_replay_rows_() {
    constexpr std::uint32_t LEN = NORMALIZE_PER_ROW ? RAND_ROW_LEN_PER_ROW_NORM : RAND_ROW_LEN_FOLDED;
    load_replay_buf<RAND_REPLAY_SLOT, LEN, false /* exec_while_loading */>(
        [] { _calculate_rand_sfp_rows_<NORMALIZE_PER_ROW>(); });
// Unrolled so scalar loop control does not open a gap between one replay and the next.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_REPLAY(
            RAND_REPLAY_SLOT, LEN, 0 /* last */, 0 /* set_mutex */, 0 /* execute_while_loading */, 0 /* load_mode */);
    }
}

/**
 * @brief Overwrite one face of Dest with values drawn uniformly from [from, from + scale].
 *
 * The face's existing contents are ignored. Each row pair takes one step of the per-lane hardware
 * PRNG, salts it so lanes decorrelate, runs it through a bijective 32-bit finalizer, and maps the
 * low 31 bits onto the interval. Folding the 2^-31 normalization into scale's exponent - which only
 * fails when that exponent has no room, or scale is Inf/NaN - drops one SFPMULI from every row pair.
 *
 * @tparam APPROXIMATION_MODE: Unused; there is no approximate variant.
 * @tparam ITERATIONS: Row pairs per face.
 * @param from: Lower bound of the interval, as FP32 bits.
 * @param scale: Width of the interval, as FP32 bits. Non-negative; 0 gives a face of constant from.
 * @note Call @ref init_rand before this - it seeds the PRNG and programs the ADDR_MOD_6 that walks
 *       Dest down the face.
 * @note Clobbers LREG0-LREG6 and re-records replay slot 0 on every call, so any other op recording
 *       into that slot must re-record after rand.
 */
template <bool APPROXIMATION_MODE /*unused*/, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_rand(const std::uint32_t from, std::uint32_t scale) {
    // Fold 2^-31 into scale unless that would underflow its exponent or it is Inf/NaN.
    const std::uint32_t exp = (scale >> FP32_EXP_SHIFT) & FP32_EXP_MASK;
    const bool normalize_per_row = (exp <= NORMALIZATION_EXPONENT) || (exp == FP32_EXP_MASK);
    if (!normalize_per_row) {
        scale -= NORMALIZATION_EXPONENT << FP32_EXP_SHIFT;
    }

    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, from >> 16);

    _rand_make_lane_salt_();

    // Prime row 0.
    TTI_SFPMOV(PRNG_RS_INDEX, p_sfpu::LREG0, SFPMOV_MOD1_FROM_RS);  // LREG0 = prng; steps the PRNG
    TTI_SFPIADD(0 /* imm12_math */, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_REG_NO_CC);  // x = salt + prng
    TTI_SFPSHFT(MIX_SHIFT_R8, p_sfpu::LREG0, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);        // LREG5 = x >> 8

    if (normalize_per_row) {
        _rand_replay_rows_<true /* NORMALIZE_PER_ROW */, ITERATIONS>();
    } else {
        _rand_replay_rows_<false /* NORMALIZE_PER_ROW */, ITERATIONS>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
