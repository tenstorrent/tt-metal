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

// Register-view indices read by the row body. Quasar source: the sfpi toolchain's sfpi_constants.h
// defines SFPCONFIG_SRC_RAND under `__riscv_xtttensixbh || __riscv_xtttensixqsr` (it is what
// sfpi::rand() reads on Quasar), and CREG_IDX_0P837300003 / CREG_IDX_TILEID for every arch including
// Quasar (sfpi::vConst0p8373 / sfpi::vConstTileId). The SFPCONFIG table in
// tt_llk_quasar/instructions/assembly.yaml lists 0x9 as a LUT constant: that is the SFPCONFIG *write*
// map, while an SFPMOV mod 0x8 *read* of index 9 returns the PRNG. test_rand_quasar's distinct-row
// and distinct-count checks on the Float32-output variants fail if either reading were wrong.
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

// SFPLOADI writes an FP32 as two 16-bit halves.
constexpr std::uint32_t FP32_HALF_BITS = 16;
constexpr std::uint32_t FP32_HALF_MASK = 0xFFFF;

// All-ones is the XNOR LFSR lock-up state.
constexpr std::uint32_t PRNG_LFSR_LOCKUP_SEED = 0xFFFFFFFF;
constexpr std::uint32_t PRNG_LFSR_LOCKUP_REPAIR = 0xFFFFFFFE;
constexpr std::uint32_t PRNG_SEED_SETTLE_NOPS = 1024;  // no seeder busy flag to poll

// The row-pair body is recorded once, by init_rand, as a head and a tail. The folded body replays
// both back to back; the per-row-normalize body issues its SFPMULI between them.
//   head = finish_mix (SFPXOR + 2 x shift-xor + SFPMUL24 + SFPMOV + SFPSETMAN + 2 x shift-xor)
//          + to_unit (SFPCAST + SFPSETSGN)
//   tail = SFPIADD + SFPMAD + SFPSHFT + SFPSTORE
constexpr std::uint32_t RAND_SHIFT_XOR_LEN = 2;  // SFPSHFT + SFPXOR
constexpr std::uint32_t RAND_FINISH_MIX_LEN = 1 + RAND_SHIFT_XOR_LEN + 3 + 2 * RAND_SHIFT_XOR_LEN;
constexpr std::uint32_t RAND_TO_UNIT_LEN = 2;
constexpr std::uint32_t RAND_HEAD_LEN = RAND_FINISH_MIX_LEN + RAND_TO_UNIT_LEN;
constexpr std::uint32_t RAND_TAIL_LEN = 4;
constexpr std::uint32_t RAND_REPLAY_SLOT = 0;
constexpr std::uint32_t RAND_TAIL_SLOT = RAND_REPLAY_SLOT + RAND_HEAD_LEN;
constexpr std::uint32_t RAND_BODY_LEN = RAND_HEAD_LEN + RAND_TAIL_LEN;
constexpr std::uint32_t RAND_REPLAY_DEPTH = 32;
static_assert(RAND_REPLAY_SLOT + RAND_BODY_LEN <= RAND_REPLAY_DEPTH, "the recorded body must fit the replay buffer");

/**
 * @brief dst = (src << SHIFT) ^ src. A negative (12-bit two's-complement) SHIFT shifts right.
 */
template <std::uint32_t SHIFT, std::uint32_t SRC, std::uint32_t DST>
inline void _rand_shift_xor_() {
    TTI_SFPSHFT(SHIFT, SRC, DST, SFPSHFT_MOD1_IMM_SRC_C);  // dst = src << SHIFT
    TTI_SFPXOR(SRC, DST);                                  // dst ^= src
}

/**
 * @brief x ^= x << SHIFT in place, through TMP. A negative (12-bit two's-complement) SHIFT shifts right.
 */
template <std::uint32_t SHIFT, std::uint32_t X, std::uint32_t TMP>
inline void _rand_xor_shift_in_place_() {
    TTI_SFPSHFT(SHIFT, X, TMP, SFPSHFT_MOD1_IMM_SRC_C);  // tmp = x << SHIFT
    TTI_SFPXOR(TMP, X);                                  // x ^= tmp
}

/**
 * @brief Load a 32-bit pattern into LREG as two 16-bit SFPLOADI halves.
 */
template <std::uint32_t LREG>
inline void _rand_load_fp32_(const std::uint32_t bits) {
    TT_SFPLOADI(LREG, sfpi::SFPLOADI_MOD0_LOWER, bits & FP32_HALF_MASK /* imm16 */);
    TT_SFPLOADI(LREG, sfpi::SFPLOADI_MOD0_UPPER, bits >> FP32_HALF_BITS /* imm16 */);
}

/**
 * @brief Finish the bijective finalizer for this row into LREG4 and fetch the next row's PRNG word.
 *
 * Completes the x ^= x >> 8 round whose shift the caller already issued, multiplies by the odd
 * constant in MIX_MULTIPLIER_LREG, grafts the top bits SFPMUL24 does not produce back on, then two
 * more xorshift rounds. The SFPMOV that steps the PRNG fills the SFPMUL24 latency shadow.
 *
 * @note Entry: LREG0 = x, LREG5 = x >> 8. Exit: LREG4 = finalized word, LREG0 = next PRNG word.
 * @note Clobbers LREG5. Issues RAND_FINISH_MIX_LEN instructions.
 */
inline void _rand_finish_mix_() {
    TTI_SFPXOR(p_sfpu::LREG5, p_sfpu::LREG0);                         // x ^= x >> 8 (shift issued by caller)
    _rand_shift_xor_<MIX_SHIFT_R16, p_sfpu::LREG0, p_sfpu::LREG5>();  // LREG5 = x ^ (x >> 16)
    TTI_SFPMUL24(
        p_sfpu::LREG5, MIX_MULTIPLIER_LREG, p_sfpu::LREG4, sfpi::SFPMUL24_MOD1_LOWER);  // low 23 bits of x * 0x56594B
    TTI_SFPMOV(PRNG_RS_INDEX, p_sfpu::LREG0, SFPMOV_MOD1_FROM_RS);  // next row's PRNG word; fills the MUL24 shadow
    TTI_SFPSETMAN(0 /* imm12_math */, p_sfpu::LREG5, p_sfpu::LREG4, 0 /* instr_mod1 */);  // restore x[31:23]
    _rand_xor_shift_in_place_<MIX_SHIFT_L8, p_sfpu::LREG4, p_sfpu::LREG5>();              // y ^= y << 8
    _rand_xor_shift_in_place_<MIX_SHIFT_R14, p_sfpu::LREG4, p_sfpu::LREG5>();             // y ^= y >> 14
}

/**
 * @brief Recorded head of the row-pair body: finish the mix and convert it to a float in [0, 2^31].
 *
 * SFPCAST rounds the finalized word's low 31 bits to an FP32 in [0, 2^31], leaving bit 31 in the
 * sign, which SFPSETSGN then forces positive.
 *
 * @note Entry: LREG0 = x, LREG5 = x >> 8. Exit: LREG6 = draw in [0, 2^31], LREG0 = next PRNG word.
 */
inline void _rand_row_head_() {
    _rand_finish_mix_();
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);  // fp32(y[30:0]), arbitrary sign
    TTI_SFPSETSGN(
        0 /* imm12_math: positive */, p_sfpu::LREG6, p_sfpu::LREG6, SFPSETSGN_MOD1_SIGN_FROM_IMM);  // [0, 2^31]
}

/**
 * @brief Recorded tail of the row-pair body: map the draw onto the interval, store it, start the next mix.
 *
 * The two instructions that start the next row's mix, SFPIADD and SFPSHFT, are interleaved so they
 * fall in the latency shadows of the arithmetic around them rather than costing cycles of their own.
 *
 * @note Exit state matches the head's entry state, advanced by one row pair: LREG0 = x,
 *       LREG5 = x >> 8, LREG1 = scale, LREG2 = from, LREG3 = lane salt, and ADDR_MOD_6 has stepped
 *       Dest by 2 rows.
 */
inline void _rand_row_tail_() {
    TTI_SFPIADD(0 /* imm12_math */, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_REG_NO_CC);       // next x = salt + prng
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0 /* instr_mod1 */);  // u * scale + from
    TTI_SFPSHFT(
        MIX_SHIFT_R8, p_sfpu::LREG0, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);  // begin next mix; fills the MAD shadow
    TTI_SFPSTORE(
        p_sfpu::LREG6, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /* done */, 0 /* dest_reg_addr */);  // store, Dest += 2
}

/**
 * @brief Seed every lane's hardware PRNG and record the row-pair body into the replay buffer.
 *
 * The seed reaches the PRNG through a RISC MMIO store to its config register. The seeder exposes no
 * busy flag, so PRNG_SEED_SETTLE_NOPS SFPNOPs stand in for polling one. The body is recorded only,
 * not recorded-and-executed (TEN-4690); it is all immediates, so recording it once here leaves each
 * face costing only its replays.
 *
 * @tparam APPROXIMATION_MODE: Unused; kept for parity with the Compute-API template tuple.
 * @param seed: 32-bit LFSR seed. All-ones is the XNOR lock-up state and is replaced by all-ones-but-one.
 * @note Calling this again restarts every lane's stream from @p seed, so call it once per stream,
 *       not once per tile.
 * @note Records replay slots [RAND_REPLAY_SLOT, RAND_REPLAY_SLOT + RAND_BODY_LEN). Call it again,
 *       with the seed the stream should continue from, after any op that records into those slots.
 */
template <bool APPROXIMATION_MODE /*unused*/>
inline void init_rand(std::uint32_t seed) {
    if (seed == PRNG_LFSR_LOCKUP_SEED) {
        seed = PRNG_LFSR_LOCKUP_REPAIR;
    }
    volatile std::uint32_t* cfg = (volatile std::uint32_t*)TENSIX_CFG_BASE;
    cfg[PRNG_SEED_Seed_Val_ADDR32] = seed;  // reseeds every lane PRNG
    for (std::uint32_t i = 0; i < PRNG_SEED_SETTLE_NOPS; i++) {
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    }

    load_replay_buf<RAND_REPLAY_SLOT, RAND_BODY_LEN, false /* exec_while_loading */>([] {
        _rand_row_head_();
        _rand_row_tail_();
    });
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
    TTI_SFPIADD(LANE_SALT_OFFSET, LANE_ID_LREG, p_sfpu::LREG4, SFPIADD_MOD1_IMM_NO_CC);  // LREG4 = lane_id + 407
    _rand_shift_xor_<LANE_SALT_SHIFT_A, p_sfpu::LREG4, p_sfpu::LREG5>();                 // LREG5 = y
    _rand_shift_xor_<LANE_SALT_SHIFT_B, p_sfpu::LREG5, p_sfpu::LREG3>();                 // LREG3 = salt
}

/**
 * @brief Overwrite one face of Dest with values drawn uniformly from [from, from + scale].
 *
 * The face's existing contents are ignored. Each row pair (2 face rows x 16 columns, the 32 SFPU
 * lanes) takes one step of the per-lane hardware PRNG, salts it so lanes decorrelate, runs it
 * through a bijective 32-bit finalizer, and maps the low 31 bits onto the interval. Folding the
 * 2^-31 normalization into scale's exponent - which only fails when that exponent has no room, or
 * scale is Inf/NaN - drops one SFPMULI from every row pair.
 *
 * @tparam APPROXIMATION_MODE: Unused; there is no approximate variant.
 * @tparam ITERATIONS: Row pairs per face.
 * @param from: Lower bound of the interval, as FP32 bits.
 * @param scale: Width of the interval, as FP32 bits. Non-negative; 0 gives a face of constant from.
 * @note Call @ref init_rand before this - it seeds the PRNG and records the body this replays.
 * @note Programs ADDR_MOD_6 (Dest += 2) on every call, so ops that reprogram it in between need no
 *       re-init. Clobbers LREG0-LREG6.
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

    // Prime row 0.
    TTI_SFPMOV(PRNG_RS_INDEX, p_sfpu::LREG0, SFPMOV_MOD1_FROM_RS);  // LREG0 = prng; steps the PRNG
    TTI_SFPIADD(0 /* imm12_math */, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_REG_NO_CC);  // x = salt + prng
    TTI_SFPSHFT(MIX_SHIFT_R8, p_sfpu::LREG0, p_sfpu::LREG5, SFPSHFT_MOD1_IMM_SRC_C);        // LREG5 = x >> 8

    // Unrolled so scalar loop control does not open a gap between one replay and the next.
    if (normalize_per_row) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_REPLAY(RAND_REPLAY_SLOT, RAND_HEAD_LEN, 0 /* last */, 0 /* set_mutex */, 0 /* exec */, 0 /* load */);
            TTI_SFPMULI(FP16B_TWO_POW_NEG_31, p_sfpu::LREG6, 0 /* instr_mod1 */);  // [0, 1]
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
