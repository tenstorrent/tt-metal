// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// SFPSETCC imm12 bit 11 = 1: interpret src_c as FP32/SMAG32 instead of two's-complement INT32.
constexpr std::uint32_t SFPSETCC_IMM_FP32_TEST = 0x800;

// SFPMAD/SFPMUL instr_mod1 bit 0 = 1: invert the sign of src_a, giving VD = VC - VA*VB.
constexpr std::uint32_t SFPMAD_MOD_NEG_SRCA = 0x1;

// SFPSHFT instr_mod1: bit 0 = shift amount from imm12, bit 2 = data from lreg_c.
// sfpi::SFPSHFT_MOD1_SHIFT_IMM is 0, which contradicts assembly.yaml and working Quasar code.
constexpr std::uint32_t SFPSHFT_MOD_IMM_SRC_C = 0x5;

// Quiet NaN 0x7FC00000, split into the SFPLOADI UPPER/LOWER halves.
constexpr std::uint32_t FP32_QUIET_NAN_HI = 0x7FC0;
constexpr std::uint32_t FP32_QUIET_NAN_LO = 0x0000;

// Everything but the fp32 sign bit, so a masked compare against 0 catches both +0.0 and -0.0.
constexpr std::uint32_t FP32_MAGNITUDE_MASK = 0x7FFFFFFF;

// 2.0f as a bf16 immediate (upper 16 bits of 0x40000000), for the Newton-Raphson reciprocal step.
constexpr std::uint32_t BF16_TWO = 0x4000;

// SFPMUL24 splits a 32-bit operand into two 23-bit halves; this is the split point. A right shift
// is a negative amount in the 12-bit immediate field, hence the (-MUL24_SPLIT_SHIFT) & 0xFFF below.
constexpr std::uint32_t MUL24_SPLIT_SHIFT = 23;

// One recorded pass of the float path: the 19 instructions of _calculate_remainder_sfp_rows_
// plus the trailing INCRWC. Fits the 32-entry replay buffer at slot 0.
constexpr std::uint32_t REMAINDER_REPLAY_SLOT = 0;
constexpr std::uint32_t REMAINDER_REPLAY_LEN = 20;

/**
 * @brief One SFPU pass (SFP_ROWS = 2 Dest rows) of the float scalar remainder.
 *
 * Expects the preamble registers set up by @ref calculate_remainder: LREG5 = |b|,
 * LREG6 = |1/b|, LREG7 = copysign(1.0, b) (or a quiet NaN when b == 0).
 *
 * @note Keep this body branch-free and 19 instructions long: @ref calculate_remainder records it
 *       into a replay buffer sized by REMAINDER_REPLAY_LEN (19 + the trailing INCRWC).
 */
inline void _calculate_remainder_sfp_rows_() {
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);  // a
    TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPABS_MOD1_FLOAT);                                // v = |a|
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, 0 /* mod1 */);  // keep a for the sign test
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG2, 0 /* mod1 */);  // q_f = v * |1/b|
    TTI_SFPXOR(p_sfpu::LREG7, p_sfpu::LREG3);  // MSB of LREG3 now means "signs of a and b differ"; fills the MUL shadow

    // Round the quotient to nearest, not toward zero: the residual then lands in (-s, s) and one
    // conditional +s below turns it into floor semantics. Undefined for |a/b| >= 2^31, where the
    // fp32 <-> int32 round trip stops being exact.
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG4, sfpi::SFPCAST_MOD1_FP32_TO_SM32_RNE);
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);

    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LREG1, SFPMAD_MOD_NEG_SRCA);  // mm = v - q*s
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);                    // 2-cycle MAD shadow

    TTI_SFPSETCC(SFPSETCC_IMM_FP32_TEST, p_sfpu::LREG1, sfpi::SFPSETCC_MOD1_LREG_LT0);        // quotient rounded up
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0 /* mod1 */);  // mm += s -> m in [0, s)
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);

    // Successive SFPSETCC calls AND into CC: fire only where the signs differ and m is non-zero.
    TTI_SFPSETCC(SFPSETCC_IMM_FP32_TEST, p_sfpu::LREG3, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(SFPSETCC_IMM_FP32_TEST, p_sfpu::LREG1, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, SFPMAD_MOD_NEG_SRCA);  // m = s - m
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);

    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG1, 0 /* mod1 */);  // take the sign of b
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);                // 2-cycle MAD shadow
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
}

/**
 * @brief Element-wise floor-based remainder against a float scalar divisor: out = a - floor(a/b) * b.
 *
 * The result carries the sign of b and lies in [0, |b|) for b > 0 and in (-|b|, 0] for b < 0
 * (torch.remainder semantics); b == 0 yields NaN. Computed as m = |a| mod |b| with a single
 * conditional correction, then the sign quadrant fix-up `signs differ && m != 0 -> |b| - m`.
 *
 * @tparam APPROXIMATION_MODE: accepted for ABI parity but ignored (there is no estimator on the data path).
 * @tparam ITERATIONS: number of SFPU passes per face.
 * @param value: divisor b as an fp32 bit pattern.
 * @param recip: 1/b as an fp32 bit pattern.
 * @note Requires @ref _llk_math_eltwise_sfpu_init_ first: it programs ADDR_MOD_7 and re-establishes
 *       LCONST_0 = 0.0 / LCONST_1 = 1.0, both of which the MAD chain reads.
 * @note Overwrites math-thread replay-buffer entries REMAINDER_REPLAY_SLOT +: REMAINDER_REPLAY_LEN
 *       on every call; do not rely on a recording made across a call to this function.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_remainder(const std::uint32_t value, const std::uint32_t recip) {
    TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);  // b low half
    TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);     // b high half -> LREG4 = b
    TT_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, recip & 0xFFFF);  // 1/b low half
    TT_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, recip >> 16);     // 1/b high half -> LREG6 = 1/b

    TTI_SFPABS(p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPABS_MOD1_FLOAT);  // s = |b|
    TTI_SFPABS(p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPABS_MOD1_FLOAT);  // |1/b|
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG7, 0 /* mod1 */);             // sign donor; SFPSETSGN overwrites it

    if ((value & FP32_MAGNITUDE_MASK) == 0) {
        // b == 0: multiplying by the carrier turns every datum into NaN.
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, FP32_QUIET_NAN_HI);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, FP32_QUIET_NAN_LO);
    } else {
        // LREG7 = copysign(1.0, b): magnitude from src_c, sign from src_b (= lreg_dest).
        TTI_SFPSETSGN(0 /* imm12 */, p_sfpu::LCONST_1, p_sfpu::LREG7, 0 /* mod1: sign from src_b */);
    }

    // Every pass issues the same 20 instructions — the Dest walk lives in ADDR_MOD_7 and in the
    // recorded INCRWC, whose counter state persists across replays — so record it once. Recording
    // with execute_while_loading = 0 costs one cycle per instruction and emits nothing downstream.
    load_replay_buf<REMAINDER_REPLAY_SLOT, REMAINDER_REPLAY_LEN, false /* exec_while_loading */>([] {
        _calculate_remainder_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // advance 2 Dest rows
    });

    for (int d = 0; d < ITERATIONS; d++) {
        TTI_REPLAY(
            REMAINDER_REPLAY_SLOT,
            REMAINDER_REPLAY_LEN,
            0 /* last */,
            0 /* set_mutex */,
            0 /* execute_while_loading: ignored when load_mode = 0 */,
            0 /* load_mode: issue the recorded instructions */);
    }
}

/**
 * @brief Element-wise unsigned 32-bit remainder against an unsigned scalar divisor: out = a mod scalar.
 *
 * Three divisor cases, all folded at compile time when the caller passes a literal:
 *   power of two   -> a & (scalar - 1)
 *   scalar >= 2^31 -> at most one conditional subtract
 *   otherwise      -> q = rint(a * (1/scalar)) with an exact low-32 q*scalar product, then bounded fix-ups
 *
 * @tparam APPROXIMATION_MODE: accepted for ABI parity but ignored.
 * @tparam ITERATIONS: number of SFPU passes per face.
 * @param scalar: unsigned integer divisor.
 * @note Dest must hold two's-complement Int32 (the UNPACR_DEST path) and dividends must be
 *       non-negative: SFPCAST reads int32 as sign-magnitude, and the two encodings only
 *       coincide for non-negative values. Exact for dividends <= 2^24.
 * @note Requires @ref _llk_math_eltwise_sfpu_init_ first (ADDR_MOD_7 plus the LCONST registers).
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_remainder_uint32_scalar(const std::uint32_t scalar) {
    if ((scalar & (scalar - 1u)) == 0u) {
        // Power of two (scalar == 0 is rejected by the caller, and would leave every datum unchanged).
        TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, (scalar - 1u) & 0xFFFF);  // mask low half
        TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, (scalar - 1u) >> 16);     // mask high half

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
            TTI_SFPAND(p_sfpu::LREG5, p_sfpu::LREG0);  // a &= scalar - 1
            TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
            ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
        }
    } else if (scalar >= 0x80000000u) {
        // a < 2^32 <= 2*scalar, so at most one subtract is needed. Unreachable from a Quasar test:
        // UInt32 is not a Quasar format and Int32 cannot carry a bit-31-set dividend. Kept for API parity.
        TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, scalar & 0xFFFF);  // divisor low half
        TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, scalar >> 16);     // divisor high half

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
            TTI_SFPMOV(p_sfpu::LREG5, p_sfpu::LREG4, 0 /* mod1 */);  // SFPIADD overwrites its dest
            TTI_SFPIADD(
                0 /* imm12 */,
                p_sfpu::LREG0,
                p_sfpu::LREG4,
                sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);  // a - scalar, CC = result >= 0
            // Only a dividend with bit 31 set can be >=u a bit-31-set divisor; without this guard the
            // signed subtract above wraps and looks non-negative.
            TTI_SFPSETCC(0 /* imm12: INT32 test */, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG0, 0 /* mod1 */);  // a -= scalar
            TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);
            TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
            ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
        }
    } else {
        TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, scalar & 0xFFFF);  // divisor low half
        TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, scalar >> 16);     // divisor high half

        // 1/scalar in fp32: LUT seed plus two Newton-Raphson steps inv = inv * (2 - s_f * inv).
        // Each MAD/MUL result is consumed by the next instruction, so every one carries a NOP shadow.
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG3, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);  // s_f
        TTI_SFPNONLINEAR(p_sfpu::LREG3, p_sfpu::LREG6, p_sfpnonlinear::RECIP_MODE);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, BF16_TWO);  // 2.0f
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG6, p_sfpu::LREG2, p_sfpu::LREG1, SFPMAD_MOD_NEG_SRCA);
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
        TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0 /* mod1 */);
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG6, p_sfpu::LREG2, p_sfpu::LREG1, SFPMAD_MOD_NEG_SRCA);
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
        TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0 /* mod1 */);
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);  // exact for a <= 2^24
            TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG2, 0 /* mod1 */);  // q_f
            TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);       // 2-cycle MAD shadow
            TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPCAST_MOD1_FP32_TO_SM32_RNE);  // q = rint(q_f)

            // Exact low 32 bits of q * scalar from 23-bit SFPMUL24 partial products.
            TTI_SFPSHFT((-MUL24_SPLIT_SHIFT) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG1, SFPSHFT_MOD_IMM_SRC_C);
            TTI_SFPSHFT((-MUL24_SPLIT_SHIFT) & 0xFFF, p_sfpu::LREG5, p_sfpu::LREG3, SFPSHFT_MOD_IMM_SRC_C);
            TTI_SFPMUL24(p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG4, sfpi::SFPMUL24_MOD1_LOWER);  // q_lo * scalar_lo
            TTI_SFPMUL24(p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG7, sfpi::SFPMUL24_MOD1_UPPER);  // q_lo * scalar_lo
            TTI_SFPMUL24(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG1, sfpi::SFPMUL24_MOD1_LOWER);  // q_hi * scalar_lo
            TTI_SFPMUL24(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPMUL24_MOD1_LOWER);  // q_lo * scalar_hi
            TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
            TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
            TTI_SFPSHFT(MUL24_SPLIT_SHIFT, p_sfpu::LREG7, p_sfpu::LREG7, SFPSHFT_MOD_IMM_SRC_C);
            TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);

            TTI_SFPIADD(
                0 /* imm12 */,
                p_sfpu::LREG0,
                p_sfpu::LREG4,
                sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);  // r = a - q*scalar

            // rint can miss by one in either direction, so two +scalar steps cover r < 0 ...
            TTI_SFPSETCC(0 /* imm12: INT32 test */, p_sfpu::LREG4, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG5, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_LREG_DST);
            TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);
            TTI_SFPSETCC(0 /* imm12: INT32 test */, p_sfpu::LREG4, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG5, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_LREG_DST);
            TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);

            // ... and one -scalar step covers r >= scalar.
            TTI_SFPMOV(p_sfpu::LREG5, p_sfpu::LREG1, 0 /* mod1 */);  // SFPIADD overwrites its dest
            TTI_SFPIADD(
                0 /* imm12 */,
                p_sfpu::LREG4,
                p_sfpu::LREG1,
                sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG4, 0 /* mod1 */);  // r -= scalar
            TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);

            TTI_SFPSTORE(p_sfpu::LREG4, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
            ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
