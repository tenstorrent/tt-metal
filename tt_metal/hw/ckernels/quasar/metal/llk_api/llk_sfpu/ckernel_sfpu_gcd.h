// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"

namespace ckernel {
namespace sfpu {

constexpr std::uint32_t GCD_ABS_MOD_INT32 = 0x0;  // SFPABS two's-complement abs
constexpr std::uint32_t GCD_MOV_MOD_COPY = 0x0;   // SFPMOV plain copy
constexpr std::uint32_t GCD_IADD_MOD_ADD_KEEP_CC =
    p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;   // SFPIADD dest = lreg_c + dest, CC untouched
constexpr std::uint32_t GCD_IADD_SUB_DEST = 0x2;  // SFPIADD mod bit 1: subtract dest instead of adding it
constexpr std::uint32_t GCD_IADD_MOD_SUB_KEEP_CC =
    GCD_IADD_SUB_DEST | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;  // SFPIADD dest = lreg_c - dest, CC untouched
constexpr std::uint32_t GCD_LZ_MOD_COUNT = 0x0;                      // SFPLZ count leading zeros only
constexpr std::uint32_t GCD_LZ_MOD_CC_NE0 = 0x2;                     // SFPLZ count + CC.Res = (src != 0)
constexpr std::uint32_t GCD_SHFT_MOD_INPLACE_VAR_LOGICAL = 0x0;  // SFPSHFT dest <<= lreg_c (negative -> right), logical
constexpr std::uint32_t GCD_SETCC_IMM12_INT32 = 0x0;  // p_sfpu::cc::FP32_SM32_EN clear: SFPSETCC tests src as INT32
constexpr std::uint32_t GCD_SETCC_MOD_EQ0 = 0x6;      // SFPSETCC CC.Res = (src == 0)
// SFPSWAP imm12 bit 0 selects the compare domain, 0 = two's-complement int32, 1 = fp32; bits 11:1
// are reserved (tt_llk_quasar/instructions/assembly.yaml, SFPSWAP imm12_math). The int32 compare
// matters: operands above 0x7F800000 are NaN bit patterns in fp32, and the Quasar emulator test
// plants such pairs.
constexpr std::uint32_t GCD_SWAP_IMM12_INT32 = 0x0;
constexpr std::uint32_t GCD_ENCC_IMM12_ENABLE = 0x1;      // SFPENCC imm12[0] = 1: enable CC
constexpr std::uint32_t GCD_ENCC_MOD_SET_EN = 0x2;        // SFPENCC CC.En <- imm12[0], CC.Res = 1
constexpr std::uint32_t GCD_ENCC_MOD_RESET_RESULT = 0x0;  // SFPENCC CC.Res = 1, CC.En kept

// Register roles. LREG1 carries b through every iteration and holds the result; LREG0 and LREG2
// ping-pong between "-a" and "the working |a|" (see _emit_gcd_step_).
constexpr std::uint32_t GCD_LREG_A = p_sfpu::LREG0;
constexpr std::uint32_t GCD_LREG_B = p_sfpu::LREG1;
constexpr std::uint32_t GCD_LREG_TMP = p_sfpu::LREG2;
constexpr std::uint32_t GCD_LREG_KBIAS = p_sfpu::LREG3;  // k - 31, where 2^k = lsb(a | b)

// Operand magnitude bound in bits: every two's-complement int32 except -2^31, which Quasar
// SFPABS saturates to 2^31 - 1.
constexpr std::uint32_t GCD_MAX_INPUT_BITS = 31;

constexpr std::uint32_t GCD_REPLAY_DEPTH = 32;
constexpr std::uint32_t GCD_STEP_INSTRS = 8;  // instructions emitted by _emit_gcd_step_
constexpr std::uint32_t GCD_REPLAY_START = 0;
constexpr std::uint32_t GCD_REPLAY_LEN = 2 * GCD_STEP_INSTRS;  // one ping-pong pair of Stein steps
static_assert(GCD_REPLAY_LEN < GCD_REPLAY_DEPTH, "replay length must fit the log2(depth) len field");

constexpr std::uint32_t GCD_PROLOGUE_START = GCD_REPLAY_START + GCD_REPLAY_LEN;
constexpr std::uint32_t GCD_PROLOGUE_LEN = 15;  // instructions emitted by _emit_gcd_prologue_
static_assert(GCD_PROLOGUE_START + GCD_PROLOGUE_LEN <= GCD_REPLAY_DEPTH, "prologue must fit after the recorded pair");

/**
 * @brief Emit the per-row prologue that sets up the Stein iteration invariant.
 *
 * Takes a in GCD_LREG_A and b in GCD_LREG_B (two's complement) and leaves -a in GCD_LREG_A,
 * b in GCD_LREG_B with exactly k = ctz(a | b) trailing zeros, and k - 31 in GCD_LREG_KBIAS.
 *
 * @note Clobbers GCD_LREG_TMP and leaves CC.Res = 1 (all lanes active).
 */
inline void _emit_gcd_prologue_() {
    // Both operands are made non-negative first so every later bit trick sees a magnitude.
    TTI_SFPABS(GCD_LREG_A, GCD_LREG_A, GCD_ABS_MOD_INT32);  // a = |a|
    TTI_SFPABS(GCD_LREG_B, GCD_LREG_B, GCD_ABS_MOD_INT32);  // b = |b|
    TTI_SFPMOV(GCD_LREG_A, GCD_LREG_TMP, GCD_MOV_MOD_COPY);
    TTI_SFPOR(GCD_LREG_B, GCD_LREG_TMP);  // TMP = a | b
    TTI_SFPMOV(GCD_LREG_TMP, GCD_LREG_KBIAS, GCD_MOV_MOD_COPY);
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, GCD_LREG_KBIAS, GCD_IADD_MOD_SUB_KEEP_CC);  // KBIAS = -(a | b)
    TTI_SFPAND(GCD_LREG_TMP, GCD_LREG_KBIAS);                                                // KBIAS = lsb(a | b) = 2^k
    TTI_SFPMOV(GCD_LREG_B, GCD_LREG_TMP, GCD_MOV_MOD_COPY);
    TTI_SFPAND(GCD_LREG_KBIAS, GCD_LREG_TMP);  // TMP = b & 2^k
    // Swap only the lanes where b has more than k trailing zeros, so afterwards b has exactly k
    // and every iteration can strip a down to k instead of tracking two counts.
    TTI_SFPSETCC(GCD_SETCC_IMM12_INT32, GCD_LREG_TMP, GCD_SETCC_MOD_EQ0);
    TTI_SFPSWAP(0 /* imm12 */, GCD_LREG_A, GCD_LREG_B, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPENCC(0 /* imm12 */, GCD_ENCC_MOD_RESET_RESULT);                                   // all lanes active again
    TTI_SFPLZ(GCD_LREG_KBIAS, GCD_LREG_KBIAS, GCD_LZ_MOD_COUNT);                             // KBIAS = 31 - k
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, GCD_LREG_KBIAS, GCD_IADD_MOD_SUB_KEEP_CC);  // KBIAS = k - 31
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, GCD_LREG_A, GCD_IADD_MOD_SUB_KEEP_CC);      // A = -a
}

/**
 * @brief Emit one Stein (binary GCD) iteration: strip a's excess trailing zeros, then
 *        replace the pair with (min, max - min).
 *
 * Entry: @p NEG_A = -a, GCD_LREG_B = b with exactly k trailing zeros, GCD_LREG_KBIAS = k - 31.
 * Exit: @p WORK = -a', GCD_LREG_B = min(a, b). The two scratch registers have exchanged roles,
 * so two steps with @p NEG_A and @p WORK swapped restore the entry assignment.
 *
 * @tparam NEG_A: LREG holding -a; overwritten with lsb(a) and then the shift amount
 * @tparam WORK: LREG used as the working |a|; holds -a' on exit
 * @note Quasar SFPSHFT can only shift its own destination register, so the working value and the
 *       shift amount cannot stay in fixed registers across iterations - hence the ping-pong.
 * @note Lanes that reach a == 0 are retired by SFPLZ's CC update and stay retired until the
 *       caller's SFPENCC; their GCD_LREG_B already holds the answer.
 */
template <std::uint32_t NEG_A, std::uint32_t WORK>
inline void _emit_gcd_step_() {
    static_assert(NEG_A != WORK, "the Stein step needs two distinct scratch registers");

    TTI_SFPABS(NEG_A, WORK, GCD_ABS_MOD_INT32);  // WORK = a
    TTI_SFPAND(WORK, NEG_A);                     // NEG_A = (-a) & a = lsb(a)
    TTI_SFPLZ(NEG_A, NEG_A, GCD_LZ_MOD_CC_NE0);  // NEG_A = 31 - ctz(a); retire a == 0 lanes
    TTI_SFPIADD(0 /* imm12 */, GCD_LREG_KBIAS, NEG_A, GCD_IADD_MOD_ADD_KEEP_CC);   // NEG_A = k - ctz(a)
    TTI_SFPSHFT(0 /* imm12 */, NEG_A, WORK, GCD_SHFT_MOD_INPLACE_VAR_LOGICAL);     // strip extra trailing zeros
    TTI_SFPSWAP(GCD_SWAP_IMM12_INT32, WORK, GCD_LREG_B, p_sfpswap::ALL_ROWS_MAX);  // B = min, WORK = max
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);     // SFPSWAP is 2-cycle
    TTI_SFPIADD(0 /* imm12 */, GCD_LREG_B, WORK, GCD_IADD_MOD_SUB_KEEP_CC);        // WORK = min - max = -a'
}

/**
 * @brief Record the iteration pair and the prologue into the replay buffer.
 *
 * @note Called by @ref calculate_gcd on entry, before the row loop; every
 *       @ref _calculate_gcd_sfp_rows_ call replays what this records. Overwrites math-thread
 *       replay slots 0-30.
 */
inline void _record_gcd_replay_() {
    lltt::record(GCD_REPLAY_START, GCD_REPLAY_LEN);
    _emit_gcd_step_<GCD_LREG_A, GCD_LREG_TMP>();
    _emit_gcd_step_<GCD_LREG_TMP, GCD_LREG_A>();

    lltt::record(GCD_PROLOGUE_START, GCD_PROLOGUE_LEN);
    _emit_gcd_prologue_();
}

/**
 * @brief Compute gcd(|a|, |b|) for one SFP row pair, with a in GCD_LREG_A and b in GCD_LREG_B.
 *
 * Runs GCD_MAX_INPUT_BITS - 1 Stein iterations. max(a, b) does not shrink every step:
 * (2^30 + 3, 2^30 + 1) becomes (2^30 + 1, 2) after one step. What does hold is that a + b
 * (with a's extra trailing zeros stripped) at least halves every step, so for magnitudes below
 * 2^GCD_MAX_INPUT_BITS b reaches the gcd within GCD_MAX_INPUT_BITS - 1 steps, and the final update
 * of a is never needed. The budget has no slack: (2147483645, 3), (1073741821, 1073741827) and
 * (715827883, 1431655765) need all 30 steps; the test plants all three.
 *
 * @tparam SIGN_MAGNITUDE_FORMAT: Dest holds sign-magnitude Int32, so convert on entry
 * @note Leaves the result in GCD_LREG_B and clobbers GCD_LREG_A, GCD_LREG_TMP, GCD_LREG_KBIAS.
 *       No conversion back for sign-magnitude Dest: the result is non-negative, where both
 *       encodings agree.
 * @note Call @ref _record_gcd_replay_ before this function - the iterations run out of the
 *       replay buffer.
 */
template <bool SIGN_MAGNITUDE_FORMAT = false>
inline void _calculate_gcd_sfp_rows_() {
    if constexpr (SIGN_MAGNITUDE_FORMAT) {
        TTI_SFPCAST(GCD_LREG_A, GCD_LREG_A, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
        TTI_SFPCAST(GCD_LREG_B, GCD_LREG_B, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
    }

    lltt::replay(GCD_PROLOGUE_START, GCD_PROLOGUE_LEN);

    constexpr std::uint32_t STEIN_ITERATIONS = GCD_MAX_INPUT_BITS - 1;
    // The recorded body is an iteration pair, so the tail replay covers the last 2 iterations
    // and stops one instruction short: the final SFPIADD only updates a, which is discarded.
    constexpr std::uint32_t FULL_PAIRS = (STEIN_ITERATIONS - 1) / 2;
    constexpr std::uint32_t TAIL_LEN = GCD_STEP_INSTRS * (STEIN_ITERATIONS - 2 * FULL_PAIRS) - 1;
    static_assert(
        2 * FULL_PAIRS + (TAIL_LEN + 1) / GCD_STEP_INSTRS == STEIN_ITERATIONS, "replays must cover every step");
    static_assert(TAIL_LEN <= GCD_REPLAY_LEN, "tail must fit the recorded pair");

    for (std::uint32_t i = 0; i < FULL_PAIRS; i++) {
        lltt::replay(GCD_REPLAY_START, GCD_REPLAY_LEN);
    }
    lltt::replay(GCD_REPLAY_START, TAIL_LEN);

    TTI_SFPENCC(0 /* imm12 */, GCD_ENCC_MOD_RESET_RESULT);  // re-enable retired lanes
}

/**
 * @brief Elementwise gcd(|in0|, |in1|) over two Int32 Dest tiles, written to a third.
 *
 * gcd(0, b) = |b|, gcd(a, 0) = |a|, gcd(0, 0) = 0. The output tile may alias either input.
 *
 * @tparam SIGN_MAGNITUDE_FORMAT: Dest holds sign-magnitude Int32 rather than two's complement
 *         (e.g. an Int8 copy_tile through an fp32-accumulating FPU); converts both operands on load
 * @tparam ITERATIONS: SFP row pairs to process per face
 * @tparam TILE_SHAPE: Dest tile shape, which fixes the tile stride
 * @param dst_index_in0: Dest tile index of the first operand
 * @param dst_index_in1: Dest tile index of the second operand
 * @param dst_index_out: Dest tile index the result is written to
 * @note Operand magnitudes must fit in GCD_MAX_INPUT_BITS bits; -2^31 is out of contract because
 *       Quasar SFPABS saturates it to 2^31 - 1.
 * @note Overwrites math-thread replay slots 0-30 on every call; re-run the init of any other
 *       replay-buffer user afterwards. Has no init of its own, but its loads and stores rely on the
 *       ADDR_MOD_7 that _llk_math_eltwise_sfpu_init_ programs. Enables CC on entry and leaves
 *       CC.En = 1, CC.Res = 1 (the firmware default) on exit.
 * @note Clobbers LREG0-LREG3.
 */
template <
    bool SIGN_MAGNITUDE_FORMAT = false,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_gcd(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

    TTI_SFPENCC(GCD_ENCC_IMM12_ENABLE, GCD_ENCC_MOD_SET_EN);  // CC.En = 1, CC.Res = 1
    _record_gcd_replay_();

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const std::uint32_t row_offset = d << 1;  // one SFP row pair = 2 Dest rows
        TT_SFPLOAD(
            GCD_LREG_A, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, in0_offset + row_offset /* dest_reg_addr */);
        TT_SFPLOAD(
            GCD_LREG_B, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, in1_offset + row_offset /* dest_reg_addr */);
        _calculate_gcd_sfp_rows_<SIGN_MAGNITUDE_FORMAT>();
        TT_SFPSTORE(
            GCD_LREG_B, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, out_offset + row_offset /* dest_reg_addr */);
    }
}

}  // namespace sfpu
}  // namespace ckernel
