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

constexpr std::uint32_t GCD_ABS_MOD_INT32 = 0x0;
constexpr std::uint32_t GCD_MOV_MOD_COPY = 0x0;
constexpr std::uint32_t GCD_IADD_MOD_ADD_KEEP_CC = p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;  // dest = c + dest
constexpr std::uint32_t GCD_IADD_SUB_DEST = 0x2;
constexpr std::uint32_t GCD_IADD_MOD_SUB_KEEP_CC =
    GCD_IADD_SUB_DEST | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;  // dest = c - dest
constexpr std::uint32_t GCD_LZ_MOD_COUNT = 0x0;
constexpr std::uint32_t GCD_LZ_MOD_CC_NE0 = 0x2;                 // also CC.Res = (src != 0)
constexpr std::uint32_t GCD_SHFT_MOD_INPLACE_VAR_LOGICAL = 0x0;  // dest <<= c, negative c shifts right
constexpr std::uint32_t GCD_SETCC_IMM12_INT32 = 0x0;
constexpr std::uint32_t GCD_SETCC_MOD_EQ0 = 0x6;
// imm12 bit 0 = 0: int32 compare (assembly.yaml). An fp32 compare would see operands above
// 0x7F800000 as NaN.
constexpr std::uint32_t GCD_SWAP_IMM12_INT32 = 0x0;
constexpr std::uint32_t GCD_ENCC_IMM12_ENABLE = 0x1;
constexpr std::uint32_t GCD_ENCC_MOD_SET_EN = 0x2;
constexpr std::uint32_t GCD_ENCC_MOD_RESET_RESULT = 0x0;

// LREG1 carries b and ends as the result; LREG0 and LREG2 ping-pong between -a and the working a.
constexpr std::uint32_t GCD_LREG_A = p_sfpu::LREG0;
constexpr std::uint32_t GCD_LREG_B = p_sfpu::LREG1;
constexpr std::uint32_t GCD_LREG_TMP = p_sfpu::LREG2;
constexpr std::uint32_t GCD_LREG_KBIAS = p_sfpu::LREG3;  // k - 31, where 2^k = lsb(a | b)

// -2^31 is excluded: Quasar SFPABS saturates it to 2^31 - 1.
constexpr std::uint32_t GCD_MAX_INPUT_BITS = 31;

constexpr std::uint32_t GCD_REPLAY_DEPTH = 32;
constexpr std::uint32_t GCD_STEP_INSTRS = 7;
constexpr std::uint32_t GCD_REPLAY_START = 0;
constexpr std::uint32_t GCD_REPLAY_LEN = 2 * GCD_STEP_INSTRS;
static_assert(GCD_REPLAY_LEN < GCD_REPLAY_DEPTH, "replay length must fit the log2(depth) len field");

constexpr std::uint32_t GCD_PROLOGUE_START = GCD_REPLAY_START + GCD_REPLAY_LEN;
constexpr std::uint32_t GCD_PROLOGUE_LEN = 15;
static_assert(GCD_PROLOGUE_START + GCD_PROLOGUE_LEN <= GCD_REPLAY_DEPTH, "prologue must fit after the recorded pair");

/**
 * @brief Per-row setup: A = -|a|, B = |b| with exactly k = ctz(a | b) trailing zeros, KBIAS = k - 31.
 */
inline void _emit_gcd_prologue_() {
    TTI_SFPABS(GCD_LREG_A, GCD_LREG_A, GCD_ABS_MOD_INT32);
    TTI_SFPABS(GCD_LREG_B, GCD_LREG_B, GCD_ABS_MOD_INT32);
    TTI_SFPMOV(GCD_LREG_A, GCD_LREG_TMP, GCD_MOV_MOD_COPY);
    TTI_SFPOR(GCD_LREG_B, GCD_LREG_TMP);
    TTI_SFPMOV(GCD_LREG_TMP, GCD_LREG_KBIAS, GCD_MOV_MOD_COPY);
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, GCD_LREG_KBIAS, GCD_IADD_MOD_SUB_KEEP_CC);
    TTI_SFPAND(GCD_LREG_TMP, GCD_LREG_KBIAS);  // KBIAS = 2^k
    TTI_SFPMOV(GCD_LREG_B, GCD_LREG_TMP, GCD_MOV_MOD_COPY);
    TTI_SFPAND(GCD_LREG_KBIAS, GCD_LREG_TMP);
    // Swap where b has more than k trailing zeros, so each step strips a to exactly k.
    TTI_SFPSETCC(GCD_SETCC_IMM12_INT32, GCD_LREG_TMP, GCD_SETCC_MOD_EQ0);
    TTI_SFPSWAP(0 /* imm12 */, GCD_LREG_A, GCD_LREG_B, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPENCC(0 /* imm12 */, GCD_ENCC_MOD_RESET_RESULT);
    TTI_SFPLZ(GCD_LREG_KBIAS, GCD_LREG_KBIAS, GCD_LZ_MOD_COUNT);
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, GCD_LREG_KBIAS, GCD_IADD_MOD_SUB_KEEP_CC);
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, GCD_LREG_A, GCD_IADD_MOD_SUB_KEEP_CC);
}

/**
 * @brief One Stein step: strip a to k trailing zeros, then (a, b) <- (max - min, min); WORK = -a'.
 *
 * @note SFPSHFT only shifts its destination, hence the NEG_A/WORK ping-pong. Lanes reaching a == 0
 *       are retired by SFPLZ until the caller's SFPENCC, which keeps their B (the gcd) intact.
 */
template <std::uint32_t NEG_A, std::uint32_t WORK>
inline void _emit_gcd_step_() {
    static_assert(NEG_A != WORK, "the Stein step needs two distinct scratch registers");

    TTI_SFPABS(NEG_A, WORK, GCD_ABS_MOD_INT32);
    TTI_SFPAND(WORK, NEG_A);  // lsb(a)
    TTI_SFPLZ(NEG_A, NEG_A, GCD_LZ_MOD_CC_NE0);
    TTI_SFPIADD(0 /* imm12 */, GCD_LREG_KBIAS, NEG_A, GCD_IADD_MOD_ADD_KEEP_CC);  // k - ctz(a)
    TTI_SFPSHFT(0 /* imm12 */, NEG_A, WORK, GCD_SHFT_MOD_INPLACE_VAR_LOGICAL);
    // No NOP after: SFPSWAP always stalls the next SFPU op itself (TEN-4581).
    TTI_SFPSWAP(GCD_SWAP_IMM12_INT32, WORK, GCD_LREG_B, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPIADD(0 /* imm12 */, GCD_LREG_B, WORK, GCD_IADD_MOD_SUB_KEEP_CC);
}

inline void _record_gcd_replay_() {
    lltt::record(GCD_REPLAY_START, GCD_REPLAY_LEN);
    _emit_gcd_step_<GCD_LREG_A, GCD_LREG_TMP>();
    _emit_gcd_step_<GCD_LREG_TMP, GCD_LREG_A>();

    lltt::record(GCD_PROLOGUE_START, GCD_PROLOGUE_LEN);
    _emit_gcd_prologue_();
}

/**
 * @brief gcd of one SFP row pair, result in GCD_LREG_B.
 *
 * a + b at least halves per step, so 30 steps suffice for 31-bit operands, with no slack:
 * (2147483645, 3) needs all 30.
 */
template <bool SIGN_MAGNITUDE_FORMAT = false>
inline void _calculate_gcd_sfp_rows_() {
    if constexpr (SIGN_MAGNITUDE_FORMAT) {
        TTI_SFPCAST(GCD_LREG_A, GCD_LREG_A, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
        TTI_SFPCAST(GCD_LREG_B, GCD_LREG_B, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
    }

    lltt::replay(GCD_PROLOGUE_START, GCD_PROLOGUE_LEN);

    constexpr std::uint32_t STEIN_ITERATIONS = GCD_MAX_INPUT_BITS - 1;
    // The tail stops one instruction short: the final SFPIADD only updates a, which is discarded.
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
 * @brief Record the gcd replay bodies into math-thread replay slots 0-28.
 *
 * @note Re-run after any op that records into those slots.
 */
inline void calculate_gcd_init() { _record_gcd_replay_(); }

/**
 * @brief Elementwise Int32 gcd(|in0|, |in1|); the output tile may alias either input.
 *
 * @tparam SIGN_MAGNITUDE_FORMAT: Dest holds SMAG32; operands are converted on load
 * @note -2^31 is out of contract. Needs @ref calculate_gcd_init and the ADDR_MOD_7 from
 *       _llk_math_eltwise_sfpu_init_. Clobbers LREG0-LREG3.
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

    TTI_SFPENCC(GCD_ENCC_IMM12_ENABLE, GCD_ENCC_MOD_SET_EN);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const std::uint32_t row_offset = d << 1;
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
