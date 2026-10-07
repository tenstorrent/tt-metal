// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

constexpr std::uint32_t LCM_MOV_MOD_COPY = 0x0;

// SFPMAD mod1 bit 0 negates src_a, so t = 1 - g*y is one MAD without a negated copy of g.
constexpr std::uint32_t LCM_MAD_MOD_PLAIN = 0x0;
constexpr std::uint32_t LCM_MAD_MOD_NEG_SRC_A = 0x1;

// SFPIADD with CC left untouched.
constexpr std::uint32_t LCM_IADD_MOD_ADD =
    sfpi::SFPIADD_MOD1_ARG_LREG_DST | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;  // dest = src_c + dest
constexpr std::uint32_t LCM_IADD_MOD_SUB =
    sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;  // dest = src_c - dest

// FP32_SM32_EN clear: SFPSETCC reads src_c as two's-complement int32.
constexpr std::uint32_t LCM_SETCC_IMM12_INT32 = 0x000;

// SFPSHFT logical, data in place, amount from lreg_c (negative = right). Raw values: sfpi's
// SHIFT_IMM / SHIFT_LREGC name bit 0 with the opposite polarity on Quasar.
constexpr std::uint32_t LCM_SHFT_MOD_VAR_LOGICAL = 0x0;
// SFPSHFT logical, data from lreg_c, amount from imm12.
constexpr std::uint32_t LCM_SHFT_MOD_IMM_FROM_C = 0x5;

// SFPMUL24 UPPER returns product bits [45:23].
constexpr std::uint32_t LCM_MUL24_HI_SHIFT = 23;

constexpr std::uint32_t LCM_SWAP_IMM12 = 0x0;

constexpr std::uint32_t LCM_ENCC_MOD_RESET = sfpi::SFPENCC_MOD1_EU_R1;

// SM32 cast is safe: every cast operand is non-negative.
constexpr std::uint32_t LCM_CAST_MOD_INT_TO_FP32 = sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE;

// Replay slots [0, 16): two GCD steps; [16, 30): the per-row quotient/product tail.
constexpr std::uint32_t LCM_GCD_REPLAY_SLOT = 0;
constexpr std::uint32_t LCM_GCD_REPLAY_LEN = 16;

constexpr std::uint32_t LCM_TAIL_REPLAY_SLOT = LCM_GCD_REPLAY_SLOT + LCM_GCD_REPLAY_LEN;
constexpr std::uint32_t LCM_TAIL_REPLAY_LEN = 14;

// q * |b| must stay below 2^31, and SFPMUL24 reads only the low 23 bits.
constexpr int LCM_MAX_INPUT_BITS = 15;

constexpr int lcm_gcd_replay_count(const int max_input_bits) { return max_input_bits / 2; }

// `#pragma GCC unroll` needs a non-dependent value.
constexpr int LCM_MAX_REPLAY_COUNT = lcm_gcd_replay_count(LCM_MAX_INPUT_BITS);

/**
 * @brief One Stein step: (a, b) <- (max - min, min) after stripping a's extra trailing zeros.
 * @note SFPLZ retires a == 0 lanes until the tail's SFPENCC; that keeps their gcd in LREG1.
 */
template <std::uint32_t LREG_NEG_A, std::uint32_t LREG_OUT>
inline void _calculate_lcm_gcd_step_() {
    TTI_SFPABS(LREG_NEG_A, LREG_OUT, sfpi::SFPABS_MOD1_INT);
    TTI_SFPAND(LREG_OUT, LREG_NEG_A);                                            // neg_a = lowest set bit of a
    TTI_SFPLZ(LREG_NEG_A, LREG_NEG_A, sfpi::SFPLZ_MOD1_CC_NE0);                  // = 31 - tz(a); retire a == 0 lanes
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG3, LREG_NEG_A, LCM_IADD_MOD_ADD);     // = k - tz(a), always <= 0
    TTI_SFPSHFT(0 /* imm12 */, LREG_NEG_A, LREG_OUT, LCM_SHFT_MOD_VAR_LOGICAL);  // out = a >> (tz(a) - k)
    TTI_SFPSWAP(LCM_SWAP_IMM12, LREG_OUT, p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // lreg1 = min, out = max
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);             // SFPSWAP is 2-cycle
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG1, LREG_OUT, LCM_IADD_MOD_SUB);
}

/**
 * @brief Replay the GCD body until LREG1 holds the gcd.
 * @note n-bit operands reach g within n - 1 steps (exhaustive check, n = 1..15), i.e. n / 2 replays.
 *       The SFPLZ retirement is load-bearing: rerunning a converged lane zeroes LREG1.
 */
template <int MAX_INPUT_BITS>
inline void _calculate_lcm_gcd_sfp_rows_() {
    constexpr int LCM_GCD_REPLAY_COUNT = lcm_gcd_replay_count(MAX_INPUT_BITS);

#pragma GCC unroll LCM_MAX_REPLAY_COUNT
    for (int r = 0; r < LCM_GCD_REPLAY_COUNT; r++) {
        lltt::replay(LCM_GCD_REPLAY_SLOT, LCM_GCD_REPLAY_LEN);
    }
}

/**
 * @brief Record the GCD and tail replay bodies that @ref calculate_lcm replays.
 * @note Re-run after any op that records into replay slots [0, 30).
 */
inline void calculate_lcm_init() {
    lltt::record(LCM_GCD_REPLAY_SLOT, LCM_GCD_REPLAY_LEN);

    // LREG2 and LREG0 trade roles so -a is back in LREG2 after two steps.
    _calculate_lcm_gcd_step_<p_sfpu::LREG2, p_sfpu::LREG0>();
    _calculate_lcm_gcd_step_<p_sfpu::LREG0, p_sfpu::LREG2>();

    lltt::record(LCM_TAIL_REPLAY_SLOT, LCM_TAIL_REPLAY_LEN);

    TTI_SFPENCC(0 /* imm12 */, LCM_ENCC_MOD_RESET);  // wake retired lanes; lreg1 = g

    // q = |a| / g: reciprocal seed + two Newton steps. The |a| cast keeps y from being consumed
    // by the instruction right after SFPNONLINEAR.
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG0, LCM_CAST_MOD_INT_TO_FP32);
    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpnonlinear::RECIP_MODE);  // lreg2 = y ~ 1/g
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG6, LCM_CAST_MOD_INT_TO_FP32);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, LCM_MAD_MOD_NEG_SRC_A);  // t = 1 - g*y
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, LCM_MAD_MOD_PLAIN);         // y = t*y + y
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, LCM_MAD_MOD_NEG_SRC_A);  // t = 1 - g*y
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, LCM_MAD_MOD_PLAIN);         // y = t*y + y
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG6, LCM_MAD_MOD_PLAIN);
    TTI_SFP_STOCH_RND(
        p_sfpu::sfp_stochrnd_rnd_mod::NearEven,
        0 /* imm8 */,
        0 /* lreg_b */,
        p_sfpu::LREG6,
        p_sfpu::LREG6,
        p_sfpu::sfp_stochrnd_mod::FP32_TO_UINT16);  // lreg6 = q, exact integer < 2^15

    // lcm = q * |b| from the SFPMUL24 halves. SFPSHFT/SFPIADD don't stall on a 2-cycle producer
    // (TEN-4581), so each consumer sits two instructions after its SFPMUL24.
    TTI_SFPMUL24(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG7, sfpi::SFPMUL24_MOD1_UPPER);    // lreg7 = (q*b) >> 23
    TTI_SFPMUL24(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPMUL24_MOD1_LOWER);    // lreg0 = low 23 bits
    TTI_SFPSHFT(LCM_MUL24_HI_SHIFT, p_sfpu::LREG7, p_sfpu::LREG7, LCM_SHFT_MOD_IMM_FROM_C);
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG7, p_sfpu::LREG0, LCM_IADD_MOD_ADD);              // lreg0 = lo + hi
}

/**
 * @brief Element-wise Int32 lcm(|a|, |b|) over Dest: Stein GCD, fp32 quotient, SFPMUL24 product.
 * @tparam SIGN_MAGNITUDE_FORMAT: Dest holds SMAG32; operands are converted on load.
 * @tparam ITERATIONS: 2-row iterations per face (8 for a 16-row face).
 * @param dst_index_out: May alias either input.
 * @note |a|, |b| must be <= 2^MAX_INPUT_BITS - 1. lcm(0, 0) = 0 only because SFPMUL24 by |b| = 0 is 0.
 * @note Call @ref calculate_lcm_init first. Clobbers LREG0..LREG7 and the CC state.
 */
template <
    bool SIGN_MAGNITUDE_FORMAT = false,
    int MAX_INPUT_BITS = LCM_MAX_INPUT_BITS,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_lcm(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        MAX_INPUT_BITS > 0 && MAX_INPUT_BITS <= LCM_MAX_INPUT_BITS, "lcm operands must fit a 15-bit magnitude budget");

    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

    for (int d = 0; d < ITERATIONS; d++) {
        // Explicit INT32 sfpmem for integer loads/stores (TEN-4674).
        TT_SFPLOAD(p_sfpu::LREG2, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, in0_offset + (d << 1));  // a
        TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, in1_offset + (d << 1));  // b

        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
        }

        TTI_SFPABS(p_sfpu::LREG2, p_sfpu::LREG4, sfpi::SFPABS_MOD1_INT);  // |a|, kept for the tail
        TTI_SFPABS(p_sfpu::LREG1, p_sfpu::LREG5, sfpi::SFPABS_MOD1_INT);  // |b|, kept for the tail

        // GCD setup: LREG2 = -a, LREG1 = b with exactly k trailing zeros, LREG3 = k - 31.
        TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG0, LCM_MOV_MOD_COPY);
        TTI_SFPOR(p_sfpu::LREG1, p_sfpu::LREG0);                                           // lreg0 = c = a | b
        TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, LCM_MOV_MOD_COPY);
        TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, p_sfpu::LREG3, LCM_IADD_MOD_SUB);     // lreg3 = -c
        TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3);                                          // lreg3 = 2^k
        TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG0, LCM_MOV_MOD_COPY);
        TTI_SFPAND(p_sfpu::LREG3, p_sfpu::LREG0);                                          // lreg0 = b & 2^k
        TTI_SFPSETCC(LCM_SETCC_IMM12_INT32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // lanes where b is even
        TTI_SFPSWAP(LCM_SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_SWAP);  // 2-cycle
        TTI_SFPENCC(0 /* imm12 */, LCM_ENCC_MOD_RESET);                                      // fills the SFPSWAP shadow
        TTI_SFPABS(p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPABS_MOD1_INT);
        TTI_SFPABS(p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPABS_MOD1_INT);
        TTI_SFPLZ(p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPLZ_MOD1_CC_NONE);              // lreg3 = 31 - k
        TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, p_sfpu::LREG3, LCM_IADD_MOD_SUB);  // lreg3 = k - 31
        TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, p_sfpu::LREG2, LCM_IADD_MOD_SUB);  // lreg2 = -a

        _calculate_lcm_gcd_sfp_rows_<MAX_INPUT_BITS>();

        lltt::replay(LCM_TAIL_REPLAY_SLOT, LCM_TAIL_REPLAY_LEN);

        // Non-negative result: no SMAG32 conversion needed on store.
        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, out_offset + (d << 1));
    }
}

}  // namespace sfpu
}  // namespace ckernel
