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

// SFPMOV mod1: plain copy.
constexpr std::uint32_t LCM_MOV_MOD_COPY = 0x0;

// SFPMAD mod1 bit 0 inverts the sign of src_a before the multiply (assembly.yaml), so
// t = 1 - g*y is a single MAD (-g)*y + 1 with no negated copy of g.
constexpr std::uint32_t LCM_MAD_MOD_PLAIN = 0x0;
constexpr std::uint32_t LCM_MAD_MOD_NEG_SRC_A = 0x1;

// SFPIADD mod1[1:0] selects the operation; bit 2 keeps CC untouched.
constexpr std::uint32_t LCM_IADD_MOD_ADD =
    sfpi::SFPIADD_MOD1_ARG_LREG_DST | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;  // dest = src_c + dest
constexpr std::uint32_t LCM_IADD_MOD_SUB =
    sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;  // dest = src_c - dest

// p_sfpu::cc::FP32_SM32_EN left clear: SFPSETCC reads src_c as two's-complement int32.
constexpr std::uint32_t LCM_SETCC_IMM12_INT32 = 0x000;

// SFPSHFT mod1 = 0: data from lreg_dest (in place), logical, amount from lreg_c (negative = right).
// sfpi's SFPSHFT_MOD1_SHIFT_IMM / SHIFT_LREGC name bit 0 with the opposite polarity on Quasar.
constexpr std::uint32_t LCM_SHFT_MOD_VAR_LOGICAL = 0x0;
// SFPSHFT mod1 = 5: data from lreg_c, logical, amount from imm12.
constexpr std::uint32_t LCM_SHFT_MOD_IMM_FROM_C = 0x5;

// SFPMUL24 UPPER returns product bits [45:23]; shift them back by 23 to recombine.
constexpr std::uint32_t LCM_MUL24_HI_SHIFT = 23;

// SFPSWAP compare type; every swapped operand is a non-negative int32.
constexpr std::uint32_t LCM_SWAP_IMM12 = 0x0;

// CC Result := 1 (re-enables every lane), CC Enable left as it was.
constexpr std::uint32_t LCM_ENCC_MOD_RESET = sfpi::SFPENCC_MOD1_EU_R1;

// int32 -> fp32, round to nearest even; every cast operand here is non-negative.
constexpr std::uint32_t LCM_CAST_MOD_INT_TO_FP32 = sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE;

// Replay slots [0, 16): two binary-GCD reduction iterations of 8 instructions each.
constexpr std::uint32_t LCM_GCD_REPLAY_SLOT = 0;
constexpr std::uint32_t LCM_GCD_REPLAY_LEN = 16;

// Replay slots [16, 30): the quotient and product tail, run once per row.
constexpr std::uint32_t LCM_TAIL_REPLAY_SLOT = LCM_GCD_REPLAY_SLOT + LCM_GCD_REPLAY_LEN;
constexpr std::uint32_t LCM_TAIL_REPLAY_LEN = 14;

// Operand magnitude ceiling in bits: q * |b| must stay below 2^31 and SFPMUL24 reads only the low 23 bits.
constexpr int LCM_MAX_INPUT_BITS = 15;

// GCD replays for an n-bit operand budget; see _calculate_lcm_gcd_sfp_rows_ for the bound.
constexpr int lcm_gcd_replay_count(const int max_input_bits) { return max_input_bits / 2; }

// Replays at the largest budget; `#pragma GCC unroll` needs a non-dependent value.
constexpr int LCM_MAX_REPLAY_COUNT = lcm_gcd_replay_count(LCM_MAX_INPUT_BITS);

/**
 * @brief One Stein reduction step: strip a's extra trailing zeros, then (a, b) <- (max - min, min).
 *
 * On entry LREG_NEG_A = -a and LREG1 = b, with b holding exactly k trailing zeros; LREG3 = k - 31.
 * On exit LREG_OUT = -(max - min), LREG1 = min. Lanes with a == 0 are retired in the CC by SFPLZ
 * and stay retired until the tail's SFPENCC, which is what keeps their gcd in LREG1 intact.
 *
 * @tparam LREG_NEG_A: LREG holding -a on entry; used as scratch.
 * @tparam LREG_OUT: LREG receiving -(max - min); holds a scratch copy of a on the way.
 */
template <std::uint32_t LREG_NEG_A, std::uint32_t LREG_OUT>
inline void _calculate_lcm_gcd_step_() {
    TTI_SFPABS(LREG_NEG_A, LREG_OUT, sfpi::SFPABS_MOD1_INT);                     // out = a
    TTI_SFPAND(LREG_OUT, LREG_NEG_A);                                            // neg_a = lowest set bit of a
    TTI_SFPLZ(LREG_NEG_A, LREG_NEG_A, sfpi::SFPLZ_MOD1_CC_NE0);                  // = 31 - tz(a); retire a == 0 lanes
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG3, LREG_NEG_A, LCM_IADD_MOD_ADD);     // = k - tz(a), always <= 0
    TTI_SFPSHFT(0 /* imm12 */, LREG_NEG_A, LREG_OUT, LCM_SHFT_MOD_VAR_LOGICAL);  // out = a >> (tz(a) - k)
    TTI_SFPSWAP(LCM_SWAP_IMM12, LREG_OUT, p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // lreg1 = min, out = max
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);             // post-SFPSWAP stall slot
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG1, LREG_OUT, LCM_IADD_MOD_SUB);                 // out = min - max
}

/**
 * @brief Replay the recorded binary-GCD reduction body enough times for LREG1 to reach the gcd.
 *
 * Divide out 2^k: after each step's shift both operands are odd, and (max - min, min) followed by
 * the next shift at least halves their sum, so the reduction converges logarithmically. For
 * |a|, |b| < 2^n LREG1 reaches g within n - 1 steps, i.e. n / 2 replays of the two-step body
 * (checked exhaustively over every operand pair for n = 1..15; tight at (3, 2^n - 3)). Later
 * steps would only drive a to 0, and the tail never reads a.
 *
 * The count is fixed rather than data-dependent: every lane runs the same replays. A lane whose
 * a has reached 0 must not run the body again — min(0, g) would zero LREG1 — so the SFPLZ
 * CC_NE0 retirement in each step is load-bearing, not an optimisation.
 *
 * @tparam MAX_INPUT_BITS: Operand magnitude budget in bits; sets the replay count.
 * @note Call only while the two-step body is live in replay slot @c LCM_GCD_REPLAY_SLOT —
 *       @ref calculate_lcm_init records it. On entry LREG2 = -a, LREG1 = b, LREG3 = k - 31; on exit
 *       LREG1 holds the gcd and the lanes whose operand hit zero are retired in the CC.
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
 * @brief Record the two lcm replay bodies: the two-step GCD reduction and the quotient/product tail.
 *
 * Every recorded instruction is an immediate that depends on neither the row, the face nor the
 * tile, so recording once here leaves @ref calculate_lcm issuing only replays.
 *
 * @note Call before @ref calculate_lcm, and again before resuming lcm after any op that records
 *       into replay slots [0, 30) — the usual *_tile_init convention.
 */
inline void calculate_lcm_init() {
    // The next 16 instructions are captured into the replay buffer, not executed.
    lltt::record(LCM_GCD_REPLAY_SLOT, LCM_GCD_REPLAY_LEN);

    // Two steps with LREG2 and LREG0 trading roles: -a in LREG2 on entry and exit.
    _calculate_lcm_gcd_step_<p_sfpu::LREG2, p_sfpu::LREG0>();
    _calculate_lcm_gcd_step_<p_sfpu::LREG0, p_sfpu::LREG2>();

    // Likewise recorded, not executed: the once-per-row quotient and product tail.
    lltt::record(LCM_TAIL_REPLAY_SLOT, LCM_TAIL_REPLAY_LEN);

    TTI_SFPENCC(0 /* imm12 */, LCM_ENCC_MOD_RESET);  // wake retired lanes; lreg1 = g

    // Exact quotient q = |a| / g via reciprocal seed + two Newton-Raphson steps. The |a| cast sits
    // between the LUT and the first MAD so y is not consumed by the instruction after SFPNONLINEAR.
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG0, LCM_CAST_MOD_INT_TO_FP32);         // lreg0 = float(g)
    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpnonlinear::RECIP_MODE);  // lreg2 = y ~ 1/g
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG6, LCM_CAST_MOD_INT_TO_FP32);         // lreg6 = float(|a|)
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, LCM_MAD_MOD_NEG_SRC_A);  // t = 1 - g*y
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, LCM_MAD_MOD_PLAIN);         // y = t*y + y
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, LCM_MAD_MOD_NEG_SRC_A);  // t = 1 - g*y
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, LCM_MAD_MOD_PLAIN);         // y = t*y + y
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG6, LCM_MAD_MOD_PLAIN);  // lreg6 = |a| * y
    TTI_SFP_STOCH_RND(
        p_sfpu::sfp_stochrnd_rnd_mod::NearEven,
        0 /* imm8 */,
        0 /* lreg_b */,
        p_sfpu::LREG6,
        p_sfpu::LREG6,
        p_sfpu::sfp_stochrnd_mod::FP32_TO_UINT16);  // lreg6 = q, exact integer < 2^15

    // lcm = q * |b| < 2^30 from the two SFPMUL24 halves. SFPSHFT/SFPIADD do not stall on a
    // 2-cycle producer (TEN-4581), so each consumer sits two instructions after its SFPMUL24.
    TTI_SFPMUL24(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG7, sfpi::SFPMUL24_MOD1_UPPER);    // lreg7 = (q*b) >> 23
    TTI_SFPMUL24(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPMUL24_MOD1_LOWER);    // lreg0 = low 23 bits
    TTI_SFPSHFT(LCM_MUL24_HI_SHIFT, p_sfpu::LREG7, p_sfpu::LREG7, LCM_SHFT_MOD_IMM_FROM_C);  // lreg7 <<= 23
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG7, p_sfpu::LREG0, LCM_IADD_MOD_ADD);              // lreg0 = lo + hi
}

/**
 * @brief Element-wise Int32 lcm(|in0|, |in1|) = (|in0| / gcd(|in0|, |in1|)) * |in1| over Dest.
 *
 * Runs in three stages per row pair: a binary (Stein) GCD driven from the replay buffer, an exact
 * integer quotient q = |a| / g computed in fp32 from an SFPNONLINEAR reciprocal seed refined by two
 * Newton-Raphson steps, and an exact q * |b| product recombined from the two SFPMUL24 halves. The
 * result is always non-negative.
 *
 * Zeros come from the tail, not the GCD stage: lcm(0, x) has g = |x| and q = 0, and lcm(x, 0) has
 * g = |x| and |b| = 0. For lcm(0, 0), g = 0, so the reciprocal seed is 1/0 and q is garbage; the
 * result is still 0 only because SFPMUL24 by |b| = 0 zeroes both product halves.
 *
 * @tparam SIGN_MAGNITUDE_FORMAT: Dest holds sign-magnitude Int32 (e.g. an Int8 copy_tile through an
 *         fp32 accumulating FPU) rather than two's complement; converts both operands on load.
 * @tparam MAX_INPUT_BITS: Operand magnitude budget in bits, 1..LCM_MAX_INPUT_BITS. Caps the GCD
 *         replay count.
 * @tparam ITERATIONS: 2-row SFPU iterations per face (8 for a 16-row face); runs once per face.
 * @tparam TILE_SHAPE: Dest tile shape, used to derive the per-tile stride.
 * @param dst_index_in0: Dest tile index of operand a.
 * @param dst_index_in1: Dest tile index of operand b.
 * @param dst_index_out: Dest tile index for the result; may alias either input.
 * @note Operands must satisfy |a|, |b| <= 2^MAX_INPUT_BITS - 1; larger inputs silently overflow.
 * @note Call @ref calculate_lcm_init first: this only replays the bodies it records. @c ADDR_MOD_7
 *       must already be programmed by the generic SFPU addrmod setup.
 * @note Clobbers LREG0..LREG7 and the CC state.
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

        TTI_SFPABS(p_sfpu::LREG2, p_sfpu::LREG4, sfpi::SFPABS_MOD1_INT);  // lreg4 = |a|, kept for the tail
        TTI_SFPABS(p_sfpu::LREG1, p_sfpu::LREG5, sfpi::SFPABS_MOD1_INT);  // lreg5 = |b|, kept for the tail

        // gcd setup: LREG2 = -a, LREG1 = b, LREG3 = k - 31, b odd relative to 2^k.
        TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG0, LCM_MOV_MOD_COPY);                        // lreg0 = a
        TTI_SFPOR(p_sfpu::LREG1, p_sfpu::LREG0);                                           // lreg0 = c = a | b
        TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, LCM_MOV_MOD_COPY);                        // lreg3 = c
        TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, p_sfpu::LREG3, LCM_IADD_MOD_SUB);     // lreg3 = -c
        TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3);                                          // lreg3 = 2^k
        TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG0, LCM_MOV_MOD_COPY);                        // lreg0 = b
        TTI_SFPAND(p_sfpu::LREG3, p_sfpu::LREG0);                                          // lreg0 = b & 2^k
        TTI_SFPSETCC(LCM_SETCC_IMM12_INT32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // lanes where b is even
        TTI_SFPSWAP(
            LCM_SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_SWAP);  // swap(a, b) there, 2-cycle
        TTI_SFPENCC(0 /* imm12 */, LCM_ENCC_MOD_RESET);  // re-enable all lanes; fills the SFPSWAP shadow
        TTI_SFPABS(p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPABS_MOD1_INT);                // lreg2 = |a|
        TTI_SFPABS(p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPABS_MOD1_INT);                // lreg1 = |b|
        TTI_SFPLZ(p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPLZ_MOD1_CC_NONE);              // lreg3 = 31 - k
        TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, p_sfpu::LREG3, LCM_IADD_MOD_SUB);  // lreg3 = k - 31
        TTI_SFPIADD(0 /* imm12 */, p_sfpu::LCONST_0, p_sfpu::LREG2, LCM_IADD_MOD_SUB);  // lreg2 = -a

        _calculate_lcm_gcd_sfp_rows_<MAX_INPUT_BITS>();

        lltt::replay(LCM_TAIL_REPLAY_SLOT, LCM_TAIL_REPLAY_LEN);  // lreg0 = lcm

        // Result is non-negative, so an SMAG32 Dest needs no conversion on the way out.
        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, out_offset + (d << 1));
    }
}

}  // namespace sfpu
}  // namespace ckernel
