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

// SFPMOV mod1: plain copy / copy with the sign bit inverted.
constexpr std::uint32_t LCM_MOV_MOD_COPY = 0x0;
constexpr std::uint32_t LCM_MOV_MOD_NEGATE = sfpi::SFPMOV_MOD1_COMPSIGN;

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

// Replay slots [16, 31): the quotient and product tail, run once per row.
constexpr std::uint32_t LCM_TAIL_REPLAY_SLOT = LCM_GCD_REPLAY_SLOT + LCM_GCD_REPLAY_LEN;
constexpr std::uint32_t LCM_TAIL_REPLAY_LEN = 15;

// (15 + 1) / 2 replays at the largest MAX_INPUT_BITS; `#pragma GCC unroll` needs a non-dependent value.
constexpr int LCM_MAX_REPLAY_COUNT = 8;

/**
 * @brief Replay the recorded binary-GCD reduction body enough times to reach its fixed point.
 *
 * Each replay covers two Stein iterations, and each iteration removes at least one bit from the
 * larger operand, so (MAX_INPUT_BITS + 1) / 2 replays always suffice. The count is fixed rather
 * than data-dependent: every SFPU lane runs the same number of replays, and a lane that has
 * already converged keeps reducing to the same value.
 *
 * @tparam MAX_INPUT_BITS: Operand magnitude budget in bits; sets the replay count.
 * @note Call only while the two-iteration body is live in replay slot @c LCM_GCD_REPLAY_SLOT —
 *       @ref calculate_lcm records it. On entry LREG2 = -a, LREG1 = b, LREG3 = k - 31; on exit
 *       LREG1 holds the gcd and the lanes whose operand hit zero are retired in the CC.
 */
template <int MAX_INPUT_BITS>
inline void _calculate_lcm_gcd_sfp_rows_() {
    constexpr int LCM_GCD_REPLAY_COUNT = (MAX_INPUT_BITS + 1) / 2;

#pragma GCC unroll LCM_MAX_REPLAY_COUNT
    for (int r = 0; r < LCM_GCD_REPLAY_COUNT; r++) {
        lltt::replay(LCM_GCD_REPLAY_SLOT, LCM_GCD_REPLAY_LEN);
    }
}

/**
 * @brief Element-wise Int32 lcm(|in0|, |in1|) = (|in0| / gcd(|in0|, |in1|)) * |in1| over Dest.
 *
 * Runs in three stages per row pair: a binary (Stein) GCD driven from the replay buffer, an exact
 * integer quotient q = |a| / g computed in fp32 from an SFPNONLINEAR reciprocal seed refined by two
 * Newton-Raphson steps, and an exact q * |b| product recombined from the two SFPMUL24 halves. The
 * result is always non-negative, and lcm(0, x) = lcm(x, 0) = 0 falls out of the GCD stage.
 *
 * @tparam SIGN_MAGNITUDE_FORMAT: Dest holds sign-magnitude Int32 (e.g. an Int8 copy_tile through an
 *         fp32 accumulating FPU) rather than two's complement; converts both operands on load.
 * @tparam MAX_INPUT_BITS: Operand magnitude budget in bits, 1..15. Caps the GCD replay count. 15 is
 *         the ceiling: q * |b| must stay below 2^31 and SFPMUL24 reads only the low 23 bits.
 * @tparam ITERATIONS: SFPU loop iterations over the Dest tile; each covers two Dest rows.
 * @tparam TILE_SHAPE: Dest tile shape, used to derive the per-tile stride.
 * @param dst_index_in0: Dest tile index of operand a.
 * @param dst_index_in1: Dest tile index of operand b.
 * @param dst_index_out: Dest tile index for the result; may alias either input.
 * @note Operands must satisfy |a|, |b| <= 2^MAX_INPUT_BITS - 1; larger inputs silently overflow.
 *       Needs no init call — the replay bodies are recorded on entry — but @c ADDR_MOD_7 must
 *       already be programmed by the generic SFPU addrmod setup.
 * @note Clobbers LREG0..LREG7, the CC state, and replay slots [0, 31).
 */
template <
    bool SIGN_MAGNITUDE_FORMAT = false,
    int MAX_INPUT_BITS = 15,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_lcm(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(MAX_INPUT_BITS > 0 && MAX_INPUT_BITS <= 15, "lcm operands must fit a 15-bit magnitude budget");

    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

    // The next 16 instructions are captured into the replay buffer, not executed.
    lltt::record(LCM_GCD_REPLAY_SLOT, LCM_GCD_REPLAY_LEN);

    // Phase A: LREG2 = -a on entry, LREG0 = -a' on exit.
    TTI_SFPABS(p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);   // lreg0 = a
    TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG2);                          // lreg2 = lowest set bit of a
    TTI_SFPLZ(p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPLZ_MOD1_CC_NE0);  // lreg2 = 31 - tz(a); retire lanes with a == 0
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG3, p_sfpu::LREG2, LCM_IADD_MOD_ADD);  // lreg2 = k - tz(a), always <= 0
    TTI_SFPSHFT(
        p_sfpu::LREG0 /* imm12: dest index; inert while the amount comes from lreg_c */,
        p_sfpu::LREG2,
        p_sfpu::LREG0,
        LCM_SHFT_MOD_VAR_LOGICAL);  // lreg0 = a >> (tz(a) - k)
    TTI_SFPSWAP(
        LCM_SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // lreg1 = min, lreg0 = max
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);          // post-SFPSWAP stall slot
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, LCM_IADD_MOD_SUB);         // lreg0 = min - max = -a'

    // Phase B: the same eight instructions with LREG0 and LREG2 trading roles.
    TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPABS_MOD1_INT);
    TTI_SFPAND(p_sfpu::LREG2, p_sfpu::LREG0);
    TTI_SFPLZ(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPLZ_MOD1_CC_NE0);
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG3, p_sfpu::LREG0, LCM_IADD_MOD_ADD);
    TTI_SFPSHFT(
        p_sfpu::LREG2 /* imm12: dest index; inert while the amount comes from lreg_c */,
        p_sfpu::LREG0,
        p_sfpu::LREG2,
        LCM_SHFT_MOD_VAR_LOGICAL);
    TTI_SFPSWAP(LCM_SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    TTI_SFPIADD(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG2, LCM_IADD_MOD_SUB);

    // Likewise recorded, not executed: the once-per-row quotient and product tail.
    lltt::record(LCM_TAIL_REPLAY_SLOT, LCM_TAIL_REPLAY_LEN);

    TTI_SFPENCC(0 /* imm12 */, LCM_ENCC_MOD_RESET);  // wake retired lanes; lreg1 = g

    // Exact quotient q = |a| / g via reciprocal seed + two Newton-Raphson steps.
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG0, LCM_CAST_MOD_INT_TO_FP32);                      // lreg0 = float(g)
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG6, LCM_CAST_MOD_INT_TO_FP32);                      // lreg6 = float(|a|)
    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpnonlinear::RECIP_MODE);               // lreg2 = y ~ 1/g
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, LCM_MOV_MOD_NEGATE);                             // lreg3 = -g
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG0, 0 /* mod1 */);  // t = 1 - g*y
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);     // y = t*y + y
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG0, 0 /* mod1 */);  // t = 1 - g*y
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);     // y = t*y + y
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG6, 0 /* mod1 */);  // lreg6 = |a| * y
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
