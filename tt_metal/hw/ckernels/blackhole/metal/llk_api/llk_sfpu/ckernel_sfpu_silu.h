// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// ---------------------------------------------------------------------------------------------------------------
// bf16-dest SiLU implementations (P4_SILUFAST experiment). SILU_BF16_IMPL selects the one calculate_silu uses:
//   0 = the sfpi loop below (exp_21f + ARECIP + 1 Newton step, constants hoisted; 25 issue slots per row)
//   1 = the same arithmetic, bit-identical, software pipelined by hand in TTI: the head of row i+1 (load .. 1+exp)
//       runs in the latency slots of the tail of row i (recip .. store); x is reloaded from Dest for the final
//       multiply, the 255 clamp bound is Prgm2 (read-only as the SFPSWAP VC operand)
//   2 = P5 "CB e2 + NR" (P2_SILUPOLY): xc = clamp(x, +-86); f = -xc*log2e + 1.5*2^23; r = -xc*log2e - (f - M);
//       2^r ~ 1 + e1*r + e2*r^2; E = 2^r * 2^k by integer add of (f << 23); y = ARECIP(1 + E) + 1 Newton step,
//       out = bf16(x*y*(2 - (1+E)*y)). Not bit-identical to 0/1: <= 1 bf16 ulp from exact-rounded silu. One row
//       per iteration (19 issue slots + reload of x).
//   3 = P5 as 2, software pipelined like 1 (head of row i+1 in the tail latency slots of row i).
// 2 and 3 are bit-identical to each other. For x < -86 they return about x * 4.5e-38 (0 in 0/1); a Prgm2 bound of
// 87.5f instead of 86.0f keeps |x| <= 86 unchanged and returns 0 there. The fp32-dest arm is unchanged in all four;
// silu_init programs Prgm2 for 1/2/3 only, which the fp32 arm does not read.
// ---------------------------------------------------------------------------------------------------------------
#ifndef SILU_BF16_IMPL
#define SILU_BF16_IMPL 3
#endif

namespace silu_detail {
constexpr std::uint32_t LREG0 = 0, LREG1 = 1, LREG2 = 2, LREG3 = 3, LREG4 = 4, LREG5 = 5, LREG6 = 6, LREG7 = 7;
constexpr std::uint32_t LCONST_0 = 9, LCONST_1 = 10, LCONST_NEG1 = 11, LREG12 = 12, LREG13 = 13, LREG14 = 14;
// head of one row: t = clamp(-x/ln2 + 127, 0, 255) -> exp_21f(-x) -> d = 1 + exp(-x) in LREG5.
// LREG3 = x (dead after the first MAD), LREG7 = t, LREG5 = exponent then d, LREG6 = Horner temp.
// Constants: LREG1 = c2, LREG4 = c1, LREG2 = c0, LREG13 = 1/ln2, LREG14 = 255.
#define SILU_S1_HEAD_A(off)                                              \
    TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, off);                              \
    TTI_SFPLOADI(LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe); /* 127.0 */ \
    TTI_SFPMAD(LREG3, LREG13, LREG7, LREG7, 1 /* NEGATE_VA */)
#define SILU_S1_HEAD_B                                                                     \
    TTI_SFPSWAP(0, LREG7, LCONST_0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); /* t = max(t, 0) */   \
    TTI_SFPSWAP(0, LREG14, LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);   /* t = min(t, 255) */ \
    TTI_SFPEXEXP(0, LREG7, LREG5, 0);                                                      \
    TTI_SFPEXMAN(0, LREG7, LREG7, sfpi::SFPEXMAN_MOD1_PAD8);                               \
    TTI_SFPSHFT(0, LREG5, LREG7, 0);                                                       \
    TTI_SFPEXEXP(0, LREG7, LREG5, 1 /* no debias */);                                      \
    TTI_SFPEXMAN(0, LREG7, LREG7, sfpi::SFPEXMAN_MOD1_PAD9);                               \
    TTI_SFPCAST(LREG7, LREG7, 0);                                                          \
    TTI_SFPMAD(LREG7, LREG1, LREG4, LREG6, 0) /* q = frac*c2 + c1 */
#define SILU_S1_HEAD_C                                               \
    TTI_SFPMAD(LREG7, LREG6, LREG2, LREG7, 0); /* p = frac*q + c0 */ \
    TTI_SFPSETEXP(0, LREG7, LREG5, 0);                               \
    TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0) /* d = 1 + exp */
// tail of one row: y = recip(d) + 1 Newton step (guarded), out = bf16(x * y). LREG0 = y, LREG6 = t.
#define SILU_S1_TAIL_A                 \
    TTI_SFPARECIP(0, LREG5, LREG0, 0); \
    TTI_SFPMAD(LREG5, LREG0, LREG12, LREG6, 2 /* NEGATE_VC: d*y - 2 */)
#define SILU_S1_TAIL_CC                                              \
    TTI_SFPGT(0, LREG6, LCONST_0, 1 /* SET_CC: t < 0 */);            \
    TTI_SFPMAD(LREG6, LREG0, LCONST_0, LREG0, 3 /* y = -t*y - 0 */); \
    TTI_SFPENCC(3, 0, 0, 10)

template <int ITERATIONS>
inline void calculate_silu_bf16_s1() {
    TTI_SFPLOADI(LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x3885);  // c0 = 1.0017248f
    TTI_SFPLOADI(LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0x3f80);
    TTI_SFPLOADI(LREG4, sfpi::SFPLOADI_MOD0_USHORT, 0x5ada);  // c1 = 7.839635491371155e-08f
    TTI_SFPLOADI(LREG4, sfpi::SFPLOADI_MOD0_UPPER, 0x33a8);
    TTI_SFPLOADI(LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0xa418);  // c2 = 4.791750143340323e-15f
    TTI_SFPLOADI(LREG1, sfpi::SFPLOADI_MOD0_UPPER, 0x27ac);
    // prologue: head of row 0
    SILU_S1_HEAD_A(0);
    TTI_SFPNOP;  // SFPSWAP's first-cycle read of the MAD result is not covered by the stall logic
    SILU_S1_HEAD_B;
    SILU_S1_HEAD_C;
    if constexpr (ITERATIONS > 1) {
        // steady state: tail of row i (Dest offset 0) interleaved with the head of row i+1 (Dest offset 2)
        constexpr int BODY = 25;
        TTI_REPLAY(0, BODY, 1, 1);
        TTI_SFPARECIP(0, LREG5, LREG0, 0);
        TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 2);
        TTI_SFPMAD(LREG5, LREG0, LREG12, LREG6, 2);
        TTI_SFPLOADI(LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);
        TTI_SFPMAD(LREG3, LREG13, LREG7, LREG7, 1);
        SILU_S1_TAIL_CC;
        TTI_SFPSWAP(0, LREG7, LCONST_0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSWAP(0, LREG14, LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPEXEXP(0, LREG7, LREG5, 0);
        TTI_SFPEXMAN(0, LREG7, LREG7, sfpi::SFPEXMAN_MOD1_PAD8);
        TTI_SFPSHFT(0, LREG5, LREG7, 0);
        TTI_SFPEXEXP(0, LREG7, LREG5, 1);
        TTI_SFPEXMAN(0, LREG7, LREG7, sfpi::SFPEXMAN_MOD1_PAD9);
        TTI_SFPCAST(LREG7, LREG7, 0);
        TTI_SFPMAD(LREG7, LREG1, LREG4, LREG6, 0);
        TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 0);  // x of row i again
        TTI_SFPMAD(LREG7, LREG6, LREG2, LREG7, 0);
        TTI_SFPMUL(LREG3, LREG0, LCONST_0, LREG3, 0);  // x * y
        TTI_SFPSETEXP(0, LREG7, LREG5, 0);
        TTI_SFP_STOCH_RND(0, 0, LREG0, LREG3, LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);
        TTI_SFPSTORE(LREG3, 0, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
        for (int i = 2; i < ITERATIONS; i++) {
            TTI_REPLAY(0, BODY, 0, 0);
        }
    }
    // epilogue: tail of the last row
    SILU_S1_TAIL_A;
    SILU_S1_TAIL_CC;
    TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 0);
    TTI_SFPMUL(LREG3, LREG0, LCONST_0, LREG3, 0);
    TTI_SFP_STOCH_RND(0, 0, LREG0, LREG3, LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(LREG3, 0, ADDR_MOD_7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}

// ------------------------------------------------------------------------------------------------ P5 (IMPL 2, 3)
// Constants: LREG1 = e2, LREG2 = e1, LREG4 = M = 1.5*2^23, LREG13 = log2(e) (Prgm1), LREG14 = 86.0 (Prgm2).
inline void silu_p5_load_consts() {
    TTI_SFPLOADI(LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x9e22);  // e2 = 0.239861041f (0x3E759E22)
    TTI_SFPLOADI(LREG1, sfpi::SFPLOADI_MOD0_UPPER, 0x3e75);
    TTI_SFPLOADI(LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0xf3c2);  // e1 = 0.702938199f (0x3F33F3C2)
    TTI_SFPLOADI(LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0x3f33);
    TTI_SFPLOADI(LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0x4b40);  // M = 12582912.0f (0x4B400000)
}
constexpr std::uint32_t SHFT_IMM_FROM_VC = 5;  // SFPSHFT_MOD1_ARG_IMM | SFPSHFT_MOD1_ARG_IMM_USE_VC: VD = VC << imm
constexpr std::uint32_t IADD_CC_NONE = 4;      // VD = VC + VD, lane flags untouched

template <int ITERATIONS>
inline void calculate_silu_bf16_p5_single() {
    silu_p5_load_consts();
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, 0);                           // x
        TTI_SFPSETSGN(0, LREG0, LREG3, 1);                              // a = |x|
        TTI_SFPSWAP(0, LREG14, LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // a = min(a, 86)
        TTI_SFPSETSGN(0, LREG3, LREG0, 0);                              // xc = sign(x) * a  (into LREG0)
        TTI_SFPMAD(LREG0, LREG13, LREG4, LREG5, 1);                     // f = -xc*log2e + M
        TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG6, 1);                   // km = M - f (exact)
        TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);                // kk = f << 23
        TTI_SFPMAD(LREG0, LREG13, LREG6, LREG6, 1);                     // r = -xc*log2e + km
        TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, 0);                           // x again
        TTI_SFPMAD(LREG6, LREG1, LREG2, LREG7, 0);                      // p = r*e2 + e1
        TTI_SFPMAD(LREG7, LREG6, LCONST_1, LREG7, 0);                   // p = p*r + 1
        TTI_SFPIADD(0, LREG7, LREG5, IADD_CC_NONE);                     // E = p + kk (integer)
        TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);                // d = 1 + E
        TTI_SFPARECIP(0, LREG5, LREG7, 0);                              // y ~ 1/d
        TTI_SFPMAD(LREG5, LREG7, LCONST_1, LREG6, 1);                   // e = 1 - d*y
        TTI_SFPMUL(LREG0, LREG7, LCONST_0, LREG3, 0);                   // xy = x*y
        TTI_SFPMAD(LREG3, LREG6, LREG3, LREG3, 0);                      // o = xy*e + xy
        TTI_SFP_STOCH_RND(0, 0, LREG0, LREG3, LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(LREG3, 0, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

// Software-pipelined P5. Steady-state body (20 slots incl. INCRWC, one SFPSWAP bubble): tail of row i reads d_i
// (LREG5, from the previous body) and reloads x_i from Dest offset 0; the head of row i+1 loads Dest offset 2
// and leaves d_{i+1} in LREG5. Register use: LREG0 = y then km, LREG3 = x' then xc then p, LREG5 = d then f then
// kk then E then d', LREG6 = e then r, LREG7 = |x'| then x then x*y then out.
template <int ITERATIONS>
inline void calculate_silu_bf16_p5_pipe() {
    silu_p5_load_consts();
    // prologue: head of row 0
    TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 0);
    TTI_SFPSETSGN(0, LREG3, LREG7, 1);
    TTI_SFPSWAP(0, LREG14, LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSETSGN(0, LREG7, LREG3, 0);
    TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);
    TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG0, 1);
    TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);
    TTI_SFPMAD(LREG3, LREG13, LREG0, LREG6, 1);
    TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);
    TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);
    TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);
    TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);
    if constexpr (ITERATIONS > 1) {
        constexpr int BODY = 20;
        TTI_REPLAY(0, BODY, 1, 1);
        TTI_SFPARECIP(0, LREG5, LREG0, 0);                              // y = ~1/d
        TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 2);                           // x' (row i+1)
        TTI_SFPMAD(LREG5, LREG0, LCONST_1, LREG6, 1);                   // e = 1 - d*y
        TTI_SFPSETSGN(0, LREG3, LREG7, 1);                              // |x'|
        TTI_SFPSWAP(0, LREG14, LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // min(|x'|, 86)
        TTI_SFPSETSGN(0, LREG7, LREG3, 0);                              // xc = copysign(min(|x'|, 86), x')
        TTI_SFPLOAD(LREG7, 0, ADDR_MOD_7, 0);                           // x (row i)
        TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);                     // f = -xc*log2e + M
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);                   // xy = x*y
        TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG0, 1);                   // km = M - f
        TTI_SFPMAD(LREG7, LREG6, LREG7, LREG7, 0);                      // o = xy*e + xy
        TTI_SFPMAD(LREG3, LREG13, LREG0, LREG6, 1);                     // r = -xc*log2e + km
        TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);        // p = r*e2 + e1
        TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);  // kk = f << 23
        TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);     // p = p*r + 1
        TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);            // out (row i)
        TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);       // E = p + kk
        TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);  // d' = 1 + E
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
        for (int i = 2; i < ITERATIONS; i++) {
            TTI_REPLAY(0, BODY, 0, 0);
        }
    }
    // epilogue: tail of the last row
    TTI_SFPARECIP(0, LREG5, LREG0, 0);
    TTI_SFPLOAD(LREG7, 0, ADDR_MOD_7, 0);
    TTI_SFPMAD(LREG5, LREG0, LCONST_1, LREG6, 1);
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFPMAD(LREG7, LREG6, LREG7, LREG7, 0);
    TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}

// Fused SwiGLU (P6): out = bf16(silu(g) * u) in ONE pass, bf16 Dest. Same P5 pipe math as calculate_silu_bf16_p5_pipe,
// but the final bf16 rounding of silu(g) is dropped: o = xy*e + xy (fp32) is multiplied by u (loaded from the tile
// SWIGLU_UP_ROWS Dest rows after the gate tile, i.e. tile idx+1) and rounded once. Stored over the gate tile.
// LREG0 is free between the r MAD and the next ARECIP, so u lives there. No 0*x fix (0*inf/NaN differ from the
// two-pass path; finite inputs give the same value, sign of zero may differ).
constexpr std::uint32_t SWIGLU_UP_ROWS = 64;
template <int ITERATIONS>
inline void calculate_swiglu_bf16_p5_pipe() {
    silu_p5_load_consts();
    TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 0);
    TTI_SFPSETSGN(0, LREG3, LREG7, 1);
    TTI_SFPSWAP(0, LREG14, LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSETSGN(0, LREG7, LREG3, 0);
    TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);
    TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG0, 1);
    TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);
    TTI_SFPMAD(LREG3, LREG13, LREG0, LREG6, 1);
    TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);
    TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);
    TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);
    TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);
    if constexpr (ITERATIONS > 1) {
        constexpr int BODY = 22;
        TTI_REPLAY(0, BODY, 1, 1);
        TTI_SFPARECIP(0, LREG5, LREG0, 0);             // y = ~1/d
        TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 2);          // x' (row i+1)
        TTI_SFPMAD(LREG5, LREG0, LCONST_1, LREG6, 1);  // e = 1 - d*y
        TTI_SFPSETSGN(0, LREG3, LREG7, 1);             // |x'|
        TTI_SFPSWAP(0, LREG14, LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSETSGN(0, LREG7, LREG3, 0);                  // xc
        TTI_SFPLOAD(LREG7, 0, ADDR_MOD_7, 0);               // x (row i)
        TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);         // f
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);       // xy = x*y
        TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG0, 1);       // km
        TTI_SFPMAD(LREG7, LREG6, LREG7, LREG7, 0);          // o = xy*e + xy (fp32, not rounded)
        TTI_SFPMAD(LREG3, LREG13, LREG0, LREG6, 1);         // r (last read of LREG0 = km)
        TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS);  // u (row i)
        TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);          // p = r*e2 + e1
        TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);    // kk
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);       // o*u
        TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);       // p = p*r + 1
        TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);       // E = p + kk
        TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);  // d' = 1 + E
        TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);            // out over gate tile
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
        for (int i = 2; i < ITERATIONS; i++) {
            TTI_REPLAY(0, BODY, 0, 0);
        }
    }
    TTI_SFPARECIP(0, LREG5, LREG0, 0);
    TTI_SFPLOAD(LREG7, 0, ADDR_MOD_7, 0);
    TTI_SFPMAD(LREG5, LREG0, LCONST_1, LREG6, 1);
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFPMAD(LREG7, LREG6, LREG7, LREG7, 0);
    TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS);
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}

// ---------------------------------------------------------------------------------------------- P25 experiments
// GLU_SILU_VARIANT (compile-time, from env TT_MATMUL_GLU_SILU_VARIANT via the factory; unset = 0 = today):
//   1 = V1: today without the Newton step (out = (x*y)*u, y = SFPARECIP(d)); body 20 slots.
//   4 = V4a: V1 without the |x| <= 87.5 clamp; body 17 slots. WRONG for |g| > ~88 (exponent wrap).
//   5 = V4b: V4a with x*u formed early and multiplied by y right after SFPARECIP; body 16 slots. Same clamp caveat.
#ifndef GLU_SILU_VARIANT
#define GLU_SILU_VARIANT 0
#endif
#if GLU_SILU_VARIANT == 1
template <int ITERATIONS>
inline void calculate_swiglu_v1() {
    silu_p5_load_consts();
    TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 0);
    TTI_SFPSETSGN(0, LREG3, LREG7, 1);
    TTI_SFPSWAP(0, LREG14, LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSETSGN(0, LREG7, LREG3, 0);
    TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);
    TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG0, 1);
    TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);
    TTI_SFPMAD(LREG3, LREG13, LREG0, LREG6, 1);
    TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);
    TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);
    TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);
    TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);
    if constexpr (ITERATIONS > 1) {
        constexpr int BODY = 20;
        TTI_REPLAY(0, BODY, 1, 1);
        TTI_SFPARECIP(0, LREG5, LREG0, 0);     // y = ~1/d
        TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 2);  // x' (row i+1)
        TTI_SFPSETSGN(0, LREG3, LREG7, 1);     // |x'|
        TTI_SFPSWAP(0, LREG14, LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSETSGN(0, LREG7, LREG3, 0);                  // xc
        TTI_SFPLOAD(LREG7, 0, ADDR_MOD_7, 0);               // x (row i)
        TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);         // f
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);       // xy = x*y
        TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG0, 1);       // km
        TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);    // kk
        TTI_SFPMAD(LREG3, LREG13, LREG0, LREG6, 1);         // r
        TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS);  // u (row i)
        TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);          // p = r*e2 + e1
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);       // xy*u
        TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);       // p = p*r + 1
        TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);       // E = p + kk
        TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);  // d' = 1 + E
        TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
        for (int i = 2; i < ITERATIONS; i++) {
            TTI_REPLAY(0, BODY, 0, 0);
        }
    }
    TTI_SFPARECIP(0, LREG5, LREG0, 0);
    TTI_SFPLOAD(LREG7, 0, ADDR_MOD_7, 0);
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS);
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}
#endif
#if GLU_SILU_VARIANT == 4
template <int ITERATIONS>
inline void calculate_swiglu_v4a() {
    silu_p5_load_consts();
    TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 0);
    TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);
    TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG0, 1);
    TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);
    TTI_SFPMAD(LREG3, LREG13, LREG0, LREG6, 1);
    TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);
    TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);
    TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);
    TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);
    if constexpr (ITERATIONS > 1) {
        constexpr int BODY = 17;
        TTI_REPLAY(0, BODY, 1, 1);
        TTI_SFPARECIP(0, LREG5, LREG0, 0);                  // y = ~1/d
        TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 2);               // x' (row i+1), no clamp
        TTI_SFPLOAD(LREG7, 0, ADDR_MOD_7, 0);               // x (row i)
        TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);         // f
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);       // xy = x*y
        TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG0, 1);       // km
        TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);    // kk
        TTI_SFPMAD(LREG3, LREG13, LREG0, LREG6, 1);         // r
        TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS);  // u (row i)
        TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);          // p = r*e2 + e1
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);       // xy*u
        TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);       // p = p*r + 1
        TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);       // E = p + kk
        TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);  // d' = 1 + E
        TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
        for (int i = 2; i < ITERATIONS; i++) {
            TTI_REPLAY(0, BODY, 0, 0);
        }
    }
    TTI_SFPARECIP(0, LREG5, LREG0, 0);
    TTI_SFPLOAD(LREG7, 0, ADDR_MOD_7, 0);
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS);
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}
#endif
#if GLU_SILU_VARIANT == 5
// State at body entry: LREG5 = d(row i), LREG7 = x_i*u_i. Body writes out(row i) and builds d and x*u of row i+1.
template <int ITERATIONS>
inline void calculate_swiglu_v4b() {
    silu_p5_load_consts();
    TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 0);               // x0
    TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS);  // u0
    TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);         // f
    TTI_SFPMUL(LREG3, LREG0, LCONST_0, LREG7, 0);       // xu0
    TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG6, 1);       // km
    TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);    // kk
    TTI_SFPMAD(LREG3, LREG13, LREG6, LREG6, 1);         // r
    TTI_SFPNOP;
    TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);  // p
    TTI_SFPNOP;
    TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);  // p2
    TTI_SFPNOP;
    TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);
    TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);  // d0
    if constexpr (ITERATIONS > 1) {
        constexpr int BODY = 16;
        TTI_REPLAY(0, BODY, 1, 1);
        TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 2);          // x' (row i+1)
        TTI_SFPARECIP(0, LREG5, LREG0, 0);             // y_i = ~1/d_i
        TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);    // f'
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);  // out_i = xu_i * y_i
        TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG6, 1);  // km'
        TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS + 2);  // u' (row i+1)
        TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);                  // out_i
        TTI_SFPMAD(LREG3, LREG13, LREG6, LREG6, 1);             // r'
        TTI_SFPMUL(LREG3, LREG0, LCONST_0, LREG7, 0);           // xu' = x'*u'
        TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);              // p'
        TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);        // kk'
        TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);           // p2'
        TTI_INCRWC(0, 2, 0, 0);
        TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);       // E'
        TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);  // d'_{i+1}
#pragma GCC unroll 8
        for (int i = 2; i < ITERATIONS; i++) {
            TTI_REPLAY(0, BODY, 0, 0);
        }
    }
    TTI_SFPNOP;
    TTI_SFPARECIP(0, LREG5, LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFPNOP;
    TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPNOP;
    TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}
#endif
#if GLU_SILU_VARIANT == 6
// V5 (env 6): V4b restructure + two-sided clamp of the exp argument to [-87.5, 87.5] with 2 SFPSWAP on x' after xu =
// x'*u' is formed (silu(g) -> g for large positive g, 0 for large negative g, like V0). Prgm2 = 87.5 (LREG14), Prgm0 =
// -87.5 (LREG12, set in silu_init). SFPSWAP mode 1: dest = min(dest, VC); mode 9: dest = max(dest, VC); sign-magnitude
// float compare.
template <int ITERATIONS>
inline void calculate_swiglu_v5() {
    silu_p5_load_consts();
    TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 0);               // x0
    TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS);  // u0
    TTI_SFPMUL(LREG3, LREG0, LCONST_0, LREG7, 0);       // xu0
    TTI_SFPSWAP(0, LREG14, LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSWAP(0, LREG12, LREG3, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);
    TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);       // f
    TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG6, 1);     // km
    TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);  // kk
    TTI_SFPMAD(LREG3, LREG13, LREG6, LREG6, 1);       // r
    TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);        // p
    TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);     // p2
    TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);
    TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);  // d0
    if constexpr (ITERATIONS > 1) {
        constexpr int BODY = 18;
        TTI_REPLAY(0, BODY, 1, 1);
        TTI_SFPLOAD(LREG3, 0, ADDR_MOD_7, 2);          // x' (row i+1)
        TTI_SFPARECIP(0, LREG5, LREG0, 0);             // y_i
        TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);  // out_i = xu_i*y_i
        TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPLOAD(LREG0, 0, ADDR_MOD_7, SWIGLU_UP_ROWS + 2);  // u' (row i+1)
        TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);                  // out_i
        TTI_SFPMUL(LREG3, LREG0, LCONST_0, LREG7, 0);           // xu' = x'*u' (unclamped x')
        TTI_SFPSWAP(0, LREG14, LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSWAP(0, LREG12, LREG3, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);
        TTI_SFPMAD(LREG3, LREG13, LREG4, LREG5, 1);       // f'
        TTI_SFPMAD(LREG5, LCONST_1, LREG4, LREG6, 1);     // km'
        TTI_SFPSHFT(23, LREG5, LREG5, SHFT_IMM_FROM_VC);  // kk'
        TTI_SFPMAD(LREG3, LREG13, LREG6, LREG6, 1);       // r'
        TTI_SFPMAD(LREG6, LREG1, LREG2, LREG3, 0);        // p'
        TTI_SFPMAD(LREG3, LREG6, LCONST_1, LREG3, 0);     // p2'
        TTI_INCRWC(0, 2, 0, 0);
        TTI_SFPIADD(0, LREG3, LREG5, IADD_CC_NONE);
        TTI_SFPADD(LCONST_1, LREG5, LCONST_1, LREG5, 0);
#pragma GCC unroll 8
        for (int i = 2; i < ITERATIONS; i++) {
            TTI_REPLAY(0, BODY, 0, 0);
        }
    }
    TTI_SFPNOP;
    TTI_SFPARECIP(0, LREG5, LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(LREG7, LREG0, LCONST_0, LREG7, 0);
    TTI_SFPNOP;
    TTI_SFP_STOCH_RND(0, 0, LREG0, LREG7, LREG7, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPNOP;
    TTI_SFPSTORE(LREG7, 0, ADDR_MOD_7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}
#endif
}  // namespace silu_detail

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_swiglu() {
    static_assert(!is_fp32_dest_acc_en, "fused SwiGLU SFPU is bf16-dest only");
#if GLU_SILU_VARIANT == 1
    silu_detail::calculate_swiglu_v1<ITERATIONS>();
#elif GLU_SILU_VARIANT == 4
    silu_detail::calculate_swiglu_v4a<ITERATIONS>();
#elif GLU_SILU_VARIANT == 5
    silu_detail::calculate_swiglu_v4b<ITERATIONS>();
#elif GLU_SILU_VARIANT == 6
    silu_detail::calculate_swiglu_v5<ITERATIONS>();
#else
    silu_detail::calculate_swiglu_bf16_p5_pipe<ITERATIONS>();
#endif
}

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (!is_fp32_dest_acc_en && SILU_BF16_IMPL == 1) {
        silu_detail::calculate_silu_bf16_s1<ITERATIONS>();
        return;
    }
    if constexpr (!is_fp32_dest_acc_en && SILU_BF16_IMPL == 2) {
        silu_detail::calculate_silu_bf16_p5_single<ITERATIONS>();
        return;
    }
    if constexpr (!is_fp32_dest_acc_en && SILU_BF16_IMPL == 3) {
        silu_detail::calculate_silu_bf16_p5_pipe<ITERATIONS>();
        return;
    }
    // The exp's constants, loaded once and kept in LRegs for the whole loop (see _sfpu_sigmoid_); 1/ln2 is
    // Prgm1 from silu_init. x stays live across the sigmoid, so the fp32 arm has room for two of its
    // three: p1 stays a per-row literal.
    HoistedIf<!is_fp32_dest_acc_en> c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
    HoistedIf<is_fp32_dest_acc_en> neg_ln2_hi = EXP_FP32_NEG_LN2_HI, p0 = EXP_FP32_P0;
    constexpr float p1 = EXP_FP32_P1;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x, c0, c1, c2, neg_ln2_hi, p0, p1);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must use non-approx
    // sigmoid_init: it programs Prgm0 = 2.0f for the reciprocal and Prgm1 = 1/ln2 for the exp.
    sigmoid_init<false>();
#if SILU_BF16_IMPL == 1
    // Prgm2 = 255.0f: the exp_21f upper clamp bound of the pipelined bf16 path (SFPSWAP VC operand, read-only).
    // The fp32 arm does not read Prgm2.
    sfpi::vConstFloatPrgm2 = 255.0f;
#elif SILU_BF16_IMPL == 2 || SILU_BF16_IMPL == 3
    // Prgm2 = 86.0f: the P5 input clamp bound |x| <= 86 (SFPSWAP VC operand, read-only). Prgm1 = log2(e) is the
    // exp's 1/ln2 from sigmoid_init. The fp32 arm does not read Prgm2.
    sfpi::vConstFloatPrgm2 = 87.5f;
#endif
#if GLU_SILU_VARIANT == 6
    sfpi::vConstFloatPrgm0 = -87.5f;  // P25 V5: lower clamp bound (LREG12); the silu path does not read Prgm0
#endif
}

}  // namespace ckernel::sfpu
