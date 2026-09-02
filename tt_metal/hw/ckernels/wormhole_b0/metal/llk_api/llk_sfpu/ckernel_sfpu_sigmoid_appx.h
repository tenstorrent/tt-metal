// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Six-segment piecewise-linear sigmoid via SFPLUTFP32 (FP16_6ENTRY_TABLE2 | SGN_RETAIN),
// replacing the legacy three-segment SFPLUT table (0x3DFF/0x21D8/0xFF10), whose 4-bit-mantissa
// coefficients saturated to 1.0 for every |x| >= 2. Max |absolute error| 0.1192 -> 0.0177 and
// max relative error 0.1353 -> 0.0182, at 4 issue slots per datum instead of 7. The residual is
// set by the |x| >= 4 tail, whose slope must be exactly 0 or the fit diverges, which pins the
// error there at 1 - sigmoid(4).
//
// The table is tt-llk's own (common/inc/sfpu/ckernel_sfpu_sigmoid.h) with one deliberate change:
// B0, the first segment's intercept, is pinned to exactly 0 instead of -0.0004997, because
// SGN_RETAIN copies the input's sign onto the intercept too -- a non-zero B0 both makes
// sigmoid(0) return 0.5004997 and fits the segment containing the origin worse (1.4e-3 against
// 8.8e-4). Worth upstreaming.
//
// Raw TTI because sfpi cannot express this instruction: sfpi::lut2()'s six-register overload
// always ORs SGN_RETAIN into the mod and __builtin_rvtt_sfplutfp32_6r rejects every mod with
// that bit set, so it can never compile -- which also makes tt-llk's own six-entry
// _calculate_sigmoid_ dead code. The restriction is sfpi's, not the hardware's: the mod
// assembles and is verified on silicon.
constexpr int SIGMOID_APPX_LUT6_MOD = SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE2 | SFPLUTFP32_MOD0_SGN_RETAIN;

// bf16 encoding of 0.5, the immediate SFPADDI recentres the odd table with.
constexpr std::uint32_t SIGMOID_APPX_HALF_BF16 = 0x3F00;

// Two data per block, 8 slots: one load, one table lookup, one add and one store each, and
// nothing else. That is the floor for this kernel on Wormhole, where the SFPU retires one
// instruction per cycle, and reaching it turns on two properties of the instructions involved.
// (No per-datum TTINCRWC either: every load and store carries its dest offset as an immediate,
// which is also why the unroll below has to be a compile-time one.)
//
// Below, odd(x) is what the table returns: copysign(A*|x| + B, x) = sigmoid(x) - 0.5.
//
// SFPLUTFP32 reads LReg[3] no matter what, but its VD is a free field. So the second datum's
// lookup writes LReg[3] in place, which is a second staging register at no cost -- the table
// owns LReg[0..2] and LReg[4..6], and LReg[16] is reachable only through SFPLOADMACRO -- and it
// is what fills the first datum's SFPADDI latency slot.
//
// SFPADDI is in-place (VD += imm), so it cannot free the register it recentres. SFPMAD can:
// LReg[7] = LReg[3] * 1.0 + LReg[12] releases LReg[3] a slot earlier than an SFPADDI would, and
// the next block's load moves into the slot that covering the add would otherwise waste.
//
//   1 LUT     LReg[7] = odd(x_K)                     x_K already in LReg[3]
//   2 LOAD    LReg[3] = x_K+1                        covers the LUT
//   3 ADDI    LReg[7] += 0.5
//   4 LUT     LReg[3] = odd(x_K+1)                   in place, covers the SFPADDI
//   5 STORE   dst[K] = LReg[7]
//   6 MAD     LReg[7] = LReg[3] + 0.5                releases LReg[3]
//   7 LOAD    LReg[3] = x_K+2                        covers the SFPMAD
//   8 STORE   dst[K+1] = LReg[7]
//
// Every dependent pair is one slot apart, which is what SFPLUTFP32, SFPADDI and SFPMAD's 2-cycle
// result latency asks for. Measured 1.32x against the same table evaluated one datum at a time
// (33,494 -> 25,326 whole-loop MATH_ISOLATE, bf16, ITERATIONS=32, n300), and bit-identical to it
// on all 65,279 finite bf16 inputs and all 256 non-finite ones.
//
// Argument order, since none of it is named at the call site: TTI_SFPLOAD / TTI_SFPSTORE take
// (VD, Mod0, AddrMod, dest_reg_addr), TTI_SFPLUTFP32 takes (VD, instr_mod1), and the trailing 0
// on TTI_SFPADDI and TTI_SFPMAD is instr_mod1.
template <int K, int ITERATIONS>
sfpi_inline void _sigmoid_appx_lut6_pair_() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    TTI_SFPLUTFP32(p_sfpu::LREG7, SIGMOID_APPX_LUT6_MOD);
    TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, 2 * (K + 1));
    TTI_SFPADDI(SIGMOID_APPX_HALF_BF16, p_sfpu::LREG7, 0);
    TTI_SFPLUTFP32(p_sfpu::LREG3, SIGMOID_APPX_LUT6_MOD);
    TTI_SFPSTORE(p_sfpu::LREG7, IM, ADDR_MOD_3, 2 * K);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG12, p_sfpu::LREG7, 0);

    // The last block has no next datum to load, and the store below still needs the SFPMAD
    // covered.
    if constexpr (K + 2 < ITERATIONS) {
        TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, 2 * (K + 2));
    } else {
        TTI_SFPNOP;
    }

    TTI_SFPSTORE(p_sfpu::LREG7, IM, ADDR_MOD_3, 2 * (K + 1));
}

// Odd tail: one datum on its own, at 5 slots because there is no second datum to interleave with.
// Only reachable for odd ITERATIONS; every caller in metal passes 8.
template <int K, int ITERATIONS>
sfpi_inline void _sigmoid_appx_lut6_last_() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    TTI_SFPLUTFP32(p_sfpu::LREG7, SIGMOID_APPX_LUT6_MOD);
    TTI_SFPNOP;
    TTI_SFPADDI(SIGMOID_APPX_HALF_BF16, p_sfpu::LREG7, 0);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG7, IM, ADDR_MOD_3, 2 * K);
}

// The unroll: a fold over the block indices. Not a self-call at the end of each block, but the
// same instruction stream -- every dest offset stays an immediate, which it has to be, because
// TTI_* assembles the instruction word under an "n" asm constraint.
template <int ITERATIONS, int... P>
sfpi_inline void _sigmoid_appx_lut6_unroll_(std::integer_sequence<int, P...>) {
    (_sigmoid_appx_lut6_pair_<2 * P, ITERATIONS>(), ...);

    if constexpr (ITERATIONS % 2 != 0) {
        _sigmoid_appx_lut6_last_<ITERATIONS - 1, ITERATIONS>();
    }
}

template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    // Prologue load; every later load is issued inside a previous datum's latency slot.
    TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, 0);
    _sigmoid_appx_lut6_unroll_<ITERATIONS>(std::make_integer_sequence<int, ITERATIONS / 2>{});
}

inline void sigmoid_appx_init() {
    // LReg[12], the constant the second half of each block recentres with. Written before the
    // table: sfpi stages a programmable-constant write through LReg[0], which the table then
    // overwrites.
    sfpi::vConstFloatPrgm0 = 0.5f;

    // Six-piece fit of sigmoid(|x|) - 0.5. LReg[0..2] hold the slopes, LReg[4..6] the
    // intercepts, two Lut16ToFp32-encoded halves per register (low half = even segment,
    // high half = odd segment).
    //
    //   |x| <  0.5   0.2452*|x|                (intercept pinned to exactly 0, see above)
    //   |x| <  1.0   0.2173*|x| + 0.0152
    //   |x| <  1.5   0.1731*|x| + 0.05988
    //   |x| <  2.0   0.1262*|x| + 0.1298
    //   |x| <  4.0   0.0485*|x| + 0.2998
    //   |x| >= 4.0                0.4998

    // imm0[15:0] = A0 = 0.2452 = 0x33D9 -- imm0[31:16] = A1 = 0.2173 = 0x32F4
    l_reg[LRegs::LReg0] = vUInt(0x32F433D9);
    // imm4[15:0] = B0 = 0 = 0x7C00 -- imm4[31:16] = B1 = 0.0152 = 0x23C8
    // (Lut16ToFp32 encodes zero as exponent 31, hence 0x7C00 rather than 0x0000.)
    l_reg[LRegs::LReg4] = vUInt(0x23C87C00);

    // imm1[15:0] = A2 = 0.1731 = 0x318A -- imm1[31:16] = A3 = 0.1262 = 0x300A
    l_reg[LRegs::LReg1] = vUInt(0x300A318A);
    // imm5[15:0] = B2 = 0.05988 = 0x2BAA -- imm5[31:16] = B3 = 0.1298 = 0x3027
    l_reg[LRegs::LReg5] = vUInt(0x30272BAA);

    // imm2[15:0] = A4 = 0.0485 = 0x2A35 -- imm2[31:16] = A5 = 0.0 = 0x7C00
    l_reg[LRegs::LReg2] = vUInt(0x7C002A35);
    // imm6[15:0] = B4 = 0.2998 = 0x34CC -- imm6[31:16] = B5 = 0.4998 = 0x37FF
    l_reg[LRegs::LReg6] = vUInt(0x37FF34CC);
}

}  // namespace sfpu
}  // namespace ckernel
