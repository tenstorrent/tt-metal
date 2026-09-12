// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Six-segment piecewise-linear sigmoid via SFPLUTFP32 (FP16_6ENTRY_TABLE2 | SGN_RETAIN),
// replacing the legacy three-segment SFPLUT table (0x3DFF/0x21D8/0xFF10), whose 4-bit-mantissa
// coefficients saturated to 1.0 for every |x| >= 2. Measured over all 65279 finite bf16 inputs
// on p100a, scored against exact sigmoid over |x| <= 8 (where it is not saturated):
//
//                          max |err|   max rel err        rms        mean (bias)
//   three-segment table     0.119203      0.135335   0.007086         -2.150e-04
//   this table              0.017986      0.020028   0.000844         -2.743e-06
//
// at 4 issue slots per datum with a 32-bit Dest and 5 with a bf16 one, against the
// three-segment body's 5. The residual is set by the |x| >= 4 tail, whose slope must be
// exactly 0 or the fit diverges, which pins max |err| there at 1 - sigmoid(4) = 0.017986.
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
constexpr int SIGMOID_APPX_LUT6_MOD = sfpi::SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE2 | sfpi::SFPLUTFP32_MOD0_SGN_RETAIN;

// bf16 encoding of 0.5, the immediate SFPADDI recentres the odd table with.
constexpr std::uint32_t SIGMOID_APPX_HALF_BF16 = 0x3F00;

// Argument order below, since none of it is named at the call site: TTI_SFPLOAD /
// TTI_SFPSTORE take (VD, Mod0, AddrMod, dest_reg_addr), TTI_SFPLUTFP32 takes (VD, instr_mod1),
// and the trailing 0 on TTI_SFPADDI is instr_mod1.
template <int K, int ITERATIONS, bool is_fp32_dest_acc_en>
sfpi_inline void _sigmoid_appx_lut6_step_() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    // LReg[7] = copysign(table(|LReg[3]|), LReg[3]), i.e. the odd part sigmoid(x) - 0.5.
    TTI_SFPLUTFP32(p_sfpu::LREG7, SIGMOID_APPX_LUT6_MOD);

    // Fills the gap between the LUT and the SFPADDI that consumes it. Safe: it writes LReg[3],
    // which the LUT already consumed on issue, and leaves LReg[7] alone.
    if constexpr (K + 1 < ITERATIONS) {
        TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_7, 2 * (K + 1));
    }

    // Recentre.
    TTI_SFPADDI(SIGMOID_APPX_HALF_BF16, p_sfpu::LREG7, 0);

    // Round fp32 -> bf16 before SFPSTORE truncates into a 16-bit Dest, as ckernel_sfpu_exp.h
    // does. Only for a bf16 Dest: a 32-bit Dest holds the mantissa and the packer rounds on
    // the way out, so narrowing here would throw away precision for a cycle it does not need.
    //
    // Worth a full issue slot (+32.0 cycles/tile, exactly one SFPU instruction) because
    // truncation is *biased*, not merely imprecise. Truncating 0.5 + t always shrinks the
    // magnitude, so every output moves toward 0.5: mean signed error -2.397e-04, a quarter of
    // the total rms, all in one direction. Bias survives averaging and accumulates through
    // sums, softmax denominators and layer statistics, where random error of the same size
    // cancels. Rounding takes it to -2.743e-06 and rms from 0.001020 to 0.000844. It changes
    // 19391 of the 65279 finite bf16 outputs. Max |err| moves the other way, 0.017742 ->
    // 0.017986, because the truncation happened to point toward the true value at x = -4,
    // where the tail's pinned zero slope already sets the worst case.
    //
    // It cannot be hidden: it reads LReg[7], the same result the SFPSTORE below waits on, so
    // it serialises into the stall this body already has rather than filling it.
    if constexpr (!is_fp32_dest_acc_en) {
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            p_sfpu::LREG7,
            p_sfpu::LREG7,
            p_sfpu::LREG7,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    }

    // Adjacent to the SFPADDI it depends on: the one stall left in the body. Nothing independent
    // is left to cover it here, because this schedule keeps one datum in flight. Two are possible
    // -- SFPLUTFP32's VD is free, so a second datum's lookup can write LReg[3] in place and act
    // as a second staging register, which is what the Wormhole kernel does for 1.32x there. It is
    // not done here because Blackhole already issues this body at 4 slots per datum and the win
    // would be whatever hardware stall remains, which has not been measured.
    TTI_SFPSTORE(p_sfpu::LREG7, IM, ADDR_MOD_7, 2 * K);
}

// The unroll: a fold over the datum indices instead of a self-call at the end of every step. Same
// instruction stream -- every offset is still an immediate, which they have to be because TTI_*
// assembles the instruction word under an "n" asm constraint -- but each step is now a leaf and
// the template nesting depth no longer tracks ITERATIONS.
template <int ITERATIONS, bool is_fp32_dest_acc_en, int... K>
sfpi_inline void _sigmoid_appx_lut6_unroll_(std::integer_sequence<int, K...>) {
    (_sigmoid_appx_lut6_step_<K, ITERATIONS, is_fp32_dest_acc_en>(), ...);
}

// The per-datum TTINCRWC is gone because load and store use immediate dest offsets, and the next
// datum's load has moved into the LUT's shadow -- most of the saving comes from separating a
// producer from its consumer rather than from the slot TTINCRWC gave up. One stall survives; see
// the store above.
//
// Measured cycles/tile (Float16_b in and out, ITERATIONS=32, p100a, TILE_LOOP MATH_ISOLATE):
//
//   32-bit Dest   three-segment 220.500  ->  this body 189.516   1.163x
//   bf16 Dest     three-segment 220.523  ->  this body 221.516   0.996x
//
// The bf16 row is not a regression against a like-for-like baseline: the three-segment body
// truncates on store, and correcting that (the only way sfpi lets you, a convert<vFloat16b>,
// which costs two issue slots where the raw SFP_STOCH_RND below costs one) puts it at 283.531.
// Against that, this body is 1.28x. See the rounding step for why truncation is not an option
// worth keeping.
template <int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sigmoid_appx() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    // Prologue load; every later load is issued inside the previous datum's LUT shadow.
    TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_7, 0);
    _sigmoid_appx_lut6_unroll_<ITERATIONS, is_fp32_dest_acc_en>(std::make_integer_sequence<int, ITERATIONS>{});
}

inline void sigmoid_appx_init() {
    // Six-piece fit of sigmoid(|x|) - 0.5. LReg[0..2] hold the slopes, LReg[4..6] the
    // intercepts, two Lut16ToFp32-encoded halves per register (low half = even segment,
    // high half = odd segment).
    //
    //   |x| <  0.5   0.2452*|x|                (intercept pinned to exactly 0)
    //   |x| <  1.0   0.2173*|x| + 0.0152
    //   |x| <  1.5   0.1731*|x| + 0.05988
    //   |x| <  2.0   0.1262*|x| + 0.1298
    //   |x| <  4.0   0.0485*|x| + 0.2998
    //   |x| >= 4.0                0.4998

    // imm0[15:0] = A0 = 0.2452 = 0x33D9 -- imm0[31:16] = A1 = 0.2173 = 0x32F4
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x32F433D9);
    // imm4[15:0] = B0 = 0 = 0x7C00 -- imm4[31:16] = B1 = 0.0152 = 0x23C8
    // (Lut16ToFp32 encodes zero as exponent 31, hence 0x7C00 rather than 0x0000.)
    sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0x23C87C00);

    // imm1[15:0] = A2 = 0.1731 = 0x318A -- imm1[31:16] = A3 = 0.1262 = 0x300A
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x300A318A);
    // imm5[15:0] = B2 = 0.05988 = 0x2BAA -- imm5[31:16] = B3 = 0.1298 = 0x3027
    sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0x30272BAA);

    // imm2[15:0] = A4 = 0.0485 = 0x2A35 -- imm2[31:16] = A5 = 0.0 = 0x7C00
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x7C002A35);
    // imm6[15:0] = B4 = 0.2998 = 0x34CC -- imm6[31:16] = B5 = 0.4998 = 0x37FF
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vUInt(0x37FF34CC);
}

}  // namespace sfpu
}  // namespace ckernel
