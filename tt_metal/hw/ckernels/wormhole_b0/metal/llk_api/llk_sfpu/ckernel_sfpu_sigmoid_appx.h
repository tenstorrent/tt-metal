// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Six-segment piecewise-linear sigmoid via SFPLUTFP32 (FP16_6ENTRY_TABLE2 | SGN_RETAIN),
// replacing the legacy three-segment SFPLUT table (0x3DFF/0x21D8/0xFF10), whose 4-bit-mantissa
// coefficients saturated to 1.0 for every |x| >= 2. Max |absolute error| 0.1192 -> 0.0177 and
// max relative error 0.1353 -> 0.0182, at 5 issue slots per datum instead of 7. The residual is
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

template <int K, int ITERATIONS>
sfpi_inline void _sigmoid_appx_lut6_step_() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    // LReg[7] = copysign(table(|LReg[3]|), LReg[3]), i.e. the odd part sigmoid(x) - 0.5.
    TTI_SFPLUTFP32(p_sfpu::LREG7, SIGMOID_APPX_LUT6_MOD);

    // Next datum's load fills the LUT's result-latency slot: it writes LReg[3], already
    // consumed by the LUT on issue, and leaves LReg[7] alone.
    if constexpr (K + 1 < ITERATIONS) {
        TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, 2 * (K + 1));
    } else {
        TTI_SFPNOP;
    }

    // Recentre. Two cycles after the LUT, so its result has landed.
    TTI_SFPADDI(SIGMOID_APPX_HALF_BF16, p_sfpu::LREG7, 0);

    // SFPADDI has the same one-cycle result latency, and unlike the LUT slot above there is
    // nothing independent left to cover it: the table owns LReg[0..2] and LReg[4..6], the LUT
    // reads LReg[3], and SFPLUTFP32 can only write VD < 8, so LReg[7] is the single staging
    // register and no second datum can be in flight. Removing this NOP needs a second one, and
    // the only candidate is LReg[16] -- reachable exclusively through SFPLOADMACRO.
    TTI_SFPNOP;

    TTI_SFPSTORE(p_sfpu::LREG7, IM, ADDR_MOD_3, 2 * K);

    if constexpr (K + 1 < ITERATIONS) {
        _sigmoid_appx_lut6_step_<K + 1, ITERATIONS>();
    }
}

// 5 issue slots per datum against the sfpi three-segment body's 7 (LOAD, LUT, NOP, ADDI, NOP,
// STORE, INCRWC): the per-datum TTINCRWC is gone because load and store use immediate dest
// offsets, and one of the two NOPs is now the next datum's load. Measured 1.10x.
template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, 0);
    _sigmoid_appx_lut6_step_<0, ITERATIONS>();
}

inline void sigmoid_appx_init() {
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
