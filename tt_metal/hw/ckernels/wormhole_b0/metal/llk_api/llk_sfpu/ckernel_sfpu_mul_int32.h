// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size = 64;
    const uint offset0 = dst_index_in0 * dst_tile_size;
    const uint offset1 = dst_index_in1 * dst_tile_size;
    const uint offset_out = dst_index_out * dst_tile_size;

    // Split the 32-bit input values into 11-bit chunks:
    //
    //   a = (a2 << 22) | (a1 << 11) | a0
    //   b = (b2 << 22) | (b1 << 11) | b0
    //
    // This allows us to cast these values to fp32 without loss of
    // precision, and furthermore, we can compute:
    //
    //   a * b = (top << 22) + (mid << 11) + low   (mod 2**32)
    //
    // Where:
    //
    //   top = a0*b2 + a1*b1 + a2*b0 (maximum 8,378,371 < 2**23)
    //   mid = a0*b1 + a1*b0         (maximum 23 bits)
    //   low = a0*b0                 (maximum 22 bits)
    //
    // Adding the magic constant 2**23 to each fp32 sum makes its raw bit pattern
    // 0x4B000000 + sum, so the integer can be read straight from the bit pattern:
    //   * mid and top are shifted left by 11 and 22, and 0x4B000000's lowest set bit
    //     is bit 24, so the constant term vanishes mod 2**32;
    //   * low is not shifted, so its 0x4B000000 would survive. Instead of an SFPEXMAN
    //     the top chain starts from 2**23 + 724: (724 << 22) == -0x4B000000 mod 2**32,
    //     cancelling low's constant term in the final integer add. top + 724 stays
    //     below 2**23, so the fp32 sums remain exact.
    //
    // Issue schedule (22 slots per row, 2 of them SFPNOP). Each SFPLOADMACRO loads
    // one operand and schedules its shift / cast / multiply on the Round, Simple and
    // MAD sub-units (`[x]` below), so the issue stream only carries the loads, the
    // four masks, five multiplies and the five-instruction integer tail. The
    // sub-unit rules this obeys: one op per sub-unit per cycle, never a Simple and a
    // Round op in the same cycle, a MAD result is read two instructions later and
    // never in the instruction right after the MAD, and every scheduled op fires
    // before the row's own store, so rows do not overlap and the Dst address modes
    // are the same as before (ADDR_MOD_3 loads, ADDR_MOD_2 store).
    //
    //  t | issue                    | Simple          | MAD                  | Round
    // ---+--------------------------+-----------------+----------------------+------------
    //  0 | [b2] load b              |                 |                      |
    //  1 | nop                      |                 |                      | [b2] >>= 22
    //  2 | [a0] load a              | [b2] = cast     |                      |
    //  3 | a0 &= 0x7ff              | a0 &= 0x7ff     |                      |
    //  4 | [a1] load a              | [a0] = cast     |                      |
    //  5 | top = a0*b2 + (2**23+724)|                 | top = ...            | [a1] >>= 11
    //  6 | a1 &= 0x7ff              | a1 &= 0x7ff     |                      |
    //  7 | [b0] load b              | [a1] = cast     |                      |
    //  8 | b0 &= 0x7ff              | b0 &= 0x7ff     |                      |
    //  9 | [b1] load b              | [b0] = cast     |                      |
    // 10 | low = a0*b0 + 2**23      |                 | low = ...            | [b1] >>= 11
    // 11 | b1 &= 0x7ff              | b1 &= 0x7ff     |                      |
    // 12 | [a2] load a              | [b1] = cast     |                      |
    // 13 | top += a1*b1             |                 | top += a1*b1         | [a2] >>= 22
    // 14 | nop                      | [a2] = cast     | [b1] mid = a0*b1+2**23|
    // 15 | top += a2*b0             |                 | top += a2*b0         |
    // 16 | mid += a1*b0             |                 | mid += a1*b0         |
    // 17 | top <<= 22               | top <<= 22      |                      |
    // 18 | mid <<= 11               | mid <<= 11      |                      |
    // 19 | top += low               | top += low      |                      |
    // 20 | top += mid               | top += mid      |                      |
    // 21 | store top                |                 |                      |
    //
    // Macro index / shape (see mul_int32_init):
    //   0: a0, b0  cast at +2
    //   1: b1      >>11 at +1, cast at +3, mid = a0*b1 + 2**23 in place at +5
    //   2: a2, b2  >>22 at +1, cast at +2
    //   3: a1      >>11 at +1, cast at +3
    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;  // becomes mid
    constexpr int a2 = p_sfpu::LREG4;
    constexpr int b2 = p_sfpu::LREG5;  // becomes top, then the result
    constexpr int low = p_sfpu::LREG6;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOADMACRO((2 << 2) | (b2 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b2 >> 2));
        TTI_SFPNOP;
        TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a0 >> 2));
        TTI_SFPAND(0, p_sfpu::LREG12, a0, 0);
        TT_SFPLOADMACRO((3 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a1 >> 2));
        TTI_SFPMAD(a0, b2, p_sfpu::LREG13, b2, 0);
        TTI_SFPAND(0, p_sfpu::LREG12, a1, 0);
        TT_SFPLOADMACRO((0 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b0 >> 2));
        TTI_SFPAND(0, p_sfpu::LREG12, b0, 0);
        TT_SFPLOADMACRO((1 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b1 >> 2));
        TTI_SFPMAD(a0, b0, p_sfpu::LREG14, low, 0);
        TTI_SFPAND(0, p_sfpu::LREG12, b1, 0);
        TT_SFPLOADMACRO((2 << 2) | (a2 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a2 >> 2));
        TTI_SFPMAD(a1, b1, b2, b2, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(a2, b0, b2, b2, 0);
        TTI_SFPMAD(a1, b0, b1, b1, 0);
        TTI_SFPSHFT(22, 0, b2, 1);
        TTI_SFPSHFT(11, 0, b1, 1);
        TTI_SFPIADD(0, low, b2, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, b1, b2, sfpi::SFPIADD_MOD1_CC_NONE);
        TT_SFPSTORE(b2, InstrModLoadStore::INT32, ADDR_MOD_2, offset_out);
    }
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    sfpi::vConstIntPrgm0 = 0x7ff;                  // LREG12: 11-bit chunk mask
    sfpi::vConstFloatPrgm1 = 8388608.0f + 724.0f;  // LREG13: 2**23 + 724, start of the top chain
    sfpi::vConstFloatPrgm2 = 8388608.0f;           // LREG14: 2**23

    // SFPLOADMACRO instruction templates; an instruction with VD = 12 + i writes
    // template i. The macro sequence overrides VD (and VB, with the 0x80 flag) with
    // the loaded register, so the shifts and the cast act in place on it and the
    // MAD computes LREG0 * reg + LREG14 into it.
    TTI_SFPSHFT2(-11 & 0xfff, 0, 12, 6);                  // template 0: SFPSHFT2_MOD1_SHFT_IMM, >> 11
    TTI_SFPCAST(0, 13, 0);                                // template 1: int32 -> fp32
    TTI_SFPSHFT2(-22 & 0xfff, 0, 14, 6);                  // template 2: SFPSHFT2_MOD1_SHFT_IMM, >> 22
    TTI_SFPMAD(p_sfpu::LREG0, 0, p_sfpu::LREG14, 15, 0);  // template 3: reg = LREG0 * reg + 2**23

    // Sequence bits per sub-unit: 0x80 = VB is the loaded register, bits 5:3 = delay
    // (fires delay + 1 instructions after the load), bits 2:0 = 4 + template index.
    // Word layout: simple | mad << 8 | round << 16 | store << 24.
    // Macro 0 (a0, b0): cast at +2.
    TTI_SFPCONFIG((1 << 3) | (4 + 1), 4 + 0, 1);
    // Macro 1 (b1): >> 11 at +1, cast at +3, mid = a0 * b1 + 2**23 at +5.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, ((0x80 | (4 << 3) | (4 + 3)) << 8) | ((2 << 3) | (4 + 1)));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x80 | (0 << 3) | (4 + 0));
    TTI_SFPCONFIG(0, 4 + 1, 0);
    // Macro 2 (a2, b2): >> 22 at +1, cast at +2.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (1 << 3) | (4 + 1));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x80 | (0 << 3) | (4 + 2));
    TTI_SFPCONFIG(0, 4 + 2, 0);
    // Macro 3 (a1): >> 11 at +1, cast at +3.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (2 << 3) | (4 + 1));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x80 | (0 << 3) | (4 + 0));
    TTI_SFPCONFIG(0, 4 + 3, 0);
    // Misc: delays count issued instructions (UnitDelayKind = all 1s); no macro store.
    TTI_SFPCONFIG(0xff0, 8, 1);
}

}  // namespace ckernel::sfpu
