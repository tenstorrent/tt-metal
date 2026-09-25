// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Selected Newton rows. Callers own replay, init and the destination walk.
namespace sfpi {
#if defined(ARCH_BLACKHOLE)
template <uint32_t Hold, uint32_t Advance>
__attribute__((always_inline)) inline void newton_cbrt_body() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, Hold, 0);                   // x
    TTI_SFPNOP;                                                        // replay load-to-consumer gap
    TTI_SFPABS(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 1);  // ax
    TTI_SFPCAST(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        0);      // f = bits*negative_third_256 + magic
    TTI_SFPNOP;  // MAD-to-shift gap
    TTI_SFPSHFT(0x008, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 7);
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        0);      // y*y
    TTI_SFPNOP;  // MUL-to-consumer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        0);      // d = ax*y*y
    TTI_SFPNOP;  // MUL-to-consumer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);      // c = d*y
    TTI_SFPNOP;  // MUL-to-consumer gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG5,
        0);      // C2*c + C1
    TTI_SFPNOP;  // MAD-to-consumer gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG0,
        0);                                                               // t = c*(C2*c+C1)+C0
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG3, 0);  // signed d
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);      // t*t
    TTI_SFPNOP;  // MUL-to-consumer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG3,
        0);      // y = signed d*(t*t)
    TTI_SFPNOP;  // MUL-to-RNE gap
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, 0, Advance, 0);
}
template <uint32_t Hold, uint32_t Advance>
inline void newton_rsqrt_body() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 0);  // 0: x
    TTI_SFPSHFT(0xFFF, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0,
                5);  // 1: i = bits(x) >> 1
    TTI_SFPIADD(0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0,
                6);  // 2: y = MAGIC - i
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);      // 3: xy = x*y
    TTI_SFPNOP;  // 4: xy producer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        1);                                           // 5: c = -(y*xy)
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0x7F80);  // 6: +inf / c gap
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG4,
        0);      // 7: C2 + c
    TTI_SFPNOP;  // 8: add producer gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG2,
        0);      // 9: t = c*(C2+c) + C1
    TTI_SFPNOP;  // 10: MAD producer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);      // 11: y = y*t
    TTI_SFPNOP;  // 12: y producer gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);      // 13: xy = x*y
    TTI_SFPNOP;  // 14: xy producer gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        1);  // 15: one_minus_xyy = 1 - y*xy
    TTI_SFPDIVP2(0xFFF, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5,
                 1);  // 16: half_y = y/2; independent gap
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG4,
               2);  // 17: x_bits
    TTI_SFPIADD(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4,
                6);  // 18: infinity_bits - x_bits
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0,
                 sfpi::SFPSETCC_MOD1_LREG_NE0);  // 19: x != inf
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0,
                 sfpi::SFPSETCC_MOD1_LREG_NE0);  // 20: and x != 0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        0);                    // 21: corrected y
    TTI_SFPCOMPC(0, 0, 0, 0);  // 22: else
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0,
               0);                                    // 23: zero/inf terminal
    TTI_SFPENCC(3, 0, 0, 10);                         // 24: close terminal branch
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0);    // 25: x < 0
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x7FC0);  // 26: qNaN
    TTI_SFPENCC(3, 0, 0, 10);                         // 27: close negative branch
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // 28: RNE
    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance,
                 0);  // 29: dest += 2
}
#endif

template <bool RawZero, bool NegativeZero, bool WhReplay, uint32_t Hold, uint32_t Advance>
inline void newton_sqrt_body() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, Hold, 0);  // 0: x
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0x7F80);  // 1: +inf (load-gap filler)
#if defined(ARCH_WORMHOLE)
    // Mod1=5 sets SFPSHFT_MOD1_ARG_IMM_USE_VC (=4), which is
    // Blackhole-only (BlackholeA0/.../SFPSHFT.md:39-40,60). Wormhole
    // masks Mod1 &= 1 and takes the shifted source from VB == VD
    // (WormholeB0/.../SFPSHFT.md:20-26), so this would ignore LREG2 and
    // compute LREG0 >>= 1 -- and LREG0 still holds the PREVIOUS
    // element's result from slot 24's SFP_STOCH_RND. That self-feeding
    // seed overflows to inf and is the measured 1598-non-finite defect.
    // Stage x into the shift's destination first.
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);  // 2a: L0 = x
    TTI_SFPSHFT(0xFFF, 0, ckernel::p_sfpu::LREG0, 1);                  // 2: i = bits(x)>>1
#else
    TTI_SFPSHFT(0xFFF, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 5);  // 2: i = bits(x)>>1
#endif
    TTI_SFPIADD(0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, 6);  // 3: y = MAGIC - i
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // 4: xy = x*y
#if defined(ARCH_WORMHOLE)
    // SFPMAD_MOD1_NEGATE_VA (=1) is Blackhole-only
    // (BlackholeA0/.../SFPMAD.md:50-51); WH SFPMAD Mod1 has only
    // INDIRECT_VA/VD (WormholeB0/.../SFPMAD.md:44-47) and SFPMUL is
    // "identical to SFPMAD" (WormholeB0/.../SFPMUL.md:3), so mod1=1 is
    // silently dropped and this would compute +y*xy. Materialise -y with
    // SFPMOV_MOD1_NEGATE (=1, WormholeB0/.../SFPMOV.md:41-43): an exact
    // sign-bit XOR on the simple sub-unit, whose page has no scheduling
    // clause, so it fits the gap NOP slot 5 already needed and is
    // readable on the next cycle. L3 is dead here (written slot 8, last
    // read slot 10), so no slot is added.
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3, 1);  // 5: ny = -y
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // 6: c = ny*xy
#else
    if constexpr (NegativeZero) {
        TTI_SFPLOAD(ckernel::p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold,
                    0);  // 5: original physical BF16
    } else {
        TTI_SFPNOP;  // 5
    }
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        1);  // 6: c = -(y*xy)
#endif
    if constexpr (WhReplay) {
        TTI_SFPLOAD(ckernel::p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);  // original sign/exponent
    } else if constexpr (NegativeZero) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT,
                     0x00ffu);  // 7
    } else {
        TTI_SFPNOP;  // 7
    }
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        0);  // 8: C2 + c
    if constexpr (WhReplay) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 0x80ff);
    } else {
        TTI_SFPNOP;  // 9
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        0);  // 10: t = c*(C2+c) + C1
    if constexpr (WhReplay) {
        TTI_SFPAND(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG7, 0);
    } else if constexpr (NegativeZero) {
        TTI_SFPAND(
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG3,
            1);  // 11: raw exponent marker after L3's final core read
    } else {
        TTI_SFPNOP;  // 11
    }
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // 12: y = y*t
    if constexpr (NegativeZero) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT,
                     0x8000u);  // 13
    } else {
        TTI_SFPNOP;  // 13
    }
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // 14: xy = x*y
#if defined(ARCH_WORMHOLE)
    // Second NEGATE_VA site; same substitution, same dead L3, same
    // reuse of the gap NOP at slot 15. No slot added.
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3, 1);  // 15: ny = -y
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG4,
        0);  // 16: 1 - y*xy
#else
    if constexpr (NegativeZero) {
        TTI_SFPAND(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG7,
                   1);  // 15: raw sign marker
    } else {
        TTI_SFPNOP;  // 15
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG4,
        1);  // 16: 1 - y*xy
#endif
    TTI_SFPDIVP2(0xFFF, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5, 1);  // 17: xy/2
    TTI_SFPIADD(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG6, 2);       // 18: CC: x < +inf
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        0);                    // 19: y = (1-y*xy)*(xy/2) + xy (predicated)
    TTI_SFPENCC(3, 0, 0, 10);  // 20: CC-balance window 1
    if constexpr (WhReplay) {
        // Positive exponent-zero words return +0. The physical sign then
        // owns every negative input, including zero/subnormal encodings.
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG7, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 0x8000);
        TTI_SFPAND(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG7, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG7, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    } else {
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, 0);  // 21: CC: x < 0
    }
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x7FC0);  // 22: y = qNaN (predicated)
    TTI_SFPENCC(3, 0, 0, 10);                         // 23: CC-balance window 2
    if constexpr (RawZero) {
        if constexpr (NegativeZero) {
            // The graph-derived partition is complete and disjoint: biased
            // exponent zero maps to +0, then its negative-sign partition maps
            // to the requested +Inf class. Classifier work occupies the
            // existing MAD gaps; this six-slot suffix keeps the body 32/32.
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0,
                         sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 24: exponent-zero
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB,
                         0x0000);  // 25: DAZ +0
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG7, 0,
                         sfpi::SFPSETCC_MOD1_LREG_NE0);  // 26: negative exponent-zero
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB,
                         0x7f80u);    // 27
            TTI_SFPENCC(0, 0, 0, 0);  // 28: closes combined predicate
            TTI_SFPNOP;               // 29
        } else {
            // The Newton seed consumes raw sign/mantissa bits before ordinary
            // arithmetic can observe the target DAZ quotient.  Reload the
            // original physical U16 only after every Newton temporary is dead;
            // LOADI supplies the raw-load hazard gap.  Physical exponent bits
            // are 7:0, so AND==0 selects exactly all 256 exponent-zero words.
            TTI_SFPLOAD(ckernel::p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold,
                        0);  // 24: raw physical BF16
            TTI_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT,
                         0x00ffu);  // 25: load gap/mask
            TTI_SFPAND(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3,
                       1);  // 26: raw exponent
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0,
                         sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 27: exponent-zero
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB,
                         0x0000);     // 28: declared DAZ result +0
            TTI_SFPENCC(0, 0, 0, 0);  // 29: row-local CC closure
        }
    }
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);            // 30: fp32 -> bf16 RNE
    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance, 0);  // 31: dest += 2
}
}  // namespace sfpi
