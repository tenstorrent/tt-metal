// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Selected Newton rows. Callers own replay, init and the destination walk.
namespace sfpi {
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

template <bool RawZero, bool NegativeZero, bool WhReplay, uint32_t Hold, uint32_t Advance>
inline void newton_sqrt_body() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, Hold, 0);                        // 0: x
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0x7F80);                        // 1: +inf (load-gap filler)
    TTI_SFPSHFT(0xFFF, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 5);  // 2: i = bits(x)>>1
    TTI_SFPIADD(0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, 6);     // 3: y = MAGIC - i
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // 4: xy = x*y
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
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 0x00ff);
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
    if constexpr (NegativeZero) {
        TTI_SFPIADD(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG7, 6);  // 0x8000 - raw: negative nonzero inputs
    } else {
        TTI_SFPNOP;  // 15
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG4,
        1);                                                                  // 16: 1 - y*xy
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
        // All exponent-zero words start at +0. Raw negative nonzero
        // encodings then retain the selected negative-domain result.
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG7, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOAD(ckernel::p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 0x8000);
        TTI_SFPIADD(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG7, 2);  // raw > 0x8000
    } else {
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, 0);  // 21: CC: x < 0
    }
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, 0x7FC0);  // 22: y = qNaN (predicated)
    TTI_SFPENCC(3, 0, 0, 10);                         // 23: CC-balance window 2
    if constexpr (RawZero) {
        if constexpr (NegativeZero) {
            // The graph-derived partition is complete and disjoint: biased
            // exponent zero maps to +0, then its negative-subnormal partition maps
            // to the requested +Inf class. Classifier work occupies the
            // existing MAD gaps; this six-slot suffix keeps the body 32/32.
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0,
                         sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 24: exponent-zero
            TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB,
                         0x0000);  // 25: DAZ +0
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG7, 0,
                         0);  // 26: negative subnormal; exact -0 retains +0
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
