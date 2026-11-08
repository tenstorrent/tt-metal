// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace ckernel::sfpu::ttpoly::log1p {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <
    uint32_t Degree,
    bool Normalized,
    uint32_t Hold,
    uint32_t LowerBits = 0x3f400000u,
    uint32_t QuotientBits = 0x7f800000u,
    uint32_t RawEqual = 0x00ffu>
inline void log1p_replay_body() {
    constexpr uint32_t exponent_scratch = false ? 5 : 6;
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, Hold, 0);  // x
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        0);  // u = x + 1.0      [MAD]
    if constexpr (Normalized) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (QuotientBits >> 16));
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    }
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0, (Normalized ? LowerBits >> 16 : 0x3f40u));
    TTI_SFPIADD(0, 1, 0, 6);    // e = bits(u) - bits(0.75)
    TTI_SFPSETMAN(0, 0, 0, 1);  // e: mantissa = 0
    TTI_SFPMOV(0, 0, 1, 2);     // L1 = e
    TTI_SFPIADD(0, 2, 1, 6);    // L1 = bits(x) - e  (2^-k x)
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0xC080);
    TTI_SFPMOV(0, 0, 2, 2);                          // L2 = e (x dead)
    TTI_SFPIADD(0, false ? 6 : 3, 2, 6);             // L2 = bits(-4) - e (s)
    TTI_SFPMULI(0xBE80, ckernel::p_sfpu::LREG2, 0);  // s *= -0.25 (exact) [MAD]
    if constexpr (Normalized) {
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG4,
            sfpi::SFPLOAD_MOD0_FMT_UINT16,
            Hold,
            0);  // raw BF16 word in the mandatory MAD gap
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG2,
        0);  // t = . - 1.0  [MAD]
    if constexpr (Normalized) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT,
                     RawEqual);  // MAD gap
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        0);  // r = 2^-k x + t   [MAD]
    if constexpr (Normalized) {
        TTI_SFPXOR(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4,
                   0);  // raw ^ 0x00ff in the mandatory MAD gap
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG3,
        0);  // p = c2*r + c1 [MAD]
    if constexpr (Normalized) {
        TTI_SFPAND(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, 0);
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG3,
        0);  // p = p*r + c0 [MAD]
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);                                 // rr = r*r         [MAD]
    TTI_SFPABS(0, 0, exponent_scratch, 0);  // |e| (int); MAD gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG1,
        0);  // res = rr*p + r   [MAD]
    TTI_SFPCAST(exponent_scratch, exponent_scratch, 0);
    TTI_SFPSETSGN(0, exponent_scratch, 0, 0);  // L0 = copysgn(e_float, e)
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        0);  // res += e_f*(ln2*2^-23) [MAD]
    if constexpr (Normalized) {
        TTI_SFPENCC(3, 0, 0, 10);  // close core; MAD gap
    } else {
        TTI_SFPNOP;
    }
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // fp32 -> bf16 RNE
    if constexpr (Normalized) {
        // Open the raw exponent-FF terminal as the last two recorded issues; the immediate suffix closes it.
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (QuotientBits >> 16));
    }
}
}  // namespace ckernel::sfpu::ttpoly::log1p
