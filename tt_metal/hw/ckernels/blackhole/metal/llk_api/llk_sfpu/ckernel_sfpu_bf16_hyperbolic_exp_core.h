// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"

// Selected P4 exp/reciprocal bodies. Callers own init, traversal and stores.
namespace sfpi {
template <uint32_t Hold>
inline void hyperbolic_exp_core() {
    ::ckernel::sfpu::bf16_sfpi::sfpload(ckernel::p_sfpu::LREG0, 0, Hold, 0);                      // 1: x
    ::ckernel::sfpu::bf16_sfpi::sfpsetsgn(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);  // 2: |x|
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0, 0);  // 3: xlog2
    ::ckernel::sfpu::bf16_sfpi::sfploadi(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);        // 4: 255
    ::ckernel::sfpu::bf16_sfpi::sfpswap(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, 1);               // 5: clamp
    ::ckernel::sfpu::bf16_sfpi::sfpexexp(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 0);              // 6
    ::ckernel::sfpu::bf16_sfpi::sfpexman(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);              // 7
    ::ckernel::sfpu::bf16_sfpi::sfpshft(0, 5, 0, 0);                                                         // 8
    ::ckernel::sfpu::bf16_sfpi::sfpexexp(
        0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, sfpi::SFPEXEXP_MOD1_NODEBIAS);  // 9
    ::ckernel::sfpu::bf16_sfpi::sfpexman(
        0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9);  // 10
    ::ckernel::sfpu::bf16_sfpi::sfpcast(0, 0, 0);                                      // 11
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG6,
        0);                                // 12: Horner head
    ::ckernel::sfpu::bf16_sfpi::sfpnop();  // 13: preserve L7 and the Horner-head source lifetime
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG6, 0);  // 14
    ::ckernel::sfpu::bf16_sfpi::sfpnop();  // 15: preserve L2 until the preceding Horner MAD retires
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG6, 0);  // 16
    ::ckernel::sfpu::bf16_sfpi::sfpnop();  // 17: preserve L3 until the preceding Horner MAD retires
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG0,
        0);                                               // 18: Horner c0
    ::ckernel::sfpu::bf16_sfpi::sfpshft(0x017, 5, 5, 7);  // 19: 2^i
    ::ckernel::sfpu::bf16_sfpi::sfpmul(
        ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0);  // 20: y
    ::ckernel::sfpu::bf16_sfpi::sfpnop();  // 21: y MAD hazard gap
    ::ckernel::sfpu::bf16_sfpi::sfparecip(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 0);  // 22
    ::ckernel::sfpu::bf16_sfpi::sfpnop();  // 23: conservative ARecip window
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG6,
        2);                                // 24: reciprocal Newton residual
    ::ckernel::sfpu::bf16_sfpi::sfpnop();  // 25
    {
        ::ckernel::sfpu::bf16_sfpi::Region cc_region;
        ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, ckernel::p_sfpu::LREG6, 0, 0, cc_region);  // 26
        ::ckernel::sfpu::bf16_sfpi::sfpmad(
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG5,
            3,
            cc_region);  // 27
        cc_region.close();
    };  // 28
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        0);  // 29: y-scale/y (L11 is programmed to -scale)
}

template <uint32_t Register, uint32_t Bits>
inline void hyperbolic_pin() {
    ::ckernel::sfpu::bf16_sfpi::sfploadi(Register, sfpi::SFPLOADI_MOD0_UPPER, Bits >> 16);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(Register, sfpi::SFPLOADI_MOD0_LOWER, Bits & 0xffffu);
}

template <typename Config>
inline void hyperbolic_exp_pins() {
    hyperbolic_pin<ckernel::p_sfpu::LREG1, Config::kMultiplierBits>();
    hyperbolic_pin<ckernel::p_sfpu::LREG2, Config::kScaledCoefficientBits[2]>();
    hyperbolic_pin<ckernel::p_sfpu::LREG3, Config::kScaledCoefficientBits[1]>();
    hyperbolic_pin<ckernel::p_sfpu::LREG4, Config::kScaledCoefficientBits[0]>();
    ::ckernel::sfpu::bf16_sfpi::sfploadi(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fc);
    hyperbolic_pin<ckernel::p_sfpu::LREG0, Config::kComposeScaleBits ^ (Config::kOdd ? 0x80000000u : 0u)>();
    ::ckernel::sfpu::bf16_sfpi::sfpconfig(0x5555, 11, 8);
}

inline void hyperbolic_restore_constant() {
    ::ckernel::sfpu::bf16_sfpi::sfploadi(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0xbf80);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    ::ckernel::sfpu::bf16_sfpi::sfpconfig(0x5555, 11, 8);
}

template <uint32_t Advance>
inline void hyperbolic_even_store() {
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfp_stoch_rnd(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    ::ckernel::sfpu::bf16_sfpi::sfpstore(ckernel::p_sfpu::LREG0, 0, Advance, 0);
}

}  // namespace sfpi
