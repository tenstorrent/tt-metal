// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <int ITERATIONS>
inline void calculate_silu_bf16_dest() {
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0xaa3b);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0xbfb8);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x3885);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0x3f80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_USHORT, 0x5ada);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0x33a8);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0xa418);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, 0x27ac);

    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG7, 0);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LCONST_0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG4, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    #pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        if (d == 0) {
            TTI_REPLAY(0, 16, 1, 1);
            TTI_SFPEXEXP(0, p_sfpu::LREG4, p_sfpu::LREG6, 0);
            TTI_SFPEXMAN(0, p_sfpu::LREG4, p_sfpu::LREG4, 0);
            TTI_SFPSHFT(0, p_sfpu::LREG6, p_sfpu::LREG4, 0);
            TTI_SFPEXEXP(0, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPEXEXP_MOD1_NODEBIAS);
            TTI_SFPEXMAN(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPEXMAN_MOD1_PAD9);
            TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
            TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG7, 0);
            TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2);
            TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG7, p_sfpu::LREG2, p_sfpu::LREG4, 0);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);
            TTI_SFPSETEXP(0, p_sfpu::LREG4, p_sfpu::LREG6, 0);
            TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);
            TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG7, 0);
            TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG6, 2);
            TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, p_sfpu::LREG6, 0);
            TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG6, p_sfpu::LREG13, p_sfpu::LREG5, 0);
        } else {
            TTI_REPLAY(0, 16, 0, 0);
        }
        TTI_SFPNOT(0, p_sfpu::LREG4, p_sfpu::LREG4, 0);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG14, p_sfpu::LREG5, 0);
        TTI_SFPSETMAN(0, p_sfpu::LREG4, p_sfpu::LREG4, 1);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, 0);
        TTI_SFPMULI(0xbf00, p_sfpu::LREG4, 0);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG5, 0);
        TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPMUL(p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LCONST_0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
        TTI_SFP_STOCH_RND( sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LCONST_0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG4, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <int ITERATIONS>
inline void calculate_silu_fp32_dest() {
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_USHORT, 0xaa3b);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0x3fb8);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x7200);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0xbf31);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    TTI_SFPNOP;
    TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
    TTI_SFPCAST(p_sfpu::LREG6, p_sfpu::LREG5, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0xbe8e);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, 0xb5bf);
    #pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        if (d == 0) {
            TTI_REPLAY(0, 16, 1, 1);
            TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG7, 0);
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATA, 0x15a5);
            TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG1, 0);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0x2f6e);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, 0xbc09);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0xadad);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, 0x3d2a);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0xaa28);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, 0xbe2a);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0xfffb);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, 0x3eff);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            TTI_SFPABS(0, p_sfpu::LREG6, p_sfpu::LREG6, 0);
        } else {
            TTI_REPLAY(0, 16, 0, 0);
        }
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG0, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG6, p_sfpu::LREG5, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
        TTI_SFPIADD(4095, p_sfpu::LREG0, p_sfpu::LREG0, 5);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPEXEXP(0, p_sfpu::LREG1, p_sfpu::LREG6, 1);
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 6);
        TTI_SFPIADD(4095, p_sfpu::LREG5, p_sfpu::LREG6, 9);
        TTI_SFPADDI(32640, p_sfpu::LREG0, 0);
        TTI_SFPIADD(3841, p_sfpu::LREG5, p_sfpu::LREG6, 1);
        TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
        TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPENCC(3, p_sfpu::LREG0, p_sfpu::LREG0, 10);
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);
        TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 2);
        TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG5, 0);
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG5, 0);
        TTI_SFPSETMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, 0);
        TTI_SFPMULI(48896, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG5, 0);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
        TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG5, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0xbe8e);
        TTI_SFPMUL(p_sfpu::LREG5, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG6, p_sfpu::LREG5, 0);
        TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, 0xb5bf);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LREG7, p_sfpu::LREG4, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (is_fp32_dest_acc_en) {
        calculate_silu_fp32_dest<ITERATIONS>();
    } else {
        calculate_silu_bf16_dest<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu always uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must
    // use non-approx sigmoid_init regardless of APPROXIMATION_MODE.
    sigmoid_init<false>();
}

}  // namespace ckernel::sfpu
