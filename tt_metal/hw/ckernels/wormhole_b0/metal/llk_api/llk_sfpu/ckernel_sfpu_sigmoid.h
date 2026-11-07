// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = sfpu_reciprocal_iter<2>(denominator);
    } else {
        result = sfpu_reciprocal_iter<1>(denominator);
    }

    return result;
}

template <int ITERATIONS>
inline void sigmoid_bf16_dest() {
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
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LCONST_0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMUL(p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            p_sfpu::LCONST_0,
            p_sfpu::LREG5,
            p_sfpu::LREG5,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG4, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_sigmoid_appx<ITERATIONS>();
    } else if constexpr (is_fp32_dest_acc_en) {
        static_assert(ITERATIONS % 2 == 0, "calculate_sigmoid processes rows in pairs");
        TTI_SFPLOADI(7, 2, 43579);
        TTI_SFPLOADI(7, 8, 16312);
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPLOAD(3, 0, 3, 0);
        TTI_SFPMOV(0, 3, 3, 1);
        TTI_SFPLOAD(2, 0, 3, 2);
        TTI_SFPMOV(0, 2, 2, 1);
        TTI_SFPMUL(7, 3, 9, 1, 0);
        TTI_SFPMUL(7, 2, 9, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 1, 1, 7);
        TTI_SFPCAST(1, 4, 0);
        TTI_SFPABS(0, 1, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 7);
        TTI_SFPCAST(0, 5, 0);
        TTI_SFPABS(0, 0, 0, 0);
        TTI_SFPLOADI(6, 2, 29184);
        TTI_SFPLOADI(6, 8, 48945);
        TTI_SFPMAD(4, 6, 3, 3, 0);
        TTI_SFPMAD(5, 6, 2, 6, 0);
        TTI_SFPLOADI(2, 2, 48782);
        TTI_SFPLOADI(2, 8, 46527);
        TTI_SFPMAD(4, 2, 3, 3, 0);
        TTI_SFPMAD(5, 2, 6, 2, 0);
        TTI_SFPSETSGN(0, 1, 4, 0);
        TTI_SFPSETSGN(0, 0, 5, 0);
        TTI_SFPLOADI(6, 1, 5541);
        TTI_SFPLOADI(1, 2, 12142);
        TTI_SFPLOADI(1, 8, 15369);
        TTI_SFPMAD(6, 3, 1, 0, 0);
        TTI_SFPMAD(6, 2, 1, 6, 0);
        TTI_SFPLOADI(1, 2, 44461);
        TTI_SFPLOADI(1, 8, 15658);
        TTI_SFPMAD(0, 3, 1, 0, 0);
        TTI_SFPMAD(6, 2, 1, 6, 0);
        TTI_SFPLOADI(1, 2, 43560);
        TTI_SFPLOADI(1, 8, 15914);
        TTI_SFPMAD(0, 3, 1, 0, 0);
        TTI_SFPMAD(6, 2, 1, 1, 0);
        TTI_SFPLOADI(6, 2, 65531);
        TTI_SFPLOADI(6, 8, 16127);
        TTI_SFPMAD(0, 3, 6, 0, 0);
        TTI_SFPMAD(1, 2, 6, 1, 0);
        TTI_SFPMAD(0, 3, 10, 0, 0);
        TTI_SFPMAD(1, 2, 10, 1, 0);
        TTI_SFPMAD(0, 3, 10, 3, 0);
        TTI_SFPMAD(1, 2, 10, 2, 0);
        TTI_SFPIADD(4095, 0, 0, 5);
        TTI_SFPMUL(0, 9, 9, 0, 0);
        TTI_SFPIADD(4095, 1, 1, 5);
        TTI_SFPMUL(1, 9, 9, 1, 0);
        TTI_SFPEXEXP(0, 3, 6, 1);
        TTI_SFPIADD(0, 4, 6, 4);
        TTI_SFPIADD(4095, 6, 4, 9);
        TTI_SFPADDI(32640, 0, 0);
        TTI_SFPIADD(3841, 6, 4, 1);
        TTI_SFPMOV(0, 6, 0, 0);
        TTI_SFPSETEXP(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPEXEXP(0, 2, 3, 1);
        TTI_SFPIADD(0, 5, 3, 4);
        TTI_SFPIADD(4095, 3, 4, 9);
        TTI_SFPADDI(32640, 1, 0);
        TTI_SFPIADD(3841, 3, 4, 1);
        TTI_SFPMOV(0, 3, 1, 0);
        TTI_SFPSETEXP(0, 2, 1, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPADD(10, 1, 10, 4, 0);
        TTI_SFPADD(10, 0, 10, 3, 0);
        TTI_SFPMOV(0, 4, 2, 2);
        TTI_SFPMOV(0, 3, 1, 2);
        TTI_SFPSETMAN(0, 11, 1, 0);
        TTI_SFPMAD(12, 1, 13, 6, 0);
        TTI_SFPSETMAN(0, 11, 2, 0);
        TTI_SFPMAD(6, 1, 14, 0, 0);
        TTI_SFPMAD(12, 2, 13, 5, 0);
        TTI_SFPMAD(1, 0, 10, 6, 0);
        TTI_SFPMAD(5, 2, 14, 5, 0);
        TTI_SFPMAD(0, 6, 0, 6, 0);
        TTI_SFPMAD(2, 5, 10, 0, 0);
        TTI_SFPMAD(1, 6, 10, 1, 0);
        TTI_SFPMAD(5, 0, 5, 0, 0);
        TTI_SFPNOT(0, 3, 5, 0);
        TTI_SFPMAD(2, 0, 10, 2, 0);
        TTI_SFPSETMAN(0, 5, 5, 1);
        TTI_SFPMULI(16128, 5, 0);
        TTI_SFPMAD(6, 1, 6, 6, 0);
        TTI_SFPNOT(0, 4, 1, 0);
        TTI_SFPSETMAN(0, 1, 1, 1);
        TTI_SFPMAD(0, 2, 0, 2, 0);
        TTI_SFPMULI(16128, 1, 0);
        TTI_SFPMUL(6, 5, 9, 0, 0);
        TTI_SFPMUL(2, 1, 9, 1, 0);
        TTI_SFPSETSGN(0, 0, 3, 0);
        TTI_SFPSTORE(3, 0, 3, 0);
        TTI_SFPSETSGN(0, 1, 4, 0);
        TTI_SFPSTORE(4, 0, 3, 2);
        TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 1
        for (int d = 2; d < ITERATIONS; d += 2) {
            TTI_REPLAY(0, 16, 0, 0);
            TTI_SFPLOADI(2, 2, 48782);
            TTI_SFPLOADI(2, 8, 46527);
            TTI_SFPMAD(4, 2, 3, 3, 0);
            TTI_SFPMAD(5, 2, 6, 2, 0);
            TTI_SFPSETSGN(0, 1, 4, 0);
            TTI_SFPSETSGN(0, 0, 5, 0);
            TTI_SFPLOADI(6, 1, 5541);
            TTI_SFPLOADI(1, 2, 12142);
            TTI_SFPLOADI(1, 8, 15369);
            TTI_SFPMAD(6, 3, 1, 0, 0);
            TTI_SFPMAD(6, 2, 1, 6, 0);
            TTI_SFPLOADI(1, 2, 44461);
            TTI_SFPLOADI(1, 8, 15658);
            TTI_SFPMAD(0, 3, 1, 0, 0);
            TTI_SFPMAD(6, 2, 1, 6, 0);
            TTI_SFPLOADI(1, 2, 43560);
            TTI_SFPLOADI(1, 8, 15914);
            TTI_SFPMAD(0, 3, 1, 0, 0);
            TTI_SFPMAD(6, 2, 1, 1, 0);
            TTI_SFPLOADI(6, 2, 65531);
            TTI_SFPLOADI(6, 8, 16127);
            TTI_SFPMAD(0, 3, 6, 0, 0);
            TTI_SFPMAD(1, 2, 6, 1, 0);
            TTI_SFPMAD(0, 3, 10, 0, 0);
            TTI_SFPMAD(1, 2, 10, 1, 0);
            TTI_SFPMAD(0, 3, 10, 3, 0);
            TTI_SFPMAD(1, 2, 10, 2, 0);
            TTI_SFPIADD(4095, 0, 0, 5);
            TTI_SFPMUL(0, 9, 9, 0, 0);
            TTI_SFPIADD(4095, 1, 1, 5);
            TTI_SFPMUL(1, 9, 9, 1, 0);
            TTI_SFPEXEXP(0, 3, 6, 1);
            TTI_SFPIADD(0, 4, 6, 4);
            TTI_SFPIADD(4095, 6, 4, 9);
            TTI_SFPADDI(32640, 0, 0);
            TTI_SFPIADD(3841, 6, 4, 1);
            TTI_SFPMOV(0, 6, 0, 0);
            TTI_SFPSETEXP(0, 3, 0, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPEXEXP(0, 2, 3, 1);
            TTI_SFPIADD(0, 5, 3, 4);
            TTI_SFPIADD(4095, 3, 4, 9);
            TTI_SFPADDI(32640, 1, 0);
            TTI_SFPIADD(3841, 3, 4, 1);
            TTI_SFPMOV(0, 3, 1, 0);
            TTI_SFPSETEXP(0, 2, 1, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPADD(10, 1, 10, 4, 0);
            TTI_SFPADD(10, 0, 10, 3, 0);
            TTI_SFPMOV(0, 4, 2, 2);
            TTI_SFPMOV(0, 3, 1, 2);
            TTI_SFPSETMAN(0, 11, 1, 0);
            TTI_SFPMAD(12, 1, 13, 6, 0);
            TTI_SFPSETMAN(0, 11, 2, 0);
            TTI_SFPMAD(6, 1, 14, 0, 0);
            TTI_SFPMAD(12, 2, 13, 5, 0);
            TTI_SFPMAD(1, 0, 10, 6, 0);
            TTI_SFPMAD(5, 2, 14, 5, 0);
            TTI_SFPMAD(0, 6, 0, 6, 0);
            TTI_SFPMAD(2, 5, 10, 0, 0);
            TTI_SFPMAD(1, 6, 10, 1, 0);
            TTI_SFPMAD(5, 0, 5, 0, 0);
            TTI_SFPNOT(0, 3, 5, 0);
            TTI_SFPMAD(2, 0, 10, 2, 0);
            TTI_SFPSETMAN(0, 5, 5, 1);
            TTI_SFPMULI(16128, 5, 0);
            TTI_SFPMAD(6, 1, 6, 6, 0);
            TTI_SFPNOT(0, 4, 1, 0);
            TTI_SFPSETMAN(0, 1, 1, 1);
            TTI_SFPMAD(0, 2, 0, 2, 0);
            TTI_SFPMULI(16128, 1, 0);
            TTI_SFPMUL(6, 5, 9, 0, 0);
            TTI_SFPMUL(2, 1, 9, 1, 0);
            TTI_SFPSETSGN(0, 0, 3, 0);
            TTI_SFPSTORE(3, 0, 3, 0);
            TTI_SFPSETSGN(0, 1, 4, 0);
            TTI_SFPSTORE(4, 0, 3, 2);
            TTI_INCRWC(0, 4, 0, 0);
        }
    } else {
        sigmoid_bf16_dest<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE) {
        recip_init<false, false>();
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
