// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_is_fp16_zero.h"
#include "sfpu/ckernel_sfpu_load_config.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// These constants and function should ideally go to SFPI
// Copied from ckernel_sfpu_int_sum.h to avoid dependency complications
#ifndef SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED
#define SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED

#define BIT_MASK_32 0xFFFFFFFF
#define SIGN 0x80000000
#define MAGNITUDE 0x7FFFFFFF

// Convert from sign-magnitude to two's complement format
sfpi_inline vInt sfpu_sign_mag_to_twos_comp(vInt value) {
    v_if(value & SIGN) {
        vInt magnitude = value & MAGNITUDE;
        value = (~magnitude + 1) & BIT_MASK_32;
    }
    v_endif;
    return value;
}

#endif  // SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp() {
    constexpr uint V = p_sfpu::LREG0;
    constexpr uint ABS_V = p_sfpu::LREG2;
    constexpr uint INF = p_sfpu::LREG5;
    constexpr uint BFLOAT16_INF = 0x7f80;

    if constexpr (
        COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_equal_zero ||
        COMP_MODE == SfpuType::greater_than_zero || COMP_MODE == SfpuType::less_than_equal_zero) {
        TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, BFLOAT16_INF);
    }

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(V, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPSETSGN(0, V, ABS_V, 1);

        // eqz: default 0, set 1 where |v| == 0 (handles ±0; NaN has |v|!=0 → stays 0)
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // nez: default 1, set 0 where |v| == 0 (handles ±0; NaN has |v|!=0 → stays 1)
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // ltz: default 0; chain: (v < 0) AND (|v| != 0) AND (|v| <= inf) → 1; NaN: |NaN| > inf → rejected
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gtz: default 0; chain: (v >= 0) AND (|v| != 0) AND (|v| <= inf) → 1; NaN: |NaN| > inf → rejected
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gez: default 1; chain1: (v<0) AND (|v|!=0) → 0 (negatives excl. -0); chain2: |v|>inf → 0 (NaN)
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // lez: default 1; chain1: (v>=0) AND (|v|!=0) → 0 (positives excl. +0); chain2: |v|>inf → 0 (NaN)
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_int() {
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt zero = 0;

        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(v == zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] != 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            v_if(v == zero) { v = zero; }
            v_else { v = 1; }
            v_endif;
        }

        // a[i] < 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            v_if(v < zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] > 0
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            v_if(v > zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] <= 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            v_if(v <= zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] >= 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            v_if(v >= zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_uint16() {
    static_assert((COMP_MODE == SfpuType::equal_zero) or (COMP_MODE == SfpuType::not_equal_zero));
    constexpr int check = ((COMP_MODE == SfpuType::equal_zero) ? SFPSETCC_MOD1_LREG_EQ0 : SFPSETCC_MOD1_LREG_NE0);
    for (int d = 0; d < ITERATIONS; d++) {
        // load in conditional uint16 value
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 == 0)
        TTI_SFPSETCC(0, 0, 0, check);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x0001);
        // end_if
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_eqz_uint32() {
    int scalar = -5;  // used for shift operation
    _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPLZ(0, 0, 1, 4);    // result in lreg1 is leading zero count
        TTI_SFPSHFT(0, 2, 1, 0);  // 32 >> 5 = 1 else 0
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_nez_uint32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 != 0)
        TTI_SFPSETCC(0, 0, 0, SFPSETCC_MOD1_LREG_NE0);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x0001);
        // end_if
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_int(int scalar) {
    // Convert both operands to two's complement format
    //
    // LOGIC:
    // - Scalar is already in two's complement (from host)
    // - Convert SFPU input data from sign-magnitude to two's complement
    // - Perform comparison with both in two's complement format

    // Scalar stays in original two's complement format
    vInt converted_scalar = scalar;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;

        // Convert input data from sign-magnitude to two's complement
        v = sfpu_sign_mag_to_twos_comp(v);

        // Now both operands are in two's complement format
        // Use simple comparison like Blackhole
        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(v != converted_scalar) { val = 1; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(v == converted_scalar) { val = 1; }
            v_endif;
        }

        dst_reg[0] = val;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
