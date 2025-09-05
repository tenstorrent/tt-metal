// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp(uint exponent_size_8) {
    const vFloat zero = 0.0f;
    const vFloat one = 1.0f;
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(_sfpu_is_fp16_zero_(v, exponent_size_8)) { v = one; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] != 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            v_if(_sfpu_is_fp16_zero_(v, exponent_size_8)) { v = zero; }
            v_else { v = one; }
            v_endif;
        }

        // a[i] < 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            v_if(v >= 0.0f) { v = zero; }
            v_else { v = one; }
            v_endif;
        }

        // a[i] >= 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            v_if(v >= 0.0f) { v = one; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] > 0
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            v_if(v > 0.0f) { v = one; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] <= 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            v_if(v > 0.0f) { v = zero; }
            v_else { v = one; }
            v_endif;
        }

        dst_reg[0] = v;
        dst_reg++;
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
        TTI_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_7, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 == 0)
        TTI_SFPSETCC(0, 0, 0, check);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x0001);
        // end_if
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_7, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_eqz_uint32() {
    int scalar = -5;  // used for shift operation
    _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        TTI_SFPLZ(0, 0, 1, 4);    // result in lreg1 is leading zero count
        TTI_SFPSHFT(0, 2, 1, 0);  // 32 >> 5 = 1 else 0
        TTI_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_7, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_nez_uint32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 != 0)
        TTI_SFPSETCC(0, 0, 0, SFPSETCC_MOD1_LREG_NE0);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x0001);
        // end_if
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_7, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_int(int scalar) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;

        // a[i] != scalar
        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(v != scalar) { val = 1; }
            v_endif;
        }
        // a[i] == scalar
        else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(v == scalar) { val = 1; }
            v_endif;
        }
        dst_reg[0] = val;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
