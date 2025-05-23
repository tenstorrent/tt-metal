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
    // vUInt zero = 23;
    // dst_reg[1] = zero;
    for (int d = 0; d < ITERATIONS; d++) {
        // vInt v = dst_reg[0];
        constexpr int cond_val_idx = 64;
        constexpr int other_val_idx = 128;
        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            TTI_SFPLOAD(0, 6, 3, 0);
            TTI_SFPSETCC(0, 0, 0, 6);
            TTI_SFPLOAD(0, 0, 3, 256);
            TTI_SFPSTORE(0, 0, 3, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            // v_if(v == zero) { v = 1; }
            // v_else { v = zero; }
            // v_endif;
        }

        // // a[i] != 0
        // if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
        //     v_if(v == zero) { v = zero; }
        //     v_else { v = 1; }
        //     v_endif;
        // }

        // // a[i] < 0
        // if constexpr (COMP_MODE == SfpuType::less_than_zero) {
        //     v_if(v < zero) { v = 1; }
        //     v_else { v = zero; }
        //     v_endif;
        // }

        // // a[i] > 0
        // if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
        //     v_if(v > zero) { v = 1; }
        //     v_else { v = zero; }
        //     v_endif;
        // }

        // // a[i] <= 0
        // if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
        //     v_if(v <= zero) { v = 1; }
        //     v_else { v = zero; }
        //     v_endif;
        // }

        // // a[i] >= 0
        // if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
        //     v_if(v >= zero) { v = 1; }
        //     v_else { v = zero; }
        //     v_endif;
        // }

        // dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_int(int scalar) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;
        vInt s = scalar;

        // a[i] != scalar
        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(v >= 0) {
                v_if(v != scalar) { val = 1; }
                v_endif;
            }  // negative comparison not working as expected in WH hence alternate implementation
            v_else {
                v_if(s < 0) {
                    vInt xor_val = reinterpret<vInt>(sfpi::abs(reinterpret<vFloat>(v))) ^ -s;
                    v_if(xor_val != 0) { val = 1; }
                    v_endif;
                }
                v_else { val = 1; }
                v_endif;
            }
            v_endif;
        }
        // a[i] == scalar
        else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(v >= 0) {
                v_if(v == scalar) { val = 1; }
                v_endif;
            }
            v_else {
                v_if(s < 0) {
                    vInt xor_val = reinterpret<vInt>(sfpi::abs(reinterpret<vFloat>(v))) ^ -s;
                    v_if(xor_val == 0) { val = 1; }
                    v_endif;
                }
                v_else { val = 0; }
                v_endif;
            }
            v_endif;
        }
        dst_reg[0] = val;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void comp_init() {
    // const uint one = 34;
    // _sfpu_load_imm16_(64, one);
}

}  // namespace sfpu
}  // namespace ckernel
