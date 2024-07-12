// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define BIT_MASK_32 0xFFFFFFFF
#define SIGN 0x80000000
#define MAGNITUDE 0x7FFFFFFF


sfpi_inline vInt sfpu_twos_comp_to_sign_mag(vInt value) {
    v_if(value & SIGN) {
        vInt magnitude = (~value + 1) & MAGNITUDE;
        value = SIGN | magnitude;
    }
    v_endif;
    return value;
}

sfpi_inline vInt sfpu_sign_mag_to_twos_comp(vInt value) {
    v_if(value & SIGN) {
        vInt magnitude = value & MAGNITUDE;
        value = (~magnitude + 1) & BIT_MASK_32;
    }
    v_endif;
    return value;
}

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_col() {
    for (size_t i = 0; i < 2; ++i) {
        vInt a = dst_reg[i];
        a = sfpu_twos_comp_to_sign_mag(a);

        for (size_t j = 2; j < 8; j += 2) {
            vInt b = dst_reg[i + j];
            b = sfpu_twos_comp_to_sign_mag(b);
            a += b;
        }

        for (size_t j = 16; j < 24; j += 2) {
            vInt b = dst_reg[i + j];
            b = sfpu_twos_comp_to_sign_mag(b);
            a += b;
        }

        a = sfpu_sign_mag_to_twos_comp(a);
        dst_reg[i] = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_row() {
  for (size_t i = 0; i < 8; i += 2) {
        vInt a = dst_reg[i];
        a = sfpu_twos_comp_to_sign_mag(a);

        int arr[] = {1, 8, 9};
        for (size_t j = 0; j < sizeof(arr)/sizeof(arr[0]); ++j) {
            vInt b = dst_reg[i + arr[j]];
            b = sfpu_twos_comp_to_sign_mag(b);
            a += b;
        }

        a = sfpu_sign_mag_to_twos_comp(a);
        dst_reg[i] = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void sum_int_init() {
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void add_int(const uint dst_offset) {
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt a = dst_reg[0];
        vInt b = dst_reg[32];
        a = sfpu_twos_comp_to_sign_mag(a);
        b = sfpu_sign_mag_to_twos_comp(b);

        vInt r = a + b;
        r = sfpu_sign_mag_to_twos_comp(r);

        dst_reg[0] = r;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
