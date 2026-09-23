// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#ifndef SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED
#define SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED

#define BIT_MASK_32 0xFFFFFFFF
#define SIGN 0x80000000
#define MAGNITUDE 0x7FFFFFFF

sfpi_inline vInt sfpu_sign_mag_to_twos_comp(vInt value) {
    v_if(value & SIGN) {
        vInt magnitude = value & MAGNITUDE;
        value = (~magnitude + 1) & BIT_MASK_32;
    }
    v_endif;
    return value;
}

#endif  // SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED

sfpi_inline vInt sfpu_twos_comp_to_sign_mag(vInt value) {
    v_if(value & SIGN) {
        vInt magnitude = (~value + 1) & MAGNITUDE;
        value = SIGN | magnitude;
    }
    v_endif;
    return value;
}

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_col() {
    for (unsigned i = 0; i < 2; ++i) {
        vInt a = dst_reg[i];
        a = sfpu_twos_comp_to_sign_mag(a);

        for (unsigned j = 2; j < 8; j += 2) {
            vInt b = dst_reg[i + j];
            b = sfpu_twos_comp_to_sign_mag(b);
            a += b;
        }

        for (unsigned j = 16; j < 24; j += 2) {
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
    for (unsigned i = 0; i < 8; i += 2) {
        vInt a = dst_reg[i];
        a = sfpu_twos_comp_to_sign_mag(a);

        int arr[] = {1, 8, 9};
        for (unsigned j = 0; j < sizeof(arr) / sizeof(arr[0]); ++j) {
            vInt b = dst_reg[i + arr[j]];
            b = sfpu_twos_comp_to_sign_mag(b);
            a += b;
        }

        a = sfpu_sign_mag_to_twos_comp(a);
        dst_reg[i] = a;
    }
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

// ---------------------------------------------------------------------------------------------------
// SumInt<APPROX, REDUCE_COLS, DST_SYNC, DST_ACCUM>
//   calculate(dst_index, vector_mode) -> calculate_sum_int_col / calculate_sum_int_row
//                                        (sfpu_sum_int_col, sfpu_sum_int_row)
//   init()                            -> shared SFPU init only
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, bool REDUCE_COLS, DstSync DST_SYNC, bool DST_ACCUM>
struct SumInt : SfpuUnaryOp<SumInt<APPROXIMATION_MODE, REDUCE_COLS, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM> {
    static void kernel() {
        if constexpr (REDUCE_COLS) {
            calculate_sum_int_col<APPROXIMATION_MODE>();
        } else {
            calculate_sum_int_row<APPROXIMATION_MODE>();
        }
    }
};

// ---------------------------------------------------------------------------------------------------
// SumIntAdd<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode, dst_offset) -> add_int (sfpu_add_int)
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct SumIntAdd : SfpuUnaryOp<SumIntAdd<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t dst_offset) { add_int<APPROXIMATION_MODE, ITERATIONS>(dst_offset); }
};
}  // namespace sfpu
}  // namespace ckernel
