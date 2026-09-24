// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_col() {
    for (size_t i = 0; i < 2; ++i) {
        vInt a = dst_reg[i];

        for (size_t j = 2; j < 8; j += 2) {
            vInt b = dst_reg[i + j];
            a += b;
        }

        for (size_t j = 16; j < 24; j += 2) {
            vInt b = dst_reg[i + j];
            a += b;
        }

        dst_reg[i] = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_sum_int_row() {
    for (size_t i = 0; i < 8; i += 2) {
        vInt a = dst_reg[i];

        int arr[] = {1, 8, 9};
        for (size_t j = 0; j < sizeof(arr) / sizeof(arr[0]); ++j) {
            vInt b = dst_reg[i + arr[j]];
            a += b;
        }

        dst_reg[i] = a;
    }
}

template <bool APPROXIMATION_MODE>
inline void sum_int_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void add_int(const std::uint32_t dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt a = dst_reg[0];
        vInt b = dst_reg[32];

        vInt r = a + b;

        dst_reg[0] = r;
        dst_reg++;
    }
}

// Op class for an int32 column sum within a tile.
template <bool APPROXIMATION_MODE>
struct SumIntCol : SfpuUnaryOp<SumIntCol<APPROXIMATION_MODE>> {
    static constexpr auto& calculate = calculate_sum_int_col<APPROXIMATION_MODE>;
};

// Op class for an int32 row sum within a tile.
template <bool APPROXIMATION_MODE>
struct SumIntRow : SfpuUnaryOp<SumIntRow<APPROXIMATION_MODE>> {
    static constexpr auto& calculate = calculate_sum_int_row<APPROXIMATION_MODE>;
};

// Op class for an elementwise int32 add of a tile and the tile dst_offset tiles after it in Dest.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct AddIntDstOffset : SfpuUnaryOp<AddIntDstOffset<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = add_int<APPROXIMATION_MODE, ITERATIONS>;
};

}  // namespace sfpu
}  // namespace ckernel
