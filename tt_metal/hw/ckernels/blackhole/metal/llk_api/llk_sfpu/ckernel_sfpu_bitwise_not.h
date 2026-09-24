// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void bitwise_not_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitwise_not() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = ~v;
        dst_reg++;
    }
}

// Op class for an elementwise bitwise NOT.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct BitwiseNot : SfpuUnaryOp<BitwiseNot<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate() {
        calculate_bitwise_not<APPROXIMATION_MODE, ITERATIONS>();
    }

    static inline __attribute__((always_inline)) void init_op() { bitwise_not_init(); }
};

}  // namespace sfpu
}  // namespace ckernel
