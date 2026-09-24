// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -val;
        sfpi::dst_reg++;
    }
}

// Op class for float negation: -x.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Negative : SfpuUnaryOp<Negative<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate() {
        _calculate_negative_<APPROXIMATION_MODE, ITERATIONS>();
    }
};

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_int_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt val = sfpi::dst_reg[0];
        v_if(val != 0) { sfpi::dst_reg[0] = sfpi::as<sfpi::vInt>(-sfpi::as<sfpi::vFloat>(val)); }
        v_endif;
        sfpi::dst_reg++;
    }
}

// Op class for int32 negation: -x.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct NegativeInt : SfpuUnaryOp<NegativeInt<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate() {
        _calculate_negative_int_<APPROXIMATION_MODE, ITERATIONS>();
    }
};

}  // namespace sfpu
}  // namespace ckernel
