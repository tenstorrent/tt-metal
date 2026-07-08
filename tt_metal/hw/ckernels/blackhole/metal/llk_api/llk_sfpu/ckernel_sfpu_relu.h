// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_relu.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Full-range unsigned clamp against a threshold, shared by relu_min and relu_max/relu6.
// IS_LOWER_BOUND selects the direction using the user-facing op's semantics:
//   IS_LOWER_BOUND == true  -> relu_min:        result = max(x, threshold)
//   IS_LOWER_BOUND == false -> relu_max/relu6:  result = min(x, threshold)
//
// The compare is done on two 16-bit halves instead of a single 32-bit compare because sfpi's vUInt
// compare does a subtract-and-test-sign, which overflows when the two operands are >= 2^31
// apart and gives wrong results above 2^31. Sign flip compare x ^ 0x80000000 was rejected since it
// reduces to the same 32-bit subtract:
// (a ^ 0x80000000) - (b ^ 0x80000000) == a - b (mod 2^32), so it overflows.
template <bool APPROXIMATION_MODE, bool IS_LOWER_BOUND, sfpi::DataLayout LAYOUT, int ITERATIONS = 8>
inline void relu_clamp_uint(uint threshold) {
    static_assert(
        LAYOUT == sfpi::DataLayout::U16 || LAYOUT == sfpi::DataLayout::U32,
        "relu_clamp_uint requires an unsigned DataLayout (U16 or U32 which also covers uint8)");
    const vUInt t = static_cast<unsigned>(threshold);
    const vUInt t_hi = t >> 16;
    const vUInt t_lo = t & 0xFFFF;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt x = dst_reg[0].mode<LAYOUT>();
        vUInt x_hi = x >> 16;
        vUInt x_lo = x & 0xFFFF;
        if constexpr (IS_LOWER_BOUND) {
            // relu_min: max(x, t): take the threshold on lanes where x < t
            v_if((x_hi < t_hi) || ((x_hi == t_hi) && (x_lo < t_lo))) { x = t; }
            v_endif;
        } else {
            // relu_max/relu6: min(x, t): take the threshold on lanes where x > t
            v_if((x_hi > t_hi) || ((x_hi == t_hi) && (x_lo > t_lo))) { x = t; }
            v_endif;
        }
        dst_reg[0].mode<LAYOUT>() = x;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void relu_min(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a < threshold) { a = threshold; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void relu_max(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a > threshold) { a = threshold; }
        v_endif;
        v_if(a < 0.0f) { a = 0.0f; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel
