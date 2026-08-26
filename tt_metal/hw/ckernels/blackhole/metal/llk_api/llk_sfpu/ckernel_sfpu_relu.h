// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_relu.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Full-range unsigned clamp against a threshold, shared by relu_min and relu_max/relu6.
// IS_LOWER_BOUND selects the path:
//   IS_LOWER_BOUND == true -> relu_min: result = max(x, threshold)
//   IS_LOWER_BOUND == false -> relu_max/relu6: result = min(x, threshold)
//
// For U32 we can't do a plain vUInt compare (it subtracts and tests the sign, which overflows for
// operands >= 2^31 apart). Instead we use the hardware sign-magnitude min/max (SFPSWAP).
// When threshold MSB = 0, invert operands (~x, ~t) so that unsigned order maps to sign magnitude
// order, run min/max, then invert the result back.
// When threshold MSB = 1, raw bits already sort correctly under sign-magnitude with min/max roles swapped.
template <bool APPROXIMATION_MODE, bool IS_LOWER_BOUND, DataFormat FORMAT, int ITERATIONS = 8>
inline void relu_clamp_uint(uint threshold) {
    static_assert(
        FORMAT == DataFormat::UInt32 || FORMAT == DataFormat::UInt16,
        "relu_clamp_uint requires DataFormat::UInt32 or UInt16 (uint8 dispatches through UInt32)");
    constexpr sfpi::DataLayout LAYOUT = (FORMAT == DataFormat::UInt16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::U32;
    if constexpr (LAYOUT == sfpi::DataLayout::U32) {
        // relu_min needs max(x, t) and relu_max/relu6 needs min(x, t)
        constexpr bool WANT_MAX = IS_LOWER_BOUND;
        const vUInt t = static_cast<unsigned>(threshold);
        if (static_cast<int>(threshold) >= 0) {
            // threshold MSB = 0: invert operands
            const vSMag nt = sfpi::as<vSMag>(~t);
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vUInt raw = dst_reg[0].mode<LAYOUT>();
                vSMag x = sfpi::as<vSMag>(~raw);
                vUInt r = sfpi::as<vUInt>(WANT_MAX ? sfpi::max(x, nt) : sfpi::min(x, nt));
                dst_reg[0].mode<LAYOUT>() = ~r;  // invert the result back
                dst_reg++;
            }
        } else {
            // threshold MSB = 1: SM min/max applies directly
            const vSMag ts = sfpi::as<vSMag>(t);
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vUInt raw = dst_reg[0].mode<LAYOUT>();
                vSMag x = sfpi::as<vSMag>(raw);
                dst_reg[0].mode<LAYOUT>() = sfpi::as<vUInt>(WANT_MAX ? sfpi::min(x, ts) : sfpi::max(x, ts));
                dst_reg++;
            }
        }
    } else {
        // U16: operands are < 2^16, so the direct vUInt compare cannot overflow.
        const vUInt t = static_cast<unsigned>(threshold);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            vUInt x = dst_reg[0].mode<LAYOUT>();
            if constexpr (IS_LOWER_BOUND) {
                v_if(x < t) { x = t; }
                v_endif;
            } else {
                v_if(x > t) { x = t; }
                v_endif;
            }
            dst_reg[0].mode<LAYOUT>() = x;
            dst_reg++;
        }
    }
}

// Signed int32 clamp against a threshold, shared by relu_min and relu_max/relu6.
// IS_LOWER_BOUND selects the path:
//   IS_LOWER_BOUND == true -> relu_min: result = max(x, threshold)
//   IS_LOWER_BOUND == false -> relu_max/relu6: result = max(0, min(x, threshold))
//
// Load/store as signed 2's complement (DataLayout::I32). Since the sign-magnitude SFPSWAP path cannot
// represent -2^31 (0x80000000 is -0), we use the plain int32 compare.
template <bool APPROXIMATION_MODE, bool IS_LOWER_BOUND, int ITERATIONS = 8>
inline void relu_clamp_int(uint threshold) {
    const vInt t = static_cast<int>(threshold);
    const bool is_pos_threshold = static_cast<int>(threshold) >= 0;
    if constexpr (IS_LOWER_BOUND) {
        // relu_min: max(x, threshold)
        if (is_pos_threshold) {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vInt x = dst_reg[0].mode<sfpi::DataLayout::I32>();
                // v_if(x < 0) is NOT redundant: the SFPU signed compare subtracts and tests the sign,
                // which overflows when x and t are >= 2^31 apart and wrongly gives x>=t. Lifting negatives
                // up to t first leaves only non-negative lanes for the x < t compare, which then cannot overflow.
                v_if(x < 0) { x = t; }
                v_endif;
                v_if(x < t) { x = t; }
                v_endif;
                dst_reg[0].mode<sfpi::DataLayout::I32>() = x;
                dst_reg++;
            }
        } else {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vInt x = dst_reg[0].mode<sfpi::DataLayout::I32>();
                v_if(x < 0) {
                    v_if(x < t) { x = t; }
                    v_endif;
                }
                v_endif;
                dst_reg[0].mode<sfpi::DataLayout::I32>() = x;
                dst_reg++;
            }
        }
    } else {
        // relu_max/relu6: max(0, min(x, threshold))
        if (is_pos_threshold) {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vInt x = dst_reg[0].mode<sfpi::DataLayout::I32>();
                v_if(x < 0) { x = 0; }
                v_endif;
                v_if(x > t) { x = t; }
                v_endif;
                dst_reg[0].mode<sfpi::DataLayout::I32>() = x;
                dst_reg++;
            }
        } else {
            // threshold < 0: max(0, min(x, threshold)) == 0 for every lane
            const vInt zero = 0;
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                dst_reg[0].mode<sfpi::DataLayout::I32>() = zero;
                dst_reg++;
            }
        }
    }
}
inline void relu_min_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

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

inline void relu_max_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

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

inline void lrelu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel
