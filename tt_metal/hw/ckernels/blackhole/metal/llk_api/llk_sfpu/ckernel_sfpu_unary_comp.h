// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// The six comparisons against a scalar, on the comparison instructions rather
// than on a subtract.
//
// `v_if(v == s)` is a subtract and an SFPSETCC on the difference, and SFPSETCC
// reads its register as an int32: -0.0 is negative and non-zero to it. On top of
// that, inf - inf is NaN, so two equal infinities never compare equal and their
// ordering is the sign of that NaN, which is not the same on Blackhole and
// Wormhole; and a NaN operand subtracts to a NaN that reads as positive, so
// NaN > x answers true.
//
// SFPGT and SFPLE compare properly: sign-magnitude is handled, ±0 compare equal,
// and the only departure from IEEE is that a NaN is ordered by its sign rather
// than unordered — which one `abs(v) + abs(s) <= inf` test removes. This is the
// treatment ckernel_sfpu_binary_comp.h already gives the two-tensor forms, and
// the answers now agree with it, subnormals included: the SFPU add flushes them,
// so `abs(v) + abs(s) == 0` reads them as zero, deliberately and as before.
//
// The scalar and everything derived from it are loop invariant, so the only
// per-element cost over the subtract is the sign and the sum.

// v == s (IS_EQUAL) or v != s.
template <int ITERATIONS, bool IS_EQUAL>
inline void _calculate_unary_comp_equal_(uint value) {
    const sfpi::vFloat s = Converter::as_float(value);
    const sfpi::vFloat abs_s = sfpi::setsgn(s, 0);
    const sfpi::vInt inf = 0x7f800000;
    constexpr float equal_result = IS_EQUAL ? 1.0f : 0.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = IS_EQUAL ? 0.0f : 1.0f;
        const sfpi::vFloat sum = sfpi::setsgn(v, 0) + abs_s;

        // treats every ±subnormal as equal to zero
        v_if(sum == 0.0f) { sfpi::dst_reg[0] = equal_result; }
        v_endif;
        // v <= s and s <= v, and abs(v) + abs(s) <= inf rejects NaN
        v_if(v <= s && s <= v && sfpi::as<sfpi::vInt>(sum) <= inf) { sfpi::dst_reg[0].mode(ADDR_MOD_6) = equal_result; }
        v_endif;
    }
}

// v > s (IS_GREATER) or v < s.
template <int ITERATIONS, bool IS_GREATER>
inline void _calculate_unary_comp_strict_(uint value) {
    const sfpi::vFloat s = Converter::as_float(value);
    const sfpi::vFloat abs_s = sfpi::setsgn(s, 0);
    const sfpi::vInt inf = 0x7f800000;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;
        const sfpi::vFloat sum = sfpi::setsgn(v, 0) + abs_s;

        // abs(v) + abs(s) != 0 rejects both zero or ±subnormal, <= inf rejects NaN
        v_if((IS_GREATER ? v > s : v < s) && sum != 0.0f && sfpi::as<sfpi::vInt>(sum) <= inf) {
            sfpi::dst_reg[0].mode(ADDR_MOD_6) = 1.0f;
        }
        v_endif;
    }
}

// v >= s (IS_GREATER) or v <= s.
template <int ITERATIONS, bool IS_GREATER>
inline void _calculate_unary_comp_weak_(uint value) {
    const sfpi::vFloat s = Converter::as_float(value);
    const sfpi::vFloat abs_s = sfpi::setsgn(s, 0);
    const sfpi::vInt inf = 0x7f800000;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 1.0f;
        const sfpi::vFloat sum = sfpi::setsgn(v, 0) + abs_s;

        // the strict comparison the other way: v < s for >=, v > s for <=;
        // abs(v) + abs(s) != 0 keeps every ±subnormal equal to zero
        v_if((IS_GREATER ? v < s : v > s) && sum != 0.0f) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        // abs(v) + abs(s) > inf: v or s is NaN
        v_if(sfpi::as<sfpi::vInt>(sum) > inf) { sfpi::dst_reg[0].mode(ADDR_MOD_6) = 0.0f; }
        v_endif;
    }
}

inline void unary_ne_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ne(uint value) {
    _calculate_unary_comp_equal_<ITERATIONS, /*IS_EQUAL=*/false>(value);
}

inline void unary_eq_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_eq(uint value) {
    _calculate_unary_comp_equal_<ITERATIONS, /*IS_EQUAL=*/true>(value);
}

inline void unary_gt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_gt(uint value) {
    _calculate_unary_comp_strict_<ITERATIONS, /*IS_GREATER=*/true>(value);
}

inline void unary_lt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_lt(uint value) {
    _calculate_unary_comp_strict_<ITERATIONS, /*IS_GREATER=*/false>(value);
}

inline void unary_ge_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ge(uint value) {
    _calculate_unary_comp_weak_<ITERATIONS, /*IS_GREATER=*/true>(value);
}

inline void unary_le_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_le(uint value) {
    _calculate_unary_comp_weak_<ITERATIONS, /*IS_GREATER=*/false>(value);
}

}  // namespace sfpu
}  // namespace ckernel
