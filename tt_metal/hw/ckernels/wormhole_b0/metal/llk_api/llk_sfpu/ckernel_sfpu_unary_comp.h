// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
// Wormhole has no SFPGT or SFPLE, so the ordering comes from SFPSWAP, whose
// min/max is the same sign-magnitude comparison: swap a copy of the value
// against the scalar and XOR it back to see whether it moved. ±0 compare equal,
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
inline void _calculate_unary_comp_equal_(std::uint32_t value) {
    const sfpi::vFloat s = Converter::as_float(value);
    const sfpi::vFloat abs_s = sfpi::setsgn(s, 0);
    const sfpi::vInt inf = 0x7f800000;
    constexpr float equal_result = IS_EQUAL ? 1.0f : 0.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = IS_EQUAL ? 0.0f : 1.0f;
        const sfpi::vFloat sum = sfpi::setsgn(v, 0) + abs_s;
        const sfpi::vInt diff = sfpi::as<sfpi::vInt>(v) ^ sfpi::as<sfpi::vInt>(s);

        // treats every ±subnormal as equal to zero
        v_if(sum == 0.0f) { sfpi::dst_reg[0] = equal_result; }
        v_endif;
        // abs(v) + abs(s) <= inf rejects NaN, then the two are bitwise identical
        v_if(sfpi::as<sfpi::vInt>(sum) <= inf && diff == 0) { sfpi::dst_reg[0].mode(ADDR_MOD_2) = equal_result; }
        v_endif;
    }
}

// v > s (IS_GREATER) or v < s.
template <int ITERATIONS, bool IS_GREATER>
inline void _calculate_unary_comp_strict_(std::uint32_t value) {
    const sfpi::vFloat s = Converter::as_float(value);
    const sfpi::vFloat abs_s = sfpi::setsgn(s, 0);
    const sfpi::vInt inf = 0x7f800000;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;
        const sfpi::vFloat sum = sfpi::setsgn(v, 0) + abs_s;

        // v > s when v is not the minimum, v < s when it is not the maximum
        const sfpi::vFloat bound = IS_GREATER ? sfpi::min(v, s) : sfpi::max(v, s);
        // abs(v) + abs(s) != 0 rejects both zero or ±subnormal, <= inf rejects NaN
        v_if(
            (sfpi::as<sfpi::vInt>(bound) ^ sfpi::as<sfpi::vInt>(v)) != 0 && sum != 0.0f &&
            sfpi::as<sfpi::vInt>(sum) <= inf) {
            sfpi::dst_reg[0].mode(ADDR_MOD_2) = 1.0f;
        }
        v_endif;
    }
}

// v >= s (IS_GREATER) or v <= s.
template <int ITERATIONS, bool IS_GREATER>
inline void _calculate_unary_comp_weak_(std::uint32_t value) {
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
        const sfpi::vFloat bound = IS_GREATER ? sfpi::max(v, s) : sfpi::min(v, s);
        v_if((sfpi::as<sfpi::vInt>(bound) ^ sfpi::as<sfpi::vInt>(v)) != 0 && sum != 0.0f) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        // abs(v) + abs(s) > inf: v or s is NaN
        v_if(sfpi::as<sfpi::vInt>(sum) > inf) { sfpi::dst_reg[0].mode(ADDR_MOD_2) = 0.0f; }
        v_endif;
    }
}

inline void unary_ne_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ne(std::uint32_t value) {
    _calculate_unary_comp_equal_<ITERATIONS, /*IS_EQUAL=*/false>(value);
}

inline void unary_eq_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_eq(std::uint32_t value) {
    _calculate_unary_comp_equal_<ITERATIONS, /*IS_EQUAL=*/true>(value);
}

inline void unary_gt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_gt(std::uint32_t value) {
    _calculate_unary_comp_strict_<ITERATIONS, /*IS_GREATER=*/true>(value);
}

inline void unary_lt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_lt(std::uint32_t value) {
    _calculate_unary_comp_strict_<ITERATIONS, /*IS_GREATER=*/false>(value);
}

inline void unary_ge_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ge(std::uint32_t value) {
    _calculate_unary_comp_weak_<ITERATIONS, /*IS_GREATER=*/true>(value);
}

inline void unary_le_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_le(std::uint32_t value) {
    _calculate_unary_comp_weak_<ITERATIONS, /*IS_GREATER=*/false>(value);
}

}  // namespace sfpu
}  // namespace ckernel
