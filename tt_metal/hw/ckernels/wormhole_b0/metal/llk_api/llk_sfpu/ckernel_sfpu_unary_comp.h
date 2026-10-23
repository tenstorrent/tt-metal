// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

inline void unary_ne_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ne(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r = 1.0f;
        v_if(v == s) { r = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = r;

        sfpi::dst_reg++;
    }
}

inline void unary_eq_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_eq(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r = 0.0f;
        v_if(v == s) { r = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = r;

        sfpi::dst_reg++;
    }
}

inline void unary_gt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_gt(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r = 0.0f;
        v_if(v > s) { r = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = r;

        sfpi::dst_reg++;
    }
}

inline void unary_lt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_lt(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r = 0.0f;
        v_if(v < s) { r = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = r;

        sfpi::dst_reg++;
    }
}

inline void unary_ge_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ge(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // Safe to recast onto GTE0 here, unlike le below: ge is the exact complement of
        // the baseline's `v < s` on the *same* difference v - s, so LT0 and GTE0 partition
        // every operand pair including the zeros and the inf - inf NaN. Only le would have
        // had to flip the operand order to reach GTE0, which is why it keeps its compare.
        sfpi::vFloat r = 0.0f;
        v_if(v - s >= 0.0f) { r = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = r;

        sfpi::dst_reg++;
    }
}

inline void unary_le_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_le(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // Deliberately NOT recast onto GTE0 as `s - v >= 0.0f`, the way ge is above:
        // s - v is the exact negation of v - s only for finite unequal operands. With
        // v = +0.0, s = -0.0 the baseline's v - s is +0.0 (not greater, so le returns 1,
        // matching torch) while s - v is -0.0, and GTE0 is a sign-bit test, so it would
        // return 0. v == s == +/-inf differs the same way: both orders produce inf - inf
        // and the two forms read that NaN's sign with opposite polarity. s is an
        // unvalidated user scalar from ttnn.le, so -0.0 and +/-inf are reachable.
        // Keeping the baseline compare and dropping only the v_else saves the same
        // SFPCOMPC and is exact by construction.
        sfpi::vFloat r = 1.0f;
        v_if(v > s) { r = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = r;

        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
