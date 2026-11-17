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

    static_assert(ITERATIONS % 2 == 0, "calculate_unary_ne processes two rows per iteration");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        // Store the default, then overwrite the taken lanes under the predicate: two
        // SFPSTOREs of the constant registers instead of two SFPMOVs and a store. `s == v`
        // (not `v == s`) makes v the subtract's destination, so the compiler does not copy
        // s first; EQ0 of the difference is the same either way. Left inside v_if (not
        // hoisted like gt below) so a literal non-zero s can still lower to SFPIADD.
        sfpi::dst_reg[0] = 1.0f;
        sfpi::dst_reg[1] = 1.0f;
        v_if(s == v0) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        v_if(s == v1) { sfpi::dst_reg[1] = 0.0f; }
        v_endif;
        sfpi::dst_reg += 2;
    }
}

inline void unary_eq_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_eq(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

    static_assert(ITERATIONS % 2 == 0, "calculate_unary_eq processes two rows per iteration");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        // Same two-store form and operand order as unary_ne above.
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg[1] = 0.0f;
        v_if(s == v0) { sfpi::dst_reg[0] = 1.0f; }
        v_endif;
        v_if(s == v1) { sfpi::dst_reg[1] = 1.0f; }
        v_endif;
        sfpi::dst_reg += 2;
    }
}

inline void unary_gt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_gt(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

    static_assert(ITERATIONS % 2 == 0, "calculate_unary_gt processes two rows per iteration");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        // Both differences up front: row 1's subtract fills row 0's MAD latency slot,
        // which an in-condition compare would pad with an SFPNOP. Same operands and
        // order as sfpi's own lowering of `v > s`.
        sfpi::vFloat d0 = s - v0;
        sfpi::vFloat d1 = s - v1;
        // Default stored first, taken arm stored under the predicate (see unary_ne).
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg[1] = 0.0f;
        v_if(d0 < 0.0f) { sfpi::dst_reg[0] = 1.0f; }
        v_endif;
        v_if(d1 < 0.0f) { sfpi::dst_reg[1] = 1.0f; }
        v_endif;
        sfpi::dst_reg += 2;
    }
}

inline void unary_lt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_lt(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

    static_assert(ITERATIONS % 2 == 0, "calculate_unary_lt processes two rows per iteration");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat d0 = v0 - s;
        sfpi::vFloat d1 = v1 - s;
        // Default stored first, taken arm stored under the predicate (see unary_ne).
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg[1] = 0.0f;
        v_if(d0 < 0.0f) { sfpi::dst_reg[0] = 1.0f; }
        v_endif;
        v_if(d1 < 0.0f) { sfpi::dst_reg[1] = 1.0f; }
        v_endif;
        sfpi::dst_reg += 2;
    }
}

inline void unary_ge_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ge(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

    static_assert(ITERATIONS % 2 == 0, "calculate_unary_ge processes two rows per iteration");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat d0 = v0 - s;
        sfpi::vFloat d1 = v1 - s;
        // Safe to recast onto GTE0 here, unlike le below: ge is the exact complement of
        // the baseline's `v < s` on the *same* difference v - s, so LT0 and GTE0 partition
        // every operand pair including the zeros and the inf - inf NaN. Only le would have
        // had to flip the operand order to reach GTE0, which is why it keeps its compare.
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg[1] = 0.0f;
        v_if(d0 >= 0.0f) { sfpi::dst_reg[0] = 1.0f; }
        v_endif;
        v_if(d1 >= 0.0f) { sfpi::dst_reg[1] = 1.0f; }
        v_endif;
        sfpi::dst_reg += 2;
    }
}

inline void unary_le_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_le(std::uint32_t value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

    static_assert(ITERATIONS % 2 == 0, "calculate_unary_le processes two rows per iteration");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat d0 = s - v0;
        sfpi::vFloat d1 = s - v1;
        // Deliberately NOT recast onto GTE0 as `s - v >= 0.0f`, the way ge is above:
        // s - v is the exact negation of v - s only for finite unequal operands. With
        // v = +0.0, s = -0.0 the baseline's v - s is +0.0 (not greater, so le returns 1,
        // matching torch) while s - v is -0.0, and GTE0 is a sign-bit test, so it would
        // return 0. v == s == +/-inf differs the same way: both orders produce inf - inf
        // and the two forms read that NaN's sign with opposite polarity. s is an
        // unvalidated user scalar from ttnn.le, so -0.0 and +/-inf are reachable.
        // Keeping the baseline compare and dropping only the v_else saves the same
        // SFPCOMPC and is exact by construction.
        sfpi::dst_reg[0] = 1.0f;
        sfpi::dst_reg[1] = 1.0f;
        v_if(d0 < 0.0f) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        v_if(d1 < 0.0f) { sfpi::dst_reg[1] = 0.0f; }
        v_endif;
        sfpi::dst_reg += 2;
    }
}

}  // namespace sfpu
}  // namespace ckernel
