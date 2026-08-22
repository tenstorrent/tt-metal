// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
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
    constexpr uint v = p_sfpu::LREG0;
    constexpr uint s = p_sfpu::LREG1;
    constexpr uint abs_v = p_sfpu::LREG2;
    constexpr uint abs_s = p_sfpu::LREG3;
    constexpr uint sum = p_sfpu::LREG4;
    constexpr uint inf = p_sfpu::LREG5;
    constexpr uint unequal_result = IS_EQUAL ? p_sfpu::LCONST_0 : p_sfpu::LCONST_1;
    constexpr uint equal_result = IS_EQUAL ? p_sfpu::LCONST_1 : p_sfpu::LCONST_0;

    TT_SFPLOADI(s, sfpi::SFPLOADI_MOD0_UPPER, (value >> 16) & 0xFFFF);
    TT_SFPLOADI(s, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
    TTI_SFPSETSGN(0, s, abs_s, 1);  // SFPSETSGN_MOD1_ARG_IMM
    TTI_SFPLOADI(inf, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(v, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSTORE(unequal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

        TTI_SFPSETSGN(0, v, abs_v, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPMAD(p_sfpu::LCONST_1, abs_v, abs_s, sum, 0);

        // if abs(v) + abs(s) == 0; treats every ±subnormal as equal to zero
        TTI_SFPSETCC(0, sum, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSTORE(equal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        // if v <= s and s <= v
        TTI_SFPLE(0, s, v, 1);  // SFPLE_MOD1_SET_CC
        TTI_SFPLE(0, v, s, 1);  // SFPLE_MOD1_SET_CC
        // if abs(v) + abs(s) <= inf; rejects NaN
        TTI_SFPIADD(0, inf, sum, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
        TTI_SFPSTORE(equal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

// v > s (IS_GREATER) or v < s.
template <int ITERATIONS, bool IS_GREATER>
inline void _calculate_unary_comp_strict_(uint value) {
    constexpr uint v = p_sfpu::LREG0;
    constexpr uint s = p_sfpu::LREG1;
    constexpr uint abs_v = p_sfpu::LREG2;
    constexpr uint abs_s = p_sfpu::LREG3;
    constexpr uint sum = p_sfpu::LREG4;
    constexpr uint inf = p_sfpu::LREG5;

    TT_SFPLOADI(s, sfpi::SFPLOADI_MOD0_UPPER, (value >> 16) & 0xFFFF);
    TT_SFPLOADI(s, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
    TTI_SFPSETSGN(0, s, abs_s, 1);  // SFPSETSGN_MOD1_ARG_IMM
    TTI_SFPLOADI(inf, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(v, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

        TTI_SFPSETSGN(0, v, abs_v, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPMAD(p_sfpu::LCONST_1, abs_v, abs_s, sum, 0);

        // if v > s, or if v < s
        if constexpr (IS_GREATER) {
            TTI_SFPGT(0, s, v, 1);  // SFPGT_MOD1_SET_CC
        } else {
            TTI_SFPGT(0, v, s, 1);  // SFPGT_MOD1_SET_CC
        }
        // if abs(v) + abs(s) != 0; rejects if both are zero or ±subnormal
        TTI_SFPSETCC(0, sum, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        // if abs(v) + abs(s) <= inf; rejects NaN
        TTI_SFPIADD(0, inf, sum, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
        TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

// v >= s (IS_GREATER) or v <= s.
template <int ITERATIONS, bool IS_GREATER>
inline void _calculate_unary_comp_weak_(uint value) {
    constexpr uint v = p_sfpu::LREG0;
    constexpr uint s = p_sfpu::LREG1;
    constexpr uint abs_v = p_sfpu::LREG2;
    constexpr uint abs_s = p_sfpu::LREG3;
    constexpr uint sum = p_sfpu::LREG4;
    constexpr uint inf = p_sfpu::LREG5;

    TT_SFPLOADI(s, sfpi::SFPLOADI_MOD0_UPPER, (value >> 16) & 0xFFFF);
    TT_SFPLOADI(s, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
    TTI_SFPSETSGN(0, s, abs_s, 1);  // SFPSETSGN_MOD1_ARG_IMM
    TTI_SFPLOADI(inf, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(v, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

        TTI_SFPSETSGN(0, v, abs_v, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPMAD(p_sfpu::LCONST_1, abs_v, abs_s, sum, 0);

        // if the strict comparison the other way holds: v < s for >=, v > s for <=
        if constexpr (IS_GREATER) {
            TTI_SFPGT(0, v, s, 1);  // SFPGT_MOD1_SET_CC
        } else {
            TTI_SFPGT(0, s, v, 1);  // SFPGT_MOD1_SET_CC
        }
        // if abs(v) + abs(s) != 0; every ±subnormal stays equal to zero
        TTI_SFPSETCC(0, sum, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        // if abs(v) + abs(s) > inf; v or s is NaN
        TTI_SFPIADD(0, inf, sum, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
        TTI_SFPENCC(0, 0, 0, 0);
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
