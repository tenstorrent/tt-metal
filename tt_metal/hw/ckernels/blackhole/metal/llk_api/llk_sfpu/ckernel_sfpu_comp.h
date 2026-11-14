// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "ckernel_sfpu_is_fp16_zero.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_sfpu_types.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void greater_than_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void greater_than_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void less_than_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void less_than_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void not_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp() {
    // Raw TTI because sfpi cannot express this sequence: lz() hardwires SFPLZ to CC_NONE, so it
    // cannot set the predicate; SFPCOMPC is only emitted as part of v_else; and the integer view
    // of a DEFAULT-mode load needs an impl_ pun (.mode<I32>() turns the load into a 32-bit Dest
    // read, which differs on a 16-bit Dest). The predicates below are integer tests on the bits
    // of the FP32 LREG image that SFPLOAD DEFAULT produces (for a 16-bit Dest, the datum widened
    // to FP32), so no SFPSETSGN is needed:
    //  - eqz/nez: SFPLZ NOSGN|CC_NE0 enables the lanes where (v & 0x7FFFFFFF) != 0, i.e. |v| != 0,
    //    so ±0 compare equal to zero and every NaN compares non-zero, in one instruction. The
    //    zero answer is the default store, the non-zero answer goes to the enabled lanes (not
    //    NOSGN|CC_EQ0, which sets the CC_COMP bit: ttsim rejects that mode).
    //  - ltz/gtz: two SFPIADD half-plane tests on the bits viewed as a signed int32.
    //    ltz <=> (v - 1 < 0) AND (-inf - v >= 0): the first test keeps [INT_MIN+1, 0] (drops
    //    -0 = INT_MIN, keeps +0), the second keeps v <= 0xFF800000, so the intersection is
    //    {-tiny .. -inf}: ±0, -NaN (0xFF800001..) and every positive pattern are rejected.
    //    gtz <=> (v - 1 >= 0) AND (+inf - v >= 0): the first test keeps [1, INT_MAX] and
    //    INT_MIN (wrap), the second keeps v <= 0x7F800000 and rejects INT_MIN (the subtraction
    //    wraps negative), so the result is {+tiny .. +inf}: ±0, ±NaN and negatives rejected.
    //    SFPIADD only writes the flag of enabled lanes, so the two tests AND together.
    //  - gez/lez: the 1-lanes are two bit intervals (gez: [+0, +inf] plus -0; lez: [-0, -inf]
    //    plus +0), so one chain cannot select them directly. Default 1. A sign test (v + 0)
    //    and (±inf - v >= 0) enable the closed half that holds the answer's infinity, and the
    //    second SFPIADD writes ±inf - v into V on the lanes of the right sign. SFPCOMPC (empty
    //    flag stack: inverts against all lanes) leaves the 0-lanes plus the other zero. SFPLZ
    //    NOSGN|CC_NE0 on V then drops that zero (V still holds it, |v| == 0), while a NaN of
    //    the answer's sign now holds ±inf - v, which is non-zero; store 0 on what is left.
    // The predicated store advances Dest through ADDR_MOD_6 (dest.incr = 2, programmed by the
    // op's init), so no TTINCRWC is needed.
    constexpr std::uint32_t V = p_sfpu::LREG0;
    constexpr std::uint32_t TMP = p_sfpu::LREG2;
    constexpr std::uint32_t INF = p_sfpu::LREG5;
    constexpr std::uint32_t BFLOAT16_INF = 0x7f80;
    constexpr std::uint32_t BFLOAT16_NEG_INF = 0xff80;
    constexpr std::uint32_t IMM12_ZERO = 0x000;
    constexpr std::uint32_t IMM12_MINUS_ONE = 0xfff;

    if constexpr (COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::less_than_equal_zero) {
        TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, BFLOAT16_NEG_INF);
    } else if constexpr (COMP_MODE == SfpuType::greater_than_zero || COMP_MODE == SfpuType::greater_than_equal_zero) {
        TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, BFLOAT16_INF);
    }

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(V, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

        // eqz: default 1, set 0 where |v| != 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPLZ(0 /*imm12_math*/, V, TMP, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // nez: default 0, set 1 where |v| != 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPLZ(0 /*imm12_math*/, V, TMP, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // ltz: default 0; (v - 1 < 0) AND (-inf - v >= 0) → 1
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPIADD(IMM12_MINUS_ONE, V, TMP, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPIADD(0, INF, V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gtz: default 0; (v - 1 >= 0) AND (+inf - v >= 0) → 1
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPIADD(IMM12_MINUS_ONE, V, TMP, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPIADD(0, INF, V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gez: default 1; enable (v >= 0) AND (+inf - v >= 0), invert, drop |V| == 0 → 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPIADD(IMM12_ZERO, V, TMP, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPIADD(0, INF, V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPCOMPC(0 /*imm12_math*/, 0 /*lreg_c*/, 0 /*lreg_dest*/, 0 /*instr_mod1*/);
            TTI_SFPLZ(0 /*imm12_math*/, V, TMP, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // lez: default 1; enable (v < 0) AND (-inf - v >= 0), invert, drop |V| == 0 → 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPIADD(IMM12_ZERO, V, TMP, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPIADD(0, INF, V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPCOMPC(0 /*imm12_math*/, 0 /*lreg_c*/, 0 /*lreg_dest*/, 0 /*instr_mod1*/);
            TTI_SFPLZ(0 /*imm12_math*/, V, TMP, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_int() {
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt zero = 0;

        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(v == zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] != 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            v_if(v == zero) { v = zero; }
            v_else { v = 1; }
            v_endif;
        }

        // a[i] < 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            v_if(v < zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] > 0
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            v_if(v > zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] <= 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            v_if(v <= zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] >= 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            v_if(v >= zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_uint16() {
    static_assert((COMP_MODE == SfpuType::equal_zero) or (COMP_MODE == SfpuType::not_equal_zero));
    // UInt16 values live in the low 16 bits of the dest word; DataLayout::U16 loads/stores them
    // directly (SFPLOAD/SFPSTORE mod = UINT16), matching the InstrModLoadStore::UINT16 path.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U16>();
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            vUInt r = 0;
            v_if(v == 0) { r = 1; }
            v_endif;
            dst_reg[0].mode<sfpi::DataLayout::U16>() = r;
        } else {
            vUInt r = 1;
            v_if(v == 0) { r = 0; }
            v_endif;
            dst_reg[0].mode<sfpi::DataLayout::U16>() = r;
        }
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_eqz_uint32() {
    // UInt32 values occupy the full dest word; DataLayout::U32 loads/stores them
    // directly (SFPLOAD/SFPSTORE mod = UINT32). eqz/nez are representation-agnostic
    // (only a compare against the all-zero word), so a plain unsigned compare works.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(v == 0) { r = 1; }
        v_endif;
        dst_reg[0].mode<sfpi::DataLayout::U32>() = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_nez_uint32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(v != 0) { r = 1; }
        v_endif;
        dst_reg[0].mode<sfpi::DataLayout::U32>() = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_int(int scalar) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;

        // a[i] != scalar
        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(v != scalar) { val = 1; }
            v_endif;
        }
        // a[i] == scalar
        else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(v == scalar) { val = 1; }
            v_endif;
        }
        dst_reg[0] = val;
        dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_int_comp(sfpi::vInt& v, int scalar, sfpi::vInt& out_val);

// a[i] != scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_ne>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val) {
    v_if(v != scalar) { out_val = 1; }
    v_endif;
}

// a[i] == scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_eq>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val) {
    v_if(v == scalar) { out_val = 1; }
    v_endif;
}

// a[i] > scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_gt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val) {
    const sfpi::vInt s = scalar;
    v_if(v >= 0 && s < 0) { out_val = 1; }
    v_elseif(v < 0 && s >= 0) { out_val = 0; }
    v_elseif(v > s) { out_val = 1; }
    v_endif;
}

// a[i] < scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_lt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val) {
    const sfpi::vInt s = scalar;
    v_if(v >= 0 && s < 0) { out_val = 0; }
    v_elseif(v < 0 && s >= 0) { out_val = 1; }
    v_elseif(v < s) { out_val = 1; }
    v_endif;
}

// a[i] >= scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_ge>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val) {
    const sfpi::vInt s = scalar;
    v_if(v >= 0 && s < 0) { out_val = 1; }
    v_elseif(v < 0 && s >= 0) { out_val = 0; }
    v_elseif(v >= s) { out_val = 1; }
    v_endif;
}

// a[i] <= scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_le>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val) {
    const sfpi::vInt s = scalar;
    v_if(v < 0 && s >= 0) { out_val = 1; }
    v_elseif(v >= 0 && s < 0) { out_val = 0; }
    v_elseif(v <= s) { out_val = 1; }
    v_else { out_val = 0; }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_int_(int scalar) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt v = sfpi::dst_reg[0];
        sfpi::vInt val = 0;

        apply_unary_int_comp<COMP_MODE>(v, scalar, val);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_float_comp(sfpi::vFloat v, sfpi::vFloat scalar, sfpi::vFloat& out_val);

// a[i] == scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_eq>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val) {
    v_if(v == s) { out_val = 1.0f; }
    v_else { out_val = 0.0f; }
    v_endif;
}

// a[i] != scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_ne>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val) {
    v_if(v == s) { out_val = 0.0f; }
    v_else { out_val = 1.0f; }
    v_endif;
}

// a[i] > scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_gt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val) {
    v_if(v > s) { out_val = 1.0f; }
    v_else { out_val = 0.0f; }
    v_endif;
}

// a[i] < scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_lt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val) {
    v_if(v < s) { out_val = 1.0f; }
    v_else { out_val = 0.0f; }
    v_endif;
}

// a[i] >= scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_ge>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val) {
    v_if(v >= s) { out_val = 1.0f; }
    v_else { out_val = 0.0f; }
    v_endif;
}

// a[i] <= scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_le>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val) {
    v_if(v <= s) { out_val = 1.0f; }
    v_else { out_val = 0.0f; }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_(std::uint32_t value) {
    const sfpi::vFloat s = Converter::as_float(value);  // value is the threshold's fp32 bit pattern

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat val = 0.0f;

        apply_unary_float_comp<COMP_MODE>(v, s, val);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
