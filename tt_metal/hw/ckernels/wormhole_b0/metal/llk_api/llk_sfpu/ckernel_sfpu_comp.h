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

// These constants and function should ideally go to SFPI
// Copied from ckernel_sfpu_int_sum.h to avoid dependency complications
#ifndef SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED
#define SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED

#define BIT_MASK_32 0xFFFFFFFF
#define SIGN 0x80000000
#define MAGNITUDE 0x7FFFFFFF

// Convert from sign-magnitude to two's complement format
sfpi_inline vInt sfpu_sign_mag_to_twos_comp(vInt value) {
    v_if(value & SIGN) {
        vInt magnitude = value & MAGNITUDE;
        value = (~magnitude + 1) & BIT_MASK_32;
    }
    v_endif;
    return value;
}

#endif  // SFPU_SIGN_MAG_TO_TWOS_COMP_DEFINED

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
    // Kept as hand-tuned TTI: the equivalent pure-SFPI form (bitwise magnitude + v_if
    // predication) measured ~1.5-2.1x slower in MATH_ISOLATE on Wormhole. The predicates
    // below are integer tests on the bits of the FP32 LREG image that SFPLOAD DEFAULT
    // produces (for a 16-bit Dest, the datum widened to FP32), so no SFPSETSGN is needed:
    //  - eqz/nez: SFPLZ NOSGN|CC_EQ0 sets the flag where (v & 0x7FFFFFFF) == 0, i.e. |v| == 0,
    //    so ±0 compare equal to zero and every NaN compares non-zero, in one instruction.
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
        TTI_SFPLOAD(V, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);

        // eqz: default 0, set 1 where |v| == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPLZ(/*imm12_math*/ 0, V, TMP, sfpi::SFPLZ_MOD1_NOSGN_CC_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // nez: default 1, set 0 where |v| == 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPLZ(/*imm12_math*/ 0, V, TMP, sfpi::SFPLZ_MOD1_NOSGN_CC_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // ltz: default 0; (v - 1 < 0) AND (-inf - v >= 0) → 1
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPIADD(IMM12_MINUS_ONE, V, TMP, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPIADD(0, INF, V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gtz: default 0; (v - 1 >= 0) AND (+inf - v >= 0) → 1
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPIADD(IMM12_MINUS_ONE, V, TMP, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPIADD(0, INF, V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gez: default 1; enable (v >= 0) AND (+inf - v >= 0), invert, drop |V| == 0 → 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPIADD(IMM12_ZERO, V, TMP, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPIADD(0, INF, V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPCOMPC(/*imm12_math*/ 0, /*lreg_c*/ 0, /*lreg_dest*/ 0, /*instr_mod1*/ 0);
            TTI_SFPLZ(/*imm12_math*/ 0, V, TMP, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // lez: default 1; enable (v < 0) AND (-inf - v >= 0), invert, drop |V| == 0 → 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPIADD(IMM12_ZERO, V, TMP, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPIADD(0, INF, V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPCOMPC(/*imm12_math*/ 0, /*lreg_c*/ 0, /*lreg_dest*/ 0, /*instr_mod1*/ 0);
            TTI_SFPLZ(/*imm12_math*/ 0, V, TMP, sfpi::SFPLZ_MOD1_NOSGN_CC_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }
    }
}

// Each mode writes the constant that the untaken side would have produced up front and
// keeps only the taken arm under a v_if. That drops the SFPCOMPC the v_else compiles to,
// and lets the default be materialised once outside the predicate.
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_int() {
    // res is pre-set below, so an unhandled COMP_MODE would silently zero the whole tile
    // instead of falling through to the identity copy the v/v_else form used to give.
    // SfpuType has dozens of enumerators and a mistyped one would otherwise compile clean.
    static_assert(
        (COMP_MODE == SfpuType::equal_zero) or (COMP_MODE == SfpuType::not_equal_zero) or
            (COMP_MODE == SfpuType::less_than_zero) or (COMP_MODE == SfpuType::greater_than_zero) or
            (COMP_MODE == SfpuType::less_than_equal_zero) or (COMP_MODE == SfpuType::greater_than_equal_zero),
        "calculate_comp_int supports only the six comparison-to-zero modes");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt zero = 0;
        vInt res = zero;

        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(v == zero) { res = 1; }
            v_endif;
        }

        // a[i] != 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            res = 1;
            v_if(v == zero) { res = zero; }
            v_endif;
        }

        // a[i] < 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            v_if(v < zero) { res = 1; }
            v_endif;
        }

        // a[i] > 0
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            v_if(v > zero) { res = 1; }
            v_endif;
        }

        // a[i] <= 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            v_if(v <= zero) { res = 1; }
            v_endif;
        }

        // a[i] >= 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            v_if(v >= zero) { res = 1; }
            v_endif;
        }

        dst_reg[0] = res;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_uint16() {
    static_assert((COMP_MODE == SfpuType::equal_zero) or (COMP_MODE == SfpuType::not_equal_zero));
    // UInt16 values live in the low 16 bits of the dest word; DataLayout::U16 loads/stores them
    // directly (SFPLOAD/SFPSTORE mod = UINT16), matching the InstrModLoadStore::LO16 path.
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
        vUInt r = 1;
        v_if(v == 0) { r = 0; }
        v_endif;
        dst_reg[0].mode<sfpi::DataLayout::U32>() = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_int(int scalar) {
    // Convert both operands to two's complement format
    //
    // LOGIC:
    // - Scalar is already in two's complement (from host)
    // - Convert SFPU input data from sign-magnitude to two's complement
    // - Perform comparison with both in two's complement format

    // Scalar stays in original two's complement format
    vInt converted_scalar = scalar;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;

        // Convert input data from sign-magnitude to two's complement
        v = sfpu_sign_mag_to_twos_comp(v);

        // Now both operands are in two's complement format
        // Use simple comparison like Blackhole
        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(v != converted_scalar) { val = 1; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(v == converted_scalar) { val = 1; }
            v_endif;
        }

        dst_reg[0] = val;
        dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_comp_int(sfpi::vInt& val, const sfpi::vInt& v, const int scalar);

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_ne>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    val = 0;
    v_if (v != scalar)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_eq>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    val = 0;
    v_if (v == scalar)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_gt>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    // Unfortunately the compiler doesn't generate correct compares
    // (#14598)
    sfpi::vInt s = scalar;

    val = 0;
    v_if ((v ^ s) >= 0)
    {
        // Same sign, compare is good
        v_if (v > s)
        {
            val = 1;
        }
        v_endif;
    }
    v_elseif (v >= 0)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_lt>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    sfpi::vInt s = scalar;

    val = 0;
    v_if ((v ^ s) >= 0)
    {
        // Same sign, compare is good
        v_if (v < s)
        {
            val = 1;
        }
        v_endif;
    }
    v_elseif (v < 0)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_ge>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    sfpi::vInt s = scalar;

    val = 0;
    v_if ((v ^ s) >= 0)
    {
        // Same sign, compare is good
        v_if (v >= s)
        {
            val = 1;
        }
        v_endif;
    }
    v_elseif (v >= 0)
    {
        val = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_int<SfpuType::unary_le>(sfpi::vInt& val, const sfpi::vInt& v, const int scalar)
{
    sfpi::vInt s = scalar;

    val = 0;
    v_if ((v ^ s) >= 0)
    {
        // Same sign, compare is good
        v_if (v <= s)
        {
            val = 1;
        }
        v_endif;
    }
    v_elseif (v < 0)
    {
        val = 1;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_int_(int scalar)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt v   = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::vInt val = 0;

        apply_unary_comp_int<COMP_MODE>(val, v, scalar);

        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = val;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_comp_float(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s);

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_eq>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v == s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_ne>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 1.0f;
    v_if (v == s)
    {
        val = 0.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_gt>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v > s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_lt>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v < s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_ge>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v >= s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_unary_comp_float<SfpuType::unary_le>(sfpi::vFloat& val, const sfpi::vFloat& v, const sfpi::vFloat& s)
{
    val = 0.0f;
    v_if (v <= s)
    {
        val = 1.0f;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_(std::uint32_t value)
{
    const sfpi::vFloat s = Converter::as_float(value); // value is the threshold's fp32 bit pattern

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat val;

        apply_unary_comp_float<COMP_MODE>(val, v, s);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
