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
    // Kept as hand-tuned TTI: the equivalent pure-SFPI form (bitwise magnitude + v_if
    // predication) measured ~1.5-2.1x slower in MATH_ISOLATE on Wormhole, because this
    // sequence gets abs in one SFPSETSGN, rejects NaN with a single SFPIADD, and stores
    // LCONST_0/1 directly under SFPSETCC instead of materialising 0/1 in a register.
    constexpr std::uint32_t V = p_sfpu::LREG0;
    constexpr std::uint32_t ABS_V = p_sfpu::LREG2;
    constexpr std::uint32_t INF = p_sfpu::LREG5;
    constexpr std::uint32_t BFLOAT16_INF = 0x7f80;

    if constexpr (
        COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_equal_zero ||
        COMP_MODE == SfpuType::greater_than_zero || COMP_MODE == SfpuType::less_than_equal_zero) {
        TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, BFLOAT16_INF);
    }

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(V, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPSETSGN(0, V, ABS_V, 1);

        // eqz: default 0, set 1 where |v| == 0 (handles ±0; NaN has |v|!=0 → stays 0)
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // nez: default 1, set 0 where |v| == 0 (handles ±0; NaN has |v|!=0 → stays 1)
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // ltz: default 0; chain: (v < 0) AND (|v| != 0) AND (|v| <= inf) → 1; NaN: |NaN| > inf → rejected
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gtz: default 0; chain: (v >= 0) AND (|v| != 0) AND (|v| <= inf) → 1; NaN: |NaN| > inf → rejected
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gez: default 1; chain1: (v<0) AND (|v|!=0) → 0 (negatives excl. -0); chain2: |v|>inf → 0 (NaN)
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // lez: default 1; chain1: (v>=0) AND (|v|!=0) → 0 (positives excl. +0); chain2: |v|>inf → 0 (NaN)
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
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
        // Int32 tiles sit in Dest as two's complement, so read the word as it is (I32 layout,
        // SFPLOAD mod0 INT32). The Wormhole Default vInt layout is SM32 (SFPLOAD mod0 INT32_SM),
        // whose sign-magnitude -> two's complement conversion has no encoding for -2^31 and
        // turns it into 0, which would make eqz/nez/ltz/gez misclassify INT32_MIN.
        //
        // Every compare below is against the immediate 0, which SFPI lowers to SFPSETCC on the
        // word itself (sign bit / zero test); `v >= 0 && v != 0` is two chained SFPSETCCs.
        // `v > 0` and `v <= 0` must not be used: SFPI lowers them through an integer
        // subtraction (SFPIADD 0 - v) that overflows for INT32_MIN and classifies it as positive.
        vInt v = dst_reg[0].mode<sfpi::DataLayout::I32>();
        vInt res = 0;

        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(v == 0) { res = 1; }
            v_endif;
        }

        // a[i] != 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            res = 1;
            v_if(v == 0) { res = 0; }
            v_endif;
        }

        // a[i] < 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            v_if(v < 0) { res = 1; }
            v_endif;
        }

        // a[i] > 0
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            v_if(v >= 0 && v != 0) { res = 1; }
            v_endif;
        }

        // a[i] <= 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            res = 1;
            v_if(v >= 0 && v != 0) { res = 0; }
            v_endif;
        }

        // a[i] >= 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            v_if(v >= 0) { res = 1; }
            v_endif;
        }

        dst_reg[0].mode<sfpi::DataLayout::I32>() = res;
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
    // eq/ne is equality of the raw 32-bit word (int32 or uint32 bit pattern; ttnn routes both
    // dtypes here) against the scalar the host passes as the same raw word, so read the Dest word
    // as it is (I32 layout, SFPLOAD mod0 INT32) and compare. The Wormhole Default vInt layout is
    // SM32 (SFPLOAD mod0 INT32_SM), whose sign-magnitude -> two's complement conversion has no
    // encoding for the word 0x80000000 (INT32_MIN, or the uint32 2^31) and turns it into 0; a
    // software re-conversion of that word cannot bring it back, so eq/ne misclassified it.
    vInt scalar_vec = scalar;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0].mode<sfpi::DataLayout::I32>();
        vInt val = 0;

        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(v != scalar_vec) { val = 1; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(v == scalar_vec) { val = 1; }
            v_endif;
        }

        dst_reg[0].mode<sfpi::DataLayout::I32>() = val;
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
