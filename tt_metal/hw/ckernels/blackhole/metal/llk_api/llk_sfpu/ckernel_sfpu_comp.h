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
        TTI_SFPLOAD(V, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSETSGN(0, V, ABS_V, 1);

        // eqz: default 0, set 1 where |v| == 0 (handles ±0; NaN has |v|!=0 → stays 0)
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // nez: default 1, set 0 where |v| == 0 (handles ±0; NaN has |v|!=0 → stays 1)
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // ltz: default 0; chain: (v < 0) AND (|v| != 0) AND (|v| <= inf) → 1
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gtz: default 0; chain: (v >= 0) AND (|v| != 0) AND (|v| <= inf) → 1
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gez: default 1; chain1: (v<0) AND (|v|!=0) → 0 (negatives excl. -0); chain2: |v|>inf → 0 (NaN)
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // lez: default 1; chain1: (v>=0) AND (|v|!=0) → 0 (positives excl. +0); chain2: |v|>inf → 0 (NaN)
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
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

sfpi_inline void _calculate_comp_init_flag_(bool check, sfpi::vFloat& flag1, sfpi::vFloat& flag2, float init)
{
    flag1 = init;
    if (check)
    {
        flag2 = init;
    }
}

template <bool APPROXIMATION_MODE, bool invert_output, bool check_zero, bool second_check, bool is_less_than_equal_zero, int ITERATIONS>
sfpi_inline void _calculate_comp_(const int iterations, std::uint32_t exponent_size_8)
{
    // output_0 and output_1 hold the outputs use use when a zero or negative check is true/false.
    // False = 0.0 = kCONST_0 (5/8-bit exponent format)
    // True  = 1.0 = kCONST_1_FP16B (8-bit exponent format)
    // SFPU uses 8-bit exponent in operations so loading these constants in 8-bit exponent format.
    // Although a command flag can tell SFPU to re-bias a 5-bit exponent to 8-bit, we are loading 8-bit
    // exponent and telling SFPU to not add any bias to these constants.
    constexpr float output_0 = invert_output ? 0.0f : 1.0f;
    constexpr float output_1 = invert_output ? 1.0f : 0.0f;

    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat flag1, flag2;
        if constexpr (check_zero)
        {
            v_if (_sfpu_is_fp16_zero_(v))
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_0);
            }
            v_else
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }
        else
        {
            v_if (v < 0.0F)
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_0);
            }
            v_else
            {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }

        sfpi::vFloat result;
        if constexpr (second_check)
        {
            // less_than_equal_zero
            // flag1 = 0x3F80(1.0) if DST < 0 else 0
            // flag2 = 0x3F80(1.0) if DST == 0 else 0
            // Do a bitwise Or (flag1 | flag2) to get <= condition.
            // flag1 < 0 OR flag2 == 0 => DST is Less than or Equal to zero.
            // Result will be either 0x0000(0.0) or 0x3F80(1.0)
            if constexpr (is_less_than_equal_zero)
            {
                result = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(flag1) | sfpi::as<sfpi::vUInt>(flag2));
            }
            else
            {
                // greater_than_zero
                // flag1 = 0x3F80(1.0) if DST >= 0 else 0
                // flag2 = 0x3F80(1.0) if DST != 0 else 0
                // Do a bitwise And (flag1 & flag2) to get > condition.
                // flag2 >= 0 AND flag1 != 0 => DST is Greater than zero
                // Result will be either 0x0000(0.0) or 0x3F80(1.0)
                result = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(flag1) & sfpi::as<sfpi::vUInt>(flag2));
            }
        }
        else
        {
            result = flag1;
        }

        sfpi::dst_reg[0] = result;

        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_zero_comp(sfpi::vFloat& v, std::uint32_t exponent_size_8);

template <>
sfpi_inline void apply_zero_comp<SfpuType::equal_zero>(sfpi::vFloat& v, std::uint32_t)
{
    v_if (_sfpu_is_fp16_zero_(v))
    {
        v = 1.0f;
    }
    v_else
    {
        v = 0.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::not_equal_zero>(sfpi::vFloat& v, std::uint32_t)
{
    v_if (_sfpu_is_fp16_zero_(v))
    {
        v = 0.0f;
    }
    v_else
    {
        v = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::less_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= 0.0f)
    {
        v = 0.0f;
    }
    v_else
    {
        v = 1.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::greater_than_equal_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= 0.0f)
    {
        v = 1.0f;
    }
    v_else
    {
        v = 0.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::greater_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > 0.0f)
    {
        v = 1.0f;
    }
    v_else
    {
        v = 0.0f;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp<SfpuType::less_than_equal_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > 0.0f)
    {
        v = 0.0f;
    }
    v_else
    {
        v = 1.0f;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_zero_comp_(std::uint32_t exponent_size_8)
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        apply_zero_comp<COMP_MODE>(v, exponent_size_8);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_zero_comp_int(sfpi::vInt& v);

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::equal_zero>(sfpi::vInt& v)
{
    v_if (v == 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::not_equal_zero>(sfpi::vInt& v)
{
    v_if (v == 0)
    {
        v = 0;
    }
    v_else
    {
        v = 1;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::less_than_zero>(sfpi::vInt& v)
{
    v_if (v < 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::greater_than_zero>(sfpi::vInt& v)
{
    v_if (v > 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::less_than_equal_zero>(sfpi::vInt& v)
{
    v_if (v <= 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <>
sfpi_inline void apply_zero_comp_int<SfpuType::greater_than_equal_zero>(sfpi::vInt& v)
{
    v_if (v >= 0)
    {
        v = 1;
    }
    v_else
    {
        v = 0;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_zero_comp_int_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt v = sfpi::dst_reg[0];
        apply_zero_comp_int<COMP_MODE>(v);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
sfpi_inline void apply_unary_int_comp(sfpi::vInt& v, int scalar, sfpi::vInt& out_val);

// a[i] != scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_ne>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v != scalar)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] == scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_eq>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v == scalar)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] > scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_gt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= 0 && s < 0)
    {
        out_val = 1;
    }
    v_elseif (v < 0 && s >= 0)
    {
        out_val = 0;
    }
    v_elseif (v > s)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] < scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_lt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= 0 && s < 0)
    {
        out_val = 0;
    }
    v_elseif (v < 0 && s >= 0)
    {
        out_val = 1;
    }
    v_elseif (v < s)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] >= scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_ge>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= 0 && s < 0)
    {
        out_val = 1;
    }
    v_elseif (v < 0 && s >= 0)
    {
        out_val = 0;
    }
    v_elseif (v >= s)
    {
        out_val = 1;
    }
    v_endif;
}

// a[i] <= scalar
template <>
sfpi_inline void apply_unary_int_comp<SfpuType::unary_le>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v < 0 && s >= 0)
    {
        out_val = 1;
    }
    v_elseif (v >= 0 && s < 0)
    {
        out_val = 0;
    }
    v_elseif (v <= s)
    {
        out_val = 1;
    }
    v_else
    {
        out_val = 0;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_comp_unary_int_(int scalar)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt v   = sfpi::dst_reg[0];
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
sfpi_inline void apply_unary_float_comp<SfpuType::unary_eq>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v == s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

// a[i] != scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_ne>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v == s)
    {
        out_val = 0.0f;
    }
    v_else
    {
        out_val = 1.0f;
    }
    v_endif;
}

// a[i] > scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_gt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v > s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

// a[i] < scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_lt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v < s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

// a[i] >= scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_ge>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v >= s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
    }
    v_endif;
}

// a[i] <= scalar
template <>
sfpi_inline void apply_unary_float_comp<SfpuType::unary_le>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v <= s)
    {
        out_val = 1.0f;
    }
    v_else
    {
        out_val = 0.0f;
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
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::vFloat val = 0.0f;

        apply_unary_float_comp<COMP_MODE>(v, s, val);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
