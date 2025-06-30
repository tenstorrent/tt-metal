// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_is_fp16_zero.h"
#include "llk_sfpu_types.h"
#include "sfpi.h"

namespace
{
constexpr std::uint32_t ONE  = 1;
constexpr std::uint32_t ZERO = 0;
} // namespace

namespace ckernel
{
namespace sfpu
{

sfpi_inline void _calculate_comp_init_flag_(bool check, sfpi::vFloat& flag1, sfpi::vFloat& flag2, float init)
{
    flag1 = init;
    if (check)
    {
        flag2 = init;
    }
}

template <bool APPROXIMATION_MODE, bool invert_output, bool check_zero, bool second_check, bool is_less_than_equal_zero, int ITERATIONS>
inline void _calculate_comp_(const int iterations, uint exponent_size_8)
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
            v_if (_sfpu_is_fp16_zero_(v, exponent_size_8))
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
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(flag1) | sfpi::reinterpret<sfpi::vUInt>(flag2));
            }
            else
            {
                // greater_than_zero
                // flag1 = 0x3F80(1.0) if DST >= 0 else 0
                // flag2 = 0x3F80(1.0) if DST != 0 else 0
                // Do a bitwise And (flag1 & flag2) to get > condition.
                // flag2 >= 0 AND flag1 != 0 => DST is Greater than zero
                // Result will be either 0x0000(0.0) or 0x3F80(1.0)
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(flag1) & sfpi::reinterpret<sfpi::vUInt>(flag2));
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
inline void apply_zero_comp(sfpi::vFloat& v, uint exponent_size_8);

template <>
inline void apply_zero_comp<SfpuType::equal_zero>(sfpi::vFloat& v, uint exponent_size_8)
{
    v_if (_sfpu_is_fp16_zero_(v, exponent_size_8))
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::not_equal_zero>(sfpi::vFloat& v, uint exponent_size_8)
{
    v_if (_sfpu_is_fp16_zero_(v, exponent_size_8))
    {
        v = ZERO;
    }
    v_else
    {
        v = ONE;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::less_than_zero>(sfpi::vFloat& v, uint /*unused*/)
{
    v_if (v >= ZERO)
    {
        v = ZERO;
    }
    v_else
    {
        v = ONE;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::greater_than_equal_zero>(sfpi::vFloat& v, uint /*unused*/)
{
    v_if (v >= ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::greater_than_zero>(sfpi::vFloat& v, uint /*unused*/)
{
    v_if (v > ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp<SfpuType::less_than_equal_zero>(sfpi::vFloat& v, uint /*unused*/)
{
    v_if (v > ZERO)
    {
        v = ZERO;
    }
    v_else
    {
        v = ONE;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_zero_comp_(uint exponent_size_8)
{
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        apply_zero_comp<COMP_MODE>(v, exponent_size_8);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
inline void apply_zero_comp_int(sfpi::vInt& v);

template <>
inline void apply_zero_comp_int<SfpuType::equal_zero>(sfpi::vInt& v)
{
    v_if (v == ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::not_equal_zero>(sfpi::vInt& v)
{
    v_if (v == ZERO)
    {
        v = ZERO;
    }
    v_else
    {
        v = ONE;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::less_than_zero>(sfpi::vInt& v)
{
    v_if (v < ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::greater_than_zero>(sfpi::vInt& v)
{
    v_if (v > ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::less_than_equal_zero>(sfpi::vInt& v)
{
    v_if (v <= ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <>
inline void apply_zero_comp_int<SfpuType::greater_than_equal_zero>(sfpi::vInt& v)
{
    v_if (v >= ZERO)
    {
        v = ONE;
    }
    v_else
    {
        v = ZERO;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_zero_comp_int_()
{
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vInt v = sfpi::dst_reg[0];
        apply_zero_comp_int<COMP_MODE>(v);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
inline void apply_unary_int_comp(sfpi::vInt& v, int scalar, sfpi::vInt& out_val);

// a[i] != scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_ne>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v != scalar)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] == scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_eq>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    v_if (v == scalar)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] > scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_gt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= ZERO && s < ZERO)
    {
        out_val = ONE;
    }
    v_elseif (v < ZERO && s >= ZERO)
    {
        out_val = ZERO;
    }
    v_elseif (v > s)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] < scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_lt>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= ZERO && s < ZERO)
    {
        out_val = ZERO;
    }
    v_elseif (v < ZERO && s >= ZERO)
    {
        out_val = ONE;
    }
    v_elseif (v < s)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] >= scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_ge>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v >= ZERO && s < ZERO)
    {
        out_val = ONE;
    }
    v_elseif (v < ZERO && s >= ZERO)
    {
        out_val = ZERO;
    }
    v_elseif (v >= s)
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] <= scalar
template <>
inline void apply_unary_int_comp<SfpuType::unary_le>(sfpi::vInt& v, int scalar, sfpi::vInt& out_val)
{
    const sfpi::vInt s = scalar;
    v_if (v < ZERO && s >= ZERO)
    {
        out_val = ONE;
    }
    v_elseif (v >= ZERO && s < ZERO)
    {
        out_val = ZERO;
    }
    v_elseif (v <= s)
    {
        out_val = ONE;
    }
    v_else
    {
        out_val = ZERO;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_comp_unary_int_(int scalar)
{
#pragma GCC unroll 8
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vInt v   = sfpi::dst_reg[0];
        sfpi::vInt val = ZERO;

        apply_unary_int_comp<COMP_MODE>(v, scalar, val);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

template <SfpuType COMP_MODE>
inline void apply_unary_float_comp(sfpi::vFloat v, sfpi::vFloat scalar, sfpi::vFloat& out_val);

// a[i] == scalar
template <>
inline void apply_unary_float_comp<SfpuType::unary_eq>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v == s)
    {
        out_val = ONE;
    }
    v_else
    {
        out_val = ZERO;
    }
    v_endif;
}

// a[i] != scalar
template <>
inline void apply_unary_float_comp<SfpuType::unary_ne>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v == s)
    {
        out_val = ZERO;
    }
    v_else
    {
        out_val = ONE;
    }
    v_endif;
}

// a[i] > scalar
template <>
inline void apply_unary_float_comp<SfpuType::unary_gt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v > s)
    {
        out_val = ONE;
    }
    v_else
    {
        out_val = ZERO;
    }
    v_endif;
}

// a[i] < scalar
template <>
inline void apply_unary_float_comp<SfpuType::unary_lt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v < s)
    {
        out_val = ONE;
    }
    v_else
    {
        out_val = ZERO;
    }
    v_endif;
}

// a[i] >= scalar
template <>
inline void apply_unary_float_comp<SfpuType::unary_ge>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v >= s)
    {
        out_val = ONE;
    }
    v_else
    {
        out_val = ZERO;
    }
    v_endif;
}

// a[i] <= scalar
template <>
inline void apply_unary_float_comp<SfpuType::unary_le>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v <= s)
    {
        out_val = ONE;
    }
    v_else
    {
        out_val = ZERO;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_comp_unary_(uint value)
{
    const sfpi::vFloat s = value;

#pragma GCC unroll 8
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::vFloat val = ZERO;

        apply_unary_float_comp<COMP_MODE>(v, s, val);

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
