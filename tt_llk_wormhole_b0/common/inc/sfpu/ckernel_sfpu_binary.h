// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

sfpi_inline sfpi::vFloat _calculate_sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow)
{
    sfpi::vFloat original_base = base;

    // Check for integer power
    sfpi::vInt pow_int       = float_to_int16(pow, 0); // int16 should be plenty, since large powers will approach 0/Inf
    sfpi::vFloat pow_rounded = int32_to_float(pow_int, 0);
    v_if (pow_rounded == pow)
    {
        // if pow is integer, set base to positive
        base = sfpi::setsgn(base, 0);
    }
    v_endif;

    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, 127); // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    sfpi::vInt exp = exexp(base);
    v_if (exp < 0)
    {
        exp = sfpi::setsgn(~exp + 1, 1);
    }
    v_endif;
    sfpi::vFloat expf = int32_to_float(exp, 0);

    // De-normalize to original range
    sfpi::vFloat vConstLn2  = 0.692871f;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if (base == 0.0f)
    { // Reload for register pressure
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    // Take exp(pow * log(base)) to produce base^pow
    sfpi::vFloat val = pow * log_result;

    // Force sign to 0 (make number positive)
    sfpi::vFloat result = _sfpu_exp_(sfpi::setsgn(val, 0));

    v_if (val < 0)
    {
        result = _sfpu_reciprocal_(result);
    }
    v_endif;

    // Check valid base range
    v_if (original_base < 0.0f)
    { // negative base
        // Check for integer power
        v_if (pow_rounded == pow)
        {
            // if pow is odd integer, set result to negative
            v_if (pow_int & 0x1)
            {
                result = sfpi::setsgn(result, 1);
            }
            v_endif;
        }
        v_else
        {
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const uint dst_offset)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr uint dst_tile_size = 32;
        sfpi::vFloat in0             = sfpi::dst_reg[0];
        sfpi::vFloat in1             = sfpi::dst_reg[dst_offset * dst_tile_size];
        sfpi::vFloat result          = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1;
        }
        else if constexpr (BINOP == BinaryOp::SUB)
        {
            result = in0 - in1;
        }
        else if constexpr (BINOP == BinaryOp::MUL)
        {
            result = in0 * in1;
        }
        else if constexpr (BINOP == BinaryOp::DIV)
        {
            v_if (in1 == 0)
            {
                v_if (in0 == 0)
                {
                    result = std::numeric_limits<float>::quiet_NaN();
                }
                v_else
                {
                    result = std::numeric_limits<float>::infinity();
                    result = sfpi::setsgn(result, in0);
                }
                v_endif;
            }
            v_elseif (in0 == in1)
            {
                result = sfpi::vConst1;
            }
            v_else
            {
                result = in0 * sfpi::setsgn(_sfpu_reciprocal_<4>(in1), in1);
            }
            v_endif;
        }
        else if constexpr (BINOP == BinaryOp::RSUB)
        {
            result = in1 - in0;
        }
        else if constexpr (BINOP == BinaryOp::POW)
        {
            result = _calculate_sfpu_binary_power_(in0, in1);
        }
        else if constexpr (BINOP == BinaryOp::XLOGY)
        {
            v_if ((in1 < 0.0f) || (in1 == nan))
            {
                result = nan;
            }
            v_else
            {
                sfpi::dst_reg[0] = in1;
                _calculate_log_body_<false>(0);
                result = sfpi::dst_reg[0] * in0;
            }
            v_endif;
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_reciprocal_<APPROXIMATION_MODE>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
}

} // namespace sfpu
} // namespace ckernel
