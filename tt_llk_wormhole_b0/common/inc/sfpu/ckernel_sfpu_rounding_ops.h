// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <climits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

inline sfpi::vInt _float_to_int32_(sfpi::vFloat in)
{
    sfpi::vInt result;
    sfpi::vInt exp = exexp(in); // extract exponent
    v_if (exp < 0)
    {
        result = 0;
    }
    v_elseif (exp > 30) // overflow occurs above this range
    {
        // set to int32 max value in case of overflow
        result = std::numeric_limits<int32_t>::max();
        // check sign
        v_if (in < 0)
        {
            result = sfpi::reinterpret<sfpi::vInt>(sfpi::setsgn(sfpi::reinterpret<sfpi::vFloat>(result), 1));
        }
        v_endif;
    }
    v_else
    {
        // extract mantissa
        sfpi::vInt man = exman8(in);
        // shift the mantissa by (23-exponent) to the right
        sfpi::vInt shift = exp - 23; // 23 is number of mantissa in float32
        man              = shft(sfpi::reinterpret<sfpi::vUInt>(man), shift);
        // check sign
        v_if (in < 0)
        {
            man = sfpi::reinterpret<sfpi::vInt>(sfpi::setsgn(sfpi::reinterpret<sfpi::vFloat>(man), 1));
        }
        v_endif;
        result = man;
    }
    v_endif;
    return result;
}

inline sfpi::vInt _float_to_int31_(sfpi::vFloat v)
{
    sfpi::vInt q = float_to_int16(v * 0x1p-15f, 0);
    sfpi::vInt r = float_to_int16(v - int32_to_float(q, 0) * 0x1p15f, 0);
    v_if (r < 0)
    {
        r = sfpi::setsgn(r, 0);
        q = (q << 15) - r;
    }
    v_else
    {
        q = (q << 15) + r;
    }
    v_endif;
    return q;
}

inline constexpr std::array<float, 84> PRECOMPUTED_POW10_TABLE = {
    1e-45F, 1e-44F, 1e-43F, 1e-42F, 1e-41F, 1e-40F, 1e-39F, 1e-38F, 1e-37F, 1e-36F, 1e-35F, 1e-34F, 1e-33F, 1e-32F, 1e-31F, 1e-30F, 1e-29F,
    1e-28F, 1e-27F, 1e-26F, 1e-25F, 1e-24F, 1e-23F, 1e-22F, 1e-21F, 1e-20F, 1e-19F, 1e-18F, 1e-17F, 1e-16F, 1e-15F, 1e-14F, 1e-13F, 1e-12F,
    1e-11F, 1e-10F, 1e-9F,  1e-8F,  1e-7F,  1e-6F,  1e-5F,  1e-4F,  1e-3F,  1e-2F,  1e-1F,  1e0F,   1e1F,   1e2F,   1e3F,   1e4F,   1e5F,
    1e6F,   1e7F,   1e8F,   1e9F,   1e10F,  1e11F,  1e12F,  1e13F,  1e14F,  1e15F,  1e16F,  1e17F,  1e18F,  1e19F,  1e20F,  1e21F,  1e22F,
    1e23F,  1e24F,  1e25F,  1e26F,  1e27F,  1e28F,  1e29F,  1e30F,  1e31F,  1e32F,  1e33F,  1e34F,  1e35F,  1e36F,  1e37F,  1e38F,
};

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool USE_FP32 = false>
inline void _calculate_floor_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v      = sfpi::dst_reg[0];
        sfpi::vFloat result = v;
        sfpi::vInt tmp;

        if constexpr (USE_FP32)
        {
            tmp = _float_to_int32_(result);
        }
        else
        {
            tmp = float_to_int16(result, 0);
        }

        result = int32_to_float(tmp, 0);

        v_if (result > v)
        {
            result = result - 1;
        }
        v_endif;

        if constexpr (!USE_FP32)
        {
            v_if (v <= SHRT_MIN || v >= SHRT_MAX)
            {
                result = v;
            }
            v_endif;
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool USE_FP32 = false>
inline void _calculate_ceil_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat result = sfpi::dst_reg[0];
        sfpi::vFloat v      = result;

        sfpi::vInt tmp;
        if constexpr (USE_FP32)
        {
            tmp = _float_to_int32_(result);
        }
        else
        {
            tmp = float_to_int16(result, 0);
        }

        result = int32_to_float(tmp, 0);

        v_if (result < v)
        {
            result = result + 1;
        }
        v_endif;

        if constexpr (!USE_FP32)
        {
            v_if (v <= SHRT_MIN || v >= SHRT_MAX)
            {
                result = v;
            }
            v_endif;
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool USE_FP32 = false, int ITERATIONS = 8>
inline void _calculate_trunc_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in     = sfpi::dst_reg[0];
        sfpi::vFloat result = in;
        sfpi::vInt tmp;

        v_if (in < 0)
        {
            result = 0 - result;
        }
        v_endif;

        sfpi::vFloat v = result;

        if constexpr (USE_FP32)
        {
            tmp = _float_to_int32_(result);
        }
        else
        {
            tmp = float_to_int16(result, 0);
        }

        result = int32_to_float(tmp, 0);

        v_if (result > v)
        {
            result = result - 1;
        }
        v_endif;

        if constexpr (!USE_FP32)
        {
            v_if (v <= SHRT_MIN || v >= SHRT_MAX)
            {
                result = v;
            }
            v_endif;
        }

        v_if (in < 0)
        {
            result = 0 - result;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool USE_FP32 = false, int ITERATIONS = 8>
inline void _calculate_frac_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in     = sfpi::dst_reg[0];
        sfpi::vFloat result = in;
        sfpi::vInt tmp;

        v_if (in < 0)
        {
            result = 0 - result;
        }
        v_endif;

        sfpi::vFloat v = result;

        if constexpr (USE_FP32)
        {
            tmp = _float_to_int32_(result);
        }
        else
        {
            tmp = float_to_int16(result, 0);
        }

        result = int32_to_float(tmp, 0);

        v_if (result > v)
        {
            result = result - 1;
        }
        v_endif;

        if constexpr (!USE_FP32)
        {
            v_if (v <= SHRT_MIN || v >= SHRT_MAX)
            {
                result = v;
            }
            v_endif;
        }

        v_if (in < 0)
        {
            result = 0 - result;
        }
        v_endif;

        sfpi::dst_reg[0] = in - result;
        sfpi::dst_reg++;
    }
}

inline sfpi::vFloat _round_even_(sfpi::vFloat v)
{
    sfpi::vFloat result;
    v_if (sfpi::abs(v) < 0x1p30f)
    {
        result = int32_to_float(_float_to_int31_(v), 0);
        v_if (sfpi::abs(v - result) == 0.5F)
        {
            sfpi::vInt res = float_to_int16(result, 0);
            res            = res & 0xFFFE; // 0xFFFE = 1111 1111 1111 1110
            result         = sfpi::int32_to_float(res, 0);
        }
        v_endif;
    }
    v_else
    {
        result = v;
    }
    v_endif;
    return result;
}

template <bool APPROXIMATE, int ITERATIONS = 8>
void _calculate_round_(const int decimals)
{
    const auto exp10i = [](int n)
    {
        if (n > 38) // 38 is max decimal places float32 can store for positive values
        {
            return 1.0F / 0.0F;
        }

        if (n < -45) // 45 is max decimal places float32 can store for negative values
        {
            return 0.0F;
        }

        return PRECOMPUTED_POW10_TABLE[n + 45];
    };

    const sfpi::vFloat coeff   = exp10i(decimals);
    const sfpi::vFloat inverse = exp10i(-decimals);

    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v      = sfpi::dst_reg[0];
        sfpi::vFloat result = inverse * _round_even_(sfpi::abs(v) * coeff);
        result              = sfpi::setsgn(result, v);
        sfpi::dst_reg[0]    = result;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
