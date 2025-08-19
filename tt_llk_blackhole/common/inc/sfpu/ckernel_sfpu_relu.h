// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

template <typename T>
constexpr bool is_supported_relu_type_v = std::is_same_v<T, float> || std::is_same_v<T, uint32_t>;

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, uint slope)
{
    sfpi::vFloat s = Converter::as_float(slope);

#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if (v < 0.0f)
        {
            v *= s;
        }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold)
{
    sfpi::vFloat result = val;
    v_if (result > threshold)
    {
        result = threshold;
    }
    v_endif;
    v_if (result < 0.0f)
    {
        result = 0.0f;
    }
    v_endif;
    return result;
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[0];
        v_if (result > threshold)
        {
            result = threshold;
        }
        v_endif;
        v_if (result < 0)
        {
            result = 0;
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(Converter::as_float(threshold));
        }
        else
        {
            v_threshold = Converter::as_float(threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType a = sfpi::dst_reg[0];
        v_if (a < threshold)
        {
            sfpi::dst_reg[0] = threshold;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(threshold);
        }
        else
        {
            v_threshold = Converter::as_float(threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

} // namespace sfpu
} // namespace ckernel
