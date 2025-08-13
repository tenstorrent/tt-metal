// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu
{

template <typename T>
constexpr bool is_supported_threshold_type_v = std::is_same_v<T, float> || std::is_same_v<T, uint32_t>;

template <bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _calculate_threshold_(T threshold, T value)
{
    static_assert(is_supported_threshold_type_v<T>, "Type T must be either float or uint32_t");

    sfpi::vFloat v_threshold;
    sfpi::vFloat v_value;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
        v_value     = value;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        v_threshold = Converter::as_float(threshold);
        v_value     = Converter::as_float(value);
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];
        v_if (in <= v_threshold)
        {
            sfpi::dst_reg[0] = v_value;
        }
        v_endif;

        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
