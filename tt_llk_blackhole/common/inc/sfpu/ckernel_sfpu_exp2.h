// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v = v * sfpi::vConstFloatPrgm0;
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en)
        {
            result = _sfpu_exp_fp32_accurate_(v);
        }
        else
        {
            result = _sfpu_exp_21f_bf16_<true>(v);
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    sfpi::vConstFloatPrgm0 = 0.6931471805f;
}

} // namespace ckernel::sfpu
