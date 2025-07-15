// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_exp2_()
{
    const bool SCALE_EN                  = false; // Exp2 does not use scale.
    const bool SKIP_POSITIVE_CHECK       = false; // Exp2 does not skip positive check.
    const uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B;

    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // log(2) = 0.6931471805;
        v = v * 0.6931471805f;
        // exp = e^(v)
        sfpi::vFloat exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(v, exp_base_scale_factor);
        sfpi::dst_reg[0] = exp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_exp2_()
{
    const uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000;
    const bool FAST_APPROX               = false; // Exp2 does not use fast approximation.
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
}

} // namespace ckernel::sfpu
