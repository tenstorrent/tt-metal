// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::vFloat res = 0.0f;
        // Quasar: sfpi::is_inf / is_nan / is_finite spelled out. SFPI lowers their nearby() compare to a CC-only
        // SFPIADD into read-only LREG8, which leaves the lane mask unchanged on Quasar.
        sfpi::vInt exp = sfpi::exexp(in, sfpi::ExponentMode::Biased);

        if constexpr (operation == SfpuType::isinf)
        {
            v_if (exp >= 255 && sfpi::exman(in) == 0)
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == SfpuType::isposinf)
        {
            v_if (sfpi::is_pos(in) && exp >= 255 && sfpi::exman(in) == 0)
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == SfpuType::isneginf)
        {
            v_if (sfpi::is_neg(in) && exp >= 255 && sfpi::exman(in) == 0)
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == SfpuType::isnan)
        {
            v_if (exp >= 255 && sfpi::exman(in) != 0)
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == SfpuType::isfinite)
        {
            v_if (exp < 255)
            {
                res = 1.0f;
            }
            v_endif;
        }

        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
