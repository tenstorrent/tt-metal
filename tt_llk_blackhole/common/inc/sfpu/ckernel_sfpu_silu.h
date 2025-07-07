// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel::sfpu
{

inline sfpi::vFloat _sigmoid_piecewise_linear_positive_(sfpi::vFloat val)
{
    sfpi::vFloat result = 0.0f;
    v_if (val >= 5.0f)
    {
        result = 1.0f;
    }
    v_elseif (val > 1.0f && val < 5.0f)
    {
        result = POLYVAL5<sfpi::vFloat>(0.00144462f, -0.01055479f, -0.01203685f, 0.24300185f, 0.50437757f, val);
    }
    v_else
    {
        result = 0.229f * val + 0.5f; // linear appx as y = 0.229x + 0.5
    }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_silu_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result;
        v_if (val < 0.0f)
        {
            result = 1.0f - _sigmoid_piecewise_linear_positive_(-val);
        }
        v_else
        {
            result = _sigmoid_piecewise_linear_positive_(val);
        }
        v_endif;
        sfpi::dst_reg[0] = val * result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
