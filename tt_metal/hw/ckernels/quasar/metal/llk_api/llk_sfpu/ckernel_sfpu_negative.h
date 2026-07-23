// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Element-wise negate for floats: out = -x. Identical to the Blackhole kernel: the plain sfpi store
// writes through the default (no-increment) address mode and dst_reg++ advances the Dest write
// counter by SFP_DESTREG_STRIDE (== SFP_ROWS == 2 on Quasar), so only the shared SFPU init is needed
// (no op-specific ADDR_MOD). There is no approximate variant (sign flip is exact), so
// APPROXIMATION_MODE is accepted for ABI parity but ignored.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -val;
        sfpi::dst_reg++;
    }
}

// Element-wise negate for int32: out = -x (two's-complement negate), for the negative_tile_int32 path.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_int_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt val   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -val;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
