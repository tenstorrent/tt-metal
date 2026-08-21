// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `zeropad-fresh` coverage row (legacy
// tt-llk experimental ckernel_sfpu_zero_pad.h _zero_pad_tile_, corpus
// manifest class D-ABSENT — zero dispatch anywhere).  Mathematical
// definition: rows [VALID_ROWS, TOTAL_ROWS) of the Dst tile are set to +0.0
// (padding-region scrub before a reduction); rows [0, VALID_ROWS) pass
// through untouched.  Pure constant store — no arithmetic, bit-exact golden.
#include <cstdint>

namespace ckernel::sfpu
{

template <int VALID_ROWS, int TOTAL_ROWS>
__attribute__((noinline)) void calculate_zero_pad_fresh_cpp()
{
    static_assert(0 <= VALID_ROWS && VALID_ROWS <= TOTAL_ROWS, "zero_pad row split must partition the tile");
    for (int d = 0; d < VALID_ROWS; ++d)
    {
        sfpi::dst_reg++;
    }
    for (int d = VALID_ROWS; d < TOTAL_ROWS; ++d)
    {
        sfpi::dst_reg[0] = sfpi::vFloat(0.0f);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
