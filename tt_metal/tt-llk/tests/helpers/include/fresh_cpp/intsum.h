// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `intsum-fresh` coverage row (metal
// ckernel_sfpu_int_sum.h calculate_sum_int_col / calculate_sum_int_row,
// corpus manifest class D-ABSENT — zero dispatch anywhere).  Mathematical
// definition (int32 strided in-tile reductions over the left face column /
// top face row, stated in 32-lane vector-row indices of one Dst tile):
//   COL (SUBOP 0): out[i]  = sum of in[i + j], j in {0,2,4,6,16,18,20,22},
//                  for i in {0,1}; every other row unchanged.
//   ROW (SUBOP 1): out[i]  = in[i] + in[i+1] + in[i+8] + in[i+9],
//                  for i in {0,2,4,6}; every other row unchanged.
// Exact int32 wraparound contract; indexed loads only, no Dst walk.
#include <cstdint>

namespace ckernel::sfpu
{

template <int SUBOP>
__attribute__((noinline)) void calculate_int_sum_fresh_cpp()
{
    if constexpr (SUBOP == 0)
    {
        for (int i = 0; i < 2; ++i)
        {
            sfpi::vInt acc = sfpi::dst_reg[i];
            for (int j = 2; j < 8; j += 2)
            {
                acc = acc + sfpi::vInt(sfpi::dst_reg[i + j]);
            }
            for (int j = 16; j < 24; j += 2)
            {
                acc = acc + sfpi::vInt(sfpi::dst_reg[i + j]);
            }
            sfpi::dst_reg[i] = acc;
        }
    }
    else
    {
        static_assert(SUBOP == 0 || SUBOP == 1, "int sum SUBOP is COL(0)/ROW(1)");
        for (int i = 0; i < 8; i += 2)
        {
            sfpi::vInt acc   = sfpi::dst_reg[i];
            acc              = acc + sfpi::vInt(sfpi::dst_reg[i + 1]);
            acc              = acc + sfpi::vInt(sfpi::dst_reg[i + 8]);
            acc              = acc + sfpi::vInt(sfpi::dst_reg[i + 9]);
            sfpi::dst_reg[i] = acc;
        }
    }
}

} // namespace ckernel::sfpu
