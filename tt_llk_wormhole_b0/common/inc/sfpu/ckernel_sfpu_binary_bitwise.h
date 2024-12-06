// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"
#include <limits.h>

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

enum {
    AND_BINARY = 0,
    OR_BINARY = 1,
    XOR_BINARY = 2,
};  // BITWISE_MODE

template <bool APPROXIMATION_MODE, int BITWISE_MODE, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_bitwise_(const uint dst_offset)
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(0,4,3,0);
        TT_SFPLOAD(1,4,3,dst_offset*dst_tile_size);

        if constexpr (BITWISE_MODE == AND_BINARY) {
            TTI_SFPAND(0,1,0,0);
        } else if constexpr (BITWISE_MODE == OR_BINARY) {
            TTI_SFPOR(0,1,0,0);
        } else if constexpr (BITWISE_MODE == XOR_BINARY) {
            TTI_SFPXOR(0,1,0,0);
        }

        TTI_SFPSTORE(0,4,3,0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
