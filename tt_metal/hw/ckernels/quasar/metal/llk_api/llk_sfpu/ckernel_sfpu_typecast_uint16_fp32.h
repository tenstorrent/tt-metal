// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
// Calculates Typecast for number of rows of output SFPU ops (Quasar = 2 rows)
//
// The store goes through ADDR_MOD_6, which post-increments Dest by SFP_ROWS, so the loop needs no
// separate increment. Requires init_typecast to have programmed ADDR_MOD_6.
inline void _calculate_typecast_uint16_to_fp32_rows() {
    const sfpi::vUInt16 value = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();

    sfpi::dst_reg[0].mode<sfpi::DataLayout::F32>(ADDR_MOD_6) =
        sfpi::convert<sfpi::vFloat>(value, sfpi::RoundMode::NearestEven);
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_typecast_uint16_to_fp32_rows();
    }
}

}  // namespace sfpu
}  // namespace ckernel
